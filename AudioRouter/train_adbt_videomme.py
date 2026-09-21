"""Train the AudioRouter adapter from an external MCQA manifest.

Every row in the supplied manifest is training data.  This module does not
create train/test splits or run held-out inference; evaluation remains in the
standalone inference and evaluation entry points.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import (
    AVFeatureQueue,
    AudioConditionedBottleneck,
    BEATsAudioEncoder,
    audio_visual_routing_contrastive_loss,
    batched_effective_rank,
    counterfactual_qa_margin_loss,
    counterfactual_teacher_kd_loss,
    gradient_activation_importance,
    teacher_routing_alignment_loss,
    teacher_visual_feature_targets,
    temporal_av_contrastive_loss,
    visual_feature_distillation_loss,
)
from AudioRouter.tasks.vision import VisionTask


LOGGER = logging.getLogger("train_adbt_videomme")


def tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    """Return cheap scalar diagnostics for a tensor without retaining a graph."""
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "nonfinite": int((~finite).sum().detach().cpu()),
        "numel": detached.numel(),
    }
    if finite.any():
        valid = detached[finite].float()
        stats.update(
            {
                "min": float(valid.min().cpu()),
                "max": float(valid.max().cpu()),
                "mean": float(valid.mean().cpu()),
                "std": float(valid.std(unbiased=False).cpu()),
                "max_abs": float(valid.abs().max().cpu()),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "max_abs": None})
    return stats


def phase41_parameter_groups(adbt: AudioConditionedBottleneck):
    """Return the disjoint parameter groups requested by Phase 4.1 diagnostics."""
    generator = adbt.query_generator
    groups = {
        "query_tokens": [generator.query_tokens],
        "audio_projection": list(generator.audio_projection.parameters()),
        "audio_cross_attention": list(generator.audio_cross_attention.parameters()),
        "time_projection": list(generator.time_projection.parameters()),
        "visual_key": list(adbt.visual_key.parameters()),
    }
    if hasattr(generator, "slot_mlp"):
        groups["slot_mlp"] = list(generator.slot_mlp.parameters())
    return groups


@torch.no_grad()
def parameter_group_grad_norms(groups) -> Dict[str, float]:
    """Compute true global L2 gradient norms for named parameter groups."""
    norms = {}
    for name, parameters in groups.items():
        squared = sum(
            float(parameter.grad.detach().float().square().sum().cpu())
            for parameter in parameters
            if parameter.grad is not None
        )
        norms[name] = math.sqrt(squared)
    return norms


@torch.no_grad()
def snapshot_parameter_groups(groups):
    """Snapshot small trainable groups only on diagnostic optimizer steps."""
    return {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }


@torch.no_grad()
def parameter_group_update_norms(groups, snapshots) -> Dict[str, float]:
    """Measure the exact L2 parameter displacement made by optimizer.step()."""
    norms = {}
    for name, parameters in groups.items():
        squared = sum(
            float((parameter.detach() - before).float().square().sum().cpu())
            for parameter, before in zip(parameters, snapshots[name])
        )
        norms[name] = math.sqrt(squared)
    return norms


def save_debug_state(
    args,
    adbt,
    optimizer,
    epoch: int,
    video_index: int,
    global_step: int,
    video_id: str,
    reason: str,
    tensors: Dict[str, torch.Tensor],
):
    """Persist the first non-finite batch so a run can be diagnosed safely."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"debug_nonfinite_step_{global_step}.pt"
    debug_tensors = {
        name: value.detach().float().cpu()
        for name, value in tensors.items()
        if torch.is_tensor(value)
    }
    payload = {
        "reason": reason,
        "epoch": epoch,
        "video_index": video_index,
        "global_step": global_step,
        "video_id": video_id,
        "tensor_stats": {name: tensor_stats(value) for name, value in tensors.items() if torch.is_tensor(value)},
        "tensors": debug_tensors,
        "audio_bottleneck": {name: value.detach().cpu() for name, value in adbt.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)
    LOGGER.error(
        "saved non-finite debug state: reason=%s epoch=%d video=%d (%s) step=%d path=%s",
        reason,
        epoch,
        video_index,
        video_id,
        global_step,
        path,
    )
    return path


def require_finite(
    value: torch.Tensor,
    name: str,
    args,
    adbt,
    optimizer,
    epoch: int,
    video_index: int,
    global_step: int,
    video_id: str,
    tensors: Dict[str, torch.Tensor],
):
    """Stop before backward/optimizer.step can propagate NaN parameters."""
    if torch.isfinite(value).all():
        return
    save_debug_state(
        args,
        adbt,
        optimizer,
        epoch,
        video_index,
        global_step,
        video_id,
        reason=f"nonfinite_{name}",
        tensors={**tensors, name: value},
    )
    raise RuntimeError(
        f"Non-finite {name} detected at epoch={epoch} video={video_index} "
        f"({video_id}) step={global_step}; debug state was saved."
    )


def init_wandb(args, train_rows):
    """Create an optional W&B run without making W&B a training dependency."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested (--wandb), but wandb is not installed. "
            "Install it with `pip install wandb` or remove --wandb."
        ) from exc

    config = vars(args).copy()
    config.update(
        {
            "dataset": args.dataset_name,
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "train_rows": len(train_rows),
            "train_videos": len({str(row["videoID"]) for row in train_rows}),
            "gpu_visible": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
        }
    )
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
        "name": args.wandb_run_name or None,
        "id": args.wandb_run_id or None,
        "resume": "allow",
        "mode": args.wandb_mode,
        "config": config,
    }
    # W&B does not accept None for every optional argument on all versions.
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    run = wandb.init(**init_kwargs)
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("adbt_latent/*", step_metric="train/step")
    run.define_metric("adbt_attention/*", step_metric="train/step")
    run.define_metric("adbt_query/*", step_metric="train/step")
    run.define_metric("adbt_qk/*", step_metric="train/step")
    run.define_metric("adbt_rank/*", step_metric="train/step")
    run.define_metric("adbt_grad/*", step_metric="train/step")
    run.define_metric("adbt_update/*", step_metric="train/step")
    run.define_metric("phase5/*", step_metric="train/step")
    run.define_metric("phase5_1/*", step_metric="train/step")
    run.summary["train_rows"] = len(train_rows)
    run.summary["train_videos"] = len({str(row["videoID"]) for row in train_rows})
    return run


def load_training_manifest(manifest_file: Path) -> tuple[str, List[Dict]]:
    """Load and validate an external four-option MCQA training manifest."""
    if not manifest_file.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("training manifest root must be a JSON object")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training manifest must contain a non-empty 'rows' list")

    required = {"videoID", "video_path", "question", "options", "answer"}
    normalized_rows = []
    paths_by_video = {}
    for index, source_row in enumerate(rows):
        if not isinstance(source_row, dict):
            raise ValueError(f"manifest row {index} must be a JSON object")
        missing = required - set(source_row)
        if missing:
            raise ValueError(f"manifest row {index} is missing fields: {sorted(missing)}")

        row = dict(source_row)
        video_id = str(row["videoID"]).strip()
        if not video_id:
            raise ValueError(f"manifest row {index} has an empty videoID")
        media_path = Path(str(row["video_path"])).expanduser()
        if not media_path.is_absolute():
            media_path = manifest_file.parent / media_path
        media_path = media_path.resolve()
        if not media_path.is_file():
            raise FileNotFoundError(
                f"manifest row {index} video does not exist: {media_path}"
            )
        previous_path = paths_by_video.setdefault(video_id, media_path)
        if previous_path != media_path:
            raise ValueError(f"videoID {video_id!r} maps to multiple video paths")

        question = str(row["question"]).strip()
        if not question:
            raise ValueError(f"manifest row {index} has an empty question")
        options = row["options"]
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"manifest row {index} must contain exactly four options")
        options = [str(option).strip() for option in options]
        if any(not option for option in options):
            raise ValueError(f"manifest row {index} contains an empty option")
        answer = str(row["answer"]).strip().upper()
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError(f"manifest row {index} answer must be one of A, B, C, D")
        query_time = row.get("query_time_seconds")
        if query_time is not None:
            try:
                query_time = float(query_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"manifest row {index} query_time_seconds must be numeric or null"
                ) from exc
            if not math.isfinite(query_time) or query_time <= 0:
                raise ValueError(
                    f"manifest row {index} query_time_seconds must be positive and finite"
                )

        row.update(
            {
                "videoID": video_id,
                "video_path": str(media_path),
                "question": question,
                "options": options,
                "answer": answer,
                "query_time_seconds": query_time,
            }
        )
        normalized_rows.append(row)

    dataset_name = str(manifest.get("dataset_name") or manifest_file.stem).strip()
    return dataset_name, normalized_rows


def sample_video(
    video_path: str,
    fps="auto",
    max_frames: int = 0,
    end_time_seconds: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    source_fps = float(reader.get_avg_fps())
    full_duration = total / source_fps
    duration = full_duration
    if end_time_seconds is not None:
        if end_time_seconds <= 0:
            raise ValueError(f"end_time_seconds must be positive, got {end_time_seconds}")
        duration = min(full_duration, float(end_time_seconds))
    if isinstance(fps, str) and fps.startswith("auto"):
        scale_text = fps[len("auto") :]
        scale = float(scale_text) if scale_text else 1.0
        if scale <= 0:
            raise ValueError(f"FPS scale must be positive, got {fps}")
        actual_fps = (0.2 if duration > 1800 else 0.5) * scale
    else:
        actual_fps = float(fps)
    target = max(1, min(total, math.ceil(duration * actual_fps)))
    if max_frames > 0:
        target = min(target, max_frames)
    last_index = min(total - 1, max(0, math.floor(duration * source_fps) - 1))
    indices = np.linspace(0, last_index, target, dtype=int)
    return reader.get_batch(indices).asnumpy(), indices.astype(np.float32) / source_fps


def sample_video_for_rows(
    video_path: str,
    rows: List[Dict],
    fps="auto",
    max_frames: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Decode the union of exact per-question causal sampling grids.

    Each returned position array indexes the union and therefore reconstructs
    exactly the frames that would be sampled for that row alone. This lets the
    frozen vision/audio frontends run once per video without giving an early
    question frames from a later timestamp.
    """
    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    source_fps = float(reader.get_avg_fps())
    full_duration = total / source_fps
    grids = []
    for row in rows:
        query_time = row.get("query_time_seconds")
        duration = (
            full_duration
            if query_time is None
            else min(full_duration, float(query_time))
        )
        if duration <= 0:
            raise ValueError(f"non-positive sampling duration for row {row}")
        if isinstance(fps, str) and fps.startswith("auto"):
            scale_text = fps[len("auto") :]
            scale = float(scale_text) if scale_text else 1.0
            if scale <= 0:
                raise ValueError(f"FPS scale must be positive, got {fps}")
            actual_fps = (0.2 if duration > 1800 else 0.5) * scale
        else:
            actual_fps = float(fps)
        target = max(1, min(total, math.ceil(duration * actual_fps)))
        if max_frames > 0:
            target = min(target, max_frames)
        last_index = min(total - 1, max(0, math.floor(duration * source_fps) - 1))
        grids.append(np.linspace(0, last_index, target, dtype=np.int64))

    union = np.unique(np.concatenate(grids))
    lookup = {int(frame_index): position for position, frame_index in enumerate(union)}
    row_positions = [
        np.asarray([lookup[int(frame_index)] for frame_index in grid], dtype=np.int64)
        for grid in grids
    ]
    frames = reader.get_batch(union).asnumpy()
    return frames, union.astype(np.float32) / source_fps, row_positions


def row_video_path(row: Dict) -> str:
    """Return the validated media path from an external training row."""
    return str(row["video_path"])


def rows_end_time(rows: List[Dict]) -> float | None:
    """Return the latest causal query boundary, or None for whole-video QA."""
    values = [
        float(row["query_time_seconds"])
        for row in rows
        if row.get("query_time_seconds") is not None
    ]
    return max(values) if values else None


def causal_prefix_length(timestamps, row: Dict) -> int:
    """Number of sampled frames available when this row's question arrives."""
    query_time = row.get("query_time_seconds")
    if query_time is None:
        return len(timestamps)
    values = np.asarray(timestamps, dtype=np.float64)
    return max(1, int(np.searchsorted(values, float(query_time) + 1e-6, side="right")))


def make_prompt(doc: Dict, tokenizer) -> torch.Tensor:
    option_prompt = (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    text = option_prompt + "\n" + str(doc["question"]) + "\n" + "\n".join(doc["options"])
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")


def build_inputs(model, tokenizer, prompt_ids: torch.Tensor, answer: str, latents: torch.Tensor):
    """Insert flattened visual latents at the image marker and label answer only."""
    if prompt_ids.dim() != 1:
        prompt_ids = prompt_ids.reshape(-1)
    image_positions = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
    if image_positions.numel() != 1:
        raise ValueError(f"Expected one image token, got ids={prompt_ids.tolist()}")
    image_position = int(image_positions.item())
    target = tokenizer.encode(str(answer).strip(), add_special_tokens=False)
    if not target:
        raise ValueError(f"Empty target for answer {answer!r}")
    target_ids = torch.tensor(target, dtype=torch.long, device=prompt_ids.device)
    full_ids = torch.cat([prompt_ids, target_ids])
    labels = torch.full_like(full_ids, -100)
    labels[-target_ids.numel() :] = target_ids

    embed_device = model.get_input_embeddings().weight.device
    embed_dtype = model.get_input_embeddings().weight.dtype
    prefix = full_ids[:image_position].to(embed_device)
    suffix = full_ids[image_position + 1 :].to(embed_device)
    prefix_embeds = model.get_input_embeddings()(prefix.unsqueeze(0))
    suffix_embeds = model.get_input_embeddings()(suffix.unsqueeze(0))
    latent_embeds = latents.reshape(1, -1, latents.shape[-1]).to(embed_device, dtype=embed_dtype)
    inputs_embeds = torch.cat([prefix_embeds, latent_embeds, suffix_embeds], dim=1)
    expanded_labels = torch.cat(
        [
            labels[:image_position].to(embed_device),
            torch.full((latent_embeds.shape[1],), -100, dtype=torch.long, device=embed_device),
            labels[image_position + 1 :].to(embed_device),
        ]
    ).unsqueeze(0)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=embed_device)
    return inputs_embeds, expanded_labels, attention_mask


def option_token_ids(tokenizer) -> torch.Tensor:
    """Return the single-token ids used by four-option MCQA answers."""
    ids = []
    for letter in "ABCD":
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"MCQA option {letter!r} is not one token: {encoded}"
            )
        ids.append(encoded[0])
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def frozen_teacher_option_logits(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    answer: str,
    full_visual_tokens: torch.Tensor,
    option_ids: torch.Tensor,
) -> torch.Tensor:
    """Return A/B/C/D logits from the frozen full-visual LLaVA teacher."""
    embeds, labels, mask = build_inputs(
        model, tokenizer, prompt_ids, answer, full_visual_tokens.detach()
    )
    outputs = model.model(
        inputs_embeds=embeds,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )
    shifted_labels = labels[:, 1:].to(outputs.last_hidden_state.device)
    answer_mask = shifted_labels.ne(-100)
    answer_hidden = outputs.last_hidden_state[:, :-1][answer_mask]
    if answer_hidden.shape[0] != 1:
        raise RuntimeError(
            "MCQA teacher expects exactly one answer token, got "
            f"{answer_hidden.shape[0]}"
        )
    logits = model.lm_head(answer_hidden)
    return logits[:, option_ids.to(logits.device)].float().detach()


def frozen_teacher_visual_importance(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    answer: str,
    full_visual_tokens: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute answer-conditioned Grad x Activation on native visual tokens.

    LLaVA parameters remain frozen.  Only the detached visual-token leaf has
    ``requires_grad=True``; the question and answer supervise the loss but are
    never inputs to the ADBT memory-construction forward path.
    """
    teacher_visual = full_visual_tokens.detach().float().requires_grad_(True)
    embeds, labels, mask = build_inputs(
        model, tokenizer, prompt_ids, answer, teacher_visual
    )
    outputs = model.model(
        inputs_embeds=embeds,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )
    shifted_labels = labels[:, 1:].to(outputs.last_hidden_state.device)
    answer_mask = shifted_labels.ne(-100)
    answer_hidden = outputs.last_hidden_state[:, :-1][answer_mask]
    answer_labels = shifted_labels[answer_mask]
    if answer_hidden.shape[0] != 1:
        raise RuntimeError(
            "MCQA routing teacher expects exactly one answer token, got "
            f"{answer_hidden.shape[0]}"
        )
    answer_logits = model.lm_head(answer_hidden)
    correct_logit = answer_logits.gather(1, answer_labels[:, None]).sum()
    # The installed LLaVA stack uses reentrant gradient checkpointing, which
    # is incompatible with torch.autograd.grad(inputs=...).  All model
    # parameters are frozen, so a scalar backward populates only this visual
    # leaf and is mathematically equivalent for Grad x Activation.
    correct_logit.backward()
    visual_gradients = teacher_visual.grad
    if visual_gradients is None:
        raise RuntimeError("full-visual teacher produced no visual gradient")
    importance = gradient_activation_importance(
        teacher_visual, visual_gradients, topk=topk
    )
    metrics = {
        "teacher_correct_logit": float(correct_logit.detach().cpu()),
        "teacher_visual_grad_norm": float(
            visual_gradients.detach().float().norm().cpu()
        ),
        "teacher_importance_nonzero": float(
            (importance > 0).float().sum(dim=-1).mean().cpu()
        ),
    }
    return importance, metrics


def resize_audio_frames(audio: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Chronologically resize cross-video audio along the frame axis."""
    if audio.dim() != 3:
        raise ValueError(f"audio must be [F,L,D], got {tuple(audio.shape)}")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if audio.shape[0] == target_frames:
        return audio.clone()
    indices = torch.linspace(0, audio.shape[0] - 1, target_frames).round().long()
    return audio.index_select(0, indices)


def counterfactual_audio(
    real_audio: torch.Tensor,
    mode: str,
    previous_video_audio: torch.Tensor | None,
    seed: int,
) -> Tuple[torch.Tensor, str]:
    """Construct a wrong-audio control without changing visual/time inputs."""
    if mode == "cycle":
        choices = ("shuffled", "stale", "crossvideo")
        mode = choices[seed % len(choices)]
    if mode == "crossvideo" and previous_video_audio is None:
        mode = "shuffled"
    if mode == "shuffled":
        if real_audio.shape[0] <= 1:
            return torch.zeros_like(real_audio), "zero_fallback"
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(real_audio.shape[0], generator=generator)
        return real_audio.index_select(0, permutation), mode
    if mode == "stale":
        if real_audio.shape[0] <= 1:
            return torch.zeros_like(real_audio), "zero_fallback"
        return torch.cat([torch.zeros_like(real_audio[:1]), real_audio[:-1]], 0), mode
    if mode == "crossvideo":
        return resize_audio_frames(previous_video_audio, real_audio.shape[0]), mode
    if mode == "zero":
        return torch.zeros_like(real_audio), mode
    raise ValueError(f"Unsupported counterfactual audio mode: {mode}")


def encode_visual(model, image_processor, frames: np.ndarray, batch_size: int):
    processed = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
    tower = model.get_vision_tower()
    projector = getattr(model.get_model(), "mm_projector", None)
    task = VisionTask(None, None)
    chunks = []
    with torch.no_grad():
        for start in range(0, processed.shape[0], batch_size):
            frame_list = list(processed[start : start + batch_size].unbind(0))
            features, _ = task.encode_vision_batch({"frames": frame_list, "grid_thw": None}, tower, projector)
            chunks.append(features)
    return torch.cat(chunks, dim=0)


def train(args):
    if args.hard_example_repeats < 1:
        raise ValueError("--hard-example-repeats must be at least 1")
    if args.latent_norm is None:
        args.latent_norm = "none" if args.value_mode == "native" else "rmsnorm"
    if args.value_mode == "native" and args.latent_norm != "none":
        raise ValueError("--value-mode native requires --latent-norm none")
    if args.architecture == "phase4" and args.value_mode != "native":
        raise ValueError("--architecture phase4 requires --value-mode native")
    if args.attention_temperature <= 0:
        raise ValueError("--attention-temperature must be > 0")
    if args.bottleneck_stage not in (1, 2, 3):
        raise ValueError("--bottleneck-stage must be 1, 2, or 3")
    if args.bottleneck_stage < 3 and args.training_objective != "phase4":
        raise ValueError(
            "Vision-only Stage 1/2 controls support only the pure-QA phase4 objective"
        )
    if args.training_objective == "av_pretrain" and args.gradient_accumulation_steps != 1:
        raise ValueError("AV pretraining currently requires gradient accumulation 1")
    if args.av_temperature <= 0:
        raise ValueError("--av-temperature must be > 0")
    if args.av_loss_weight < 0 or args.cf_loss_weight < 0:
        raise ValueError("Phase-5 loss weights must be non-negative")
    if min(
        args.qa_loss_weight,
        args.route_loss_weight,
        args.feature_loss_weight,
        args.av_routing_loss_weight,
    ) < 0:
        raise ValueError("Phase-5.1 loss weights must be non-negative")
    if args.training_objective == "phase5_1" and not any(
        weight > 0
        for weight in (
            args.qa_loss_weight,
            args.route_loss_weight,
            args.feature_loss_weight,
            args.av_routing_loss_weight,
        )
    ):
        raise ValueError("phase5_1 requires at least one non-zero loss weight")
    if args.av_routing_temperature <= 0:
        raise ValueError("--av-routing-temperature must be positive")
    if args.teacher_importance_topk < 0:
        raise ValueError("--teacher-importance-topk must be non-negative")
    if args.cf_margin < 0:
        raise ValueError("--cf-margin must be non-negative")
    if args.cf_kd_temperature <= 0:
        raise ValueError("--cf-kd-temperature must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dataset_name, train_rows = load_training_manifest(Path(args.dataset_manifest))
    if args.hard_example_result:
        hard_result = json.loads(
            Path(args.hard_example_result).read_text(encoding="utf-8")
        )
        if "temperature_results" in hard_result:
            temperature_key = str(args.hard_example_temperature)
            if temperature_key not in hard_result["temperature_results"]:
                available = sorted(hard_result["temperature_results"])
                raise ValueError(
                    f"hard-example temperature {temperature_key} not found; "
                    f"available={available}"
                )
            hard_records = hard_result["temperature_results"][temperature_key][
                "records"
            ]
        else:
            hard_records = hard_result["records"]
        hard_question_ids = {
            str(record["question_id"])
            for record in hard_records
            if not bool(record["option_correct"])
            and (
                args.hard_example_max_margin <= 0
                or float(record["option_top2_margin"])
                <= args.hard_example_max_margin
            )
        }
        train_rows = [
            row
            for row in train_rows
            if str(row.get("question_id")) in hard_question_ids
        ]
        if not train_rows:
            raise ValueError("hard-example filtering selected zero training rows")
        train_rows = train_rows * args.hard_example_repeats
        LOGGER.info(
            "hard-example curriculum source=%s temperature=%s max_margin=%s "
            "unique_questions=%d repeats=%d scheduled_rows=%d",
            args.hard_example_result,
            args.hard_example_temperature,
            args.hard_example_max_margin,
            len(hard_question_ids),
            args.hard_example_repeats,
            len(train_rows),
        )
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if not train_rows:
        raise ValueError("training dataset contains zero rows after filtering")
    train_video_count = len({str(row["videoID"]) for row in train_rows})
    LOGGER.info(
        "training dataset=%s manifest=%s rows=%d videos=%d",
        args.dataset_name,
        Path(args.dataset_manifest).resolve(),
        len(train_rows),
        train_video_count,
    )
    wandb_run = init_wandb(args, train_rows)

    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, model_name, device_map="auto", multimodal=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(model.model, "gradient_checkpointing_enable"):
        # Frozen decoder weights still need activations for dL/d(visual
        # latents). Checkpointing recomputes layer activations instead of
        # retaining the full 7B-model graph on GPU. Dropout is zero in this
        # checkpoint, so train/eval mode does not change stochastic behavior.
        model.model.gradient_checkpointing_enable()
        model.model.train()
    visual_dim = int(model.config.hidden_size)
    # Keep trainable adapter weights/optimizer state in fp32. The frozen
    # LLaVA and cached visual features remain fp16; this avoids AdamW fp16
    # overflow while preserving the inference tensor contract.
    adbt = AudioConditionedBottleneck(
        visual_dim=visual_dim,
        num_queries=args.num_queries,
        hidden_size=args.bottleneck_hidden,
        num_heads=args.num_heads,
        stage=args.bottleneck_stage,
        latent_norm=args.latent_norm,
        value_mode=args.value_mode,
        architecture=args.architecture,
        attention_temperature=args.attention_temperature,
    ).to(next(model.get_vision_tower().parameters()).device, dtype=torch.float32)
    audio_encoder = None
    if args.bottleneck_stage >= 2:
        audio_encoder = BEATsAudioEncoder(
            args.beats_checkpoint,
            batch_size=args.beats_batch_size,
            output_mode="temporal" if args.architecture == "phase4" else "mean",
        )
    else:
        LOGGER.info(
            "Vision-only Stage 1: BEATs is disabled and static learnable queries "
            "construct the visual bottleneck"
        )
    optimizer = torch.optim.AdamW(adbt.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.init_adapter:
        checkpoint = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        source_state = checkpoint.get("audio_bottleneck", checkpoint)
        target_state = adbt.state_dict()
        checkpoint_args = checkpoint.get("args", {})
        source_architecture = (
            checkpoint_args.get("architecture", "legacy")
            if isinstance(checkpoint_args, dict) else "legacy"
        )
        if source_architecture != args.architecture:
            raise ValueError(
                "--init-adapter architecture mismatch: "
                f"source={source_architecture} target={args.architecture}; "
                "Phase-4 must start from a Phase-4 checkpoint or fresh initialization"
            )
        source_value_mode = (
            checkpoint_args.get("value_mode", "projected")
            if isinstance(checkpoint_args, dict) else "projected"
        )
        if source_value_mode != args.value_mode:
            if source_value_mode == "projected" and args.value_mode == "native":
                dropped = sorted(set(source_state) - set(target_state))
                source_state = {
                    key: value for key, value in source_state.items() if key in target_state
                }
                LOGGER.info(
                    "migrating projected-value adapter to native-value mode; "
                    "loaded Audio Query Generator/visual Key and dropped %d legacy tensors",
                    len(dropped),
                )
            else:
                raise ValueError(
                    "unsupported --init-adapter value-mode migration: "
                    f"source={source_value_mode} target={args.value_mode}"
                )
        query_key = "query_generator.query_tokens"
        if query_key in source_state and source_state[query_key].shape != target_state[query_key].shape:
            source_queries = source_state[query_key]
            target_queries = target_state[query_key].clone()
            if (
                source_queries.ndim != 3
                or target_queries.ndim != 3
                or source_queries.shape[0] != target_queries.shape[0]
                or source_queries.shape[2] != target_queries.shape[2]
            ):
                raise ValueError(
                    "--init-adapter can resize only the query-slot dimension; "
                    f"source={tuple(source_queries.shape)} target={tuple(target_queries.shape)}"
                )
            copied_queries = min(source_queries.shape[1], target_queries.shape[1])
            target_queries[:, :copied_queries].copy_(source_queries[:, :copied_queries])
            source_state = dict(source_state)
            source_state[query_key] = target_queries
            LOGGER.info(
                "initialized %d/%d query slots from %s; %d new slots keep their independent random initialization",
                copied_queries,
                target_queries.shape[1],
                args.init_adapter,
                max(0, target_queries.shape[1] - copied_queries),
            )
        shape_mismatches = {
            key: (tuple(value.shape), tuple(target_state[key].shape))
            for key, value in source_state.items()
            if key in target_state and value.shape != target_state[key].shape
        }
        if shape_mismatches:
            raise ValueError(
                "--init-adapter is incompatible with the requested architecture; "
                f"shape mismatches={shape_mismatches}"
            )
        missing, unexpected = adbt.load_state_dict(source_state, strict=False)
        invalid_missing = [key for key in missing if not key.startswith("latent_norm.")]
        if invalid_missing or unexpected:
            raise ValueError(
                "invalid --init-adapter checkpoint: "
                f"missing={invalid_missing} unexpected={unexpected}"
            )
        LOGGER.info("initialized adapter weights from %s", args.init_adapter)

    resume_epoch, resume_video_idx, global_step = 0, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_architecture = (
            checkpoint_args.get("architecture", "legacy")
            if isinstance(checkpoint_args, dict) else "legacy"
        )
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                "--resume architecture mismatch: "
                f"checkpoint={checkpoint_architecture} requested={args.architecture}"
            )
        if args.architecture == "phase4" and float(
            checkpoint_args.get("attention_temperature", 0.2)
        ) != float(args.attention_temperature):
            raise ValueError(
                "--resume attention temperature does not match the checkpoint"
            )
        checkpoint_value_mode = (
            checkpoint_args.get("value_mode", "projected")
            if isinstance(checkpoint_args, dict) else "projected"
        )
        if checkpoint_value_mode != args.value_mode:
            raise ValueError(
                "--resume value mode does not match the requested architecture: "
                f"checkpoint={checkpoint_value_mode} requested={args.value_mode}; "
                "use --init-adapter for a supported architecture migration"
            )
        missing, unexpected = adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=False)
        if missing or unexpected:
            LOGGER.warning(
                "checkpoint adapter keys differ from current model (missing=%s unexpected=%s); "
                "new stability parameters keep their initialization",
                missing,
                unexpected,
            )
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if args.override_resume_learning_rate:
                    restored_lrs = [group["lr"] for group in optimizer.param_groups]
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate
                    LOGGER.info(
                        "overrode resumed optimizer learning rates %s -> %.8g",
                        restored_lrs,
                        args.learning_rate,
                    )
            except ValueError as exc:
                # Adding latent normalization/scale changes the parameter
                # group shape. Keep the adapter weights but start a fresh
                # optimizer rather than silently pairing states incorrectly.
                LOGGER.warning("could not restore optimizer state; using a fresh optimizer: %s", exc)
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_video_idx = int(checkpoint.get("video_idx", 0))
        global_step = int(checkpoint.get("global_step", 0))
        LOGGER.info("resuming from epoch=%d video=%d step=%d", resume_epoch, resume_video_idx, global_step)
    if args.training_objective == "av_pretrain":
        for parameter in adbt.parameters():
            parameter.requires_grad_(False)
        for parameter in adbt.query_generator.audio_projection.parameters():
            parameter.requires_grad_(True)
        for parameter in adbt.visual_key.parameters():
            parameter.requires_grad_(True)
        LOGGER.info(
            "Phase-5 Stage A: training only shared audio_projection and visual_key"
        )
    av_queue = AVFeatureQueue(args.av_queue_size)
    previous_video_audio = None
    video_option_token_ids = option_token_ids(tokenizer)
    optimizer.zero_grad(set_to_none=True)
    attention_collapse_streak = 0
    rows_by_video = {}
    for row in train_rows:
        rows_by_video.setdefault(str(row["videoID"]), []).append(row)
    skipped_videos = {str(value) for value in args.skip_video}
    unknown_skips = skipped_videos - set(rows_by_video)
    if unknown_skips:
        LOGGER.warning("requested skip videos are not in the training manifest: %s", sorted(unknown_skips))
    effective_skips = skipped_videos & set(rows_by_video)
    if effective_skips:
        LOGGER.warning(
            "explicitly skipping %d undecodable train videos (%d rows): %s",
            len(effective_skips),
            sum(len(rows_by_video[value]) for value in effective_skips),
            sorted(effective_skips),
        )
    total_train_videos = len(rows_by_video) - len(effective_skips)
    for epoch in range(resume_epoch, args.epochs):
        video_order = list(rows_by_video)
        random.Random(args.seed + epoch).shuffle(video_order)
        video_order = [value for value in video_order if value not in effective_skips]
        if epoch == resume_epoch and resume_video_idx:
            video_order = video_order[resume_video_idx:]
        video_offset = resume_video_idx if epoch == resume_epoch else 0
        for video_idx, current_video in enumerate(video_order):
            rows = rows_by_video[current_video]
            current_video_index = video_offset + video_idx + 1
            video_loss_sum = 0.0
            video_loss_count = 0
            path = row_video_path(rows[0])
            frames, timestamps, row_frame_positions = sample_video_for_rows(
                path, rows, max_frames=args.train_max_frames
            )
            visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
            audio = None
            audio_embeddings = None
            audio_timestamps = torch.as_tensor(
                timestamps, device=visual.device, dtype=torch.float32
            )
            if args.bottleneck_stage >= 2:
                audio = audio_encoder.encode_video(
                    path, timestamps, visual.device, ablation="real"
                )
                audio_embeddings = audio["embeddings"].to(
                    visual.device, dtype=torch.float32
                )
                audio_timestamps = audio["end_timestamps"].to(visual.device)
            adbt.to(device=visual.device, dtype=torch.float32)
            if args.training_objective == "av_pretrain":
                optimizer.zero_grad(set_to_none=True)
                av_result = temporal_av_contrastive_loss(
                    adbt,
                    audio_embeddings,
                    visual.float(),
                    audio_timestamps,
                    temperature=args.av_temperature,
                    min_temporal_offset=args.av_min_temporal_offset,
                    queue=av_queue,
                )
                require_finite(
                    av_result.loss,
                    "av_loss",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"visual": visual, "audio_embeddings": audio_embeddings},
                )
                av_result.loss.backward()
                clipped_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in adbt.parameters() if p.requires_grad],
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                )
                optimizer.step()
                global_step += 1
                av_queue.enqueue(
                    audio_embeddings.mean(dim=1), visual.detach().float().mean(dim=1)
                )
                if global_step == 1 or global_step % args.log_every == 0:
                    LOGGER.info(
                        "phase5_av epoch=%d video=%d/%d step=%d loss=%.5f "
                        "r1=%.4f r5=%.4f temporal=%.4f queue=%d grad=%.5f",
                        epoch + 1,
                        current_video_index,
                        total_train_videos,
                        global_step,
                        float(av_result.loss.detach().cpu()),
                        av_result.metrics["recall_at_1"],
                        av_result.metrics["recall_at_5"],
                        av_result.metrics["temporal_hard_accuracy"],
                        len(av_queue),
                        clipped_grad_norm,
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/epoch": epoch + 1,
                            "train/video_index": current_video_index,
                            "train/grad_norm": clipped_grad_norm,
                            "phase5/av_loss": float(av_result.loss.detach().cpu()),
                            "phase5/av_recall_at_1": av_result.metrics["recall_at_1"],
                            "phase5/av_recall_at_5": av_result.metrics["recall_at_5"],
                            "phase5/av_temporal_hard_accuracy": av_result.metrics[
                                "temporal_hard_accuracy"
                            ],
                            "phase5/av_queue_size": len(av_queue),
                        },
                        step=global_step,
                    )
                if (
                    args.save_every_videos > 0
                    and current_video_index % args.save_every_videos == 0
                ):
                    progress = (
                        Path(args.output_dir)
                        / f"adbt_epoch_{epoch + 1}_video_{current_video_index}.pt"
                    )
                    progress.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "audio_bottleneck": adbt.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "video_idx": current_video_index,
                            "global_step": global_step,
                            "args": vars(args),
                            "init_adapter": args.init_adapter,
                            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                            "dataset_name": args.dataset_name,
                        },
                        progress,
                    )
                    LOGGER.info("saved progress checkpoint %s", progress)
                del av_result, audio_embeddings, audio_timestamps, visual, audio, frames, timestamps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            # Reuse frozen visual/audio features for questions that share one
            # video, but run each language loss separately so multiple frozen
            # LLM activation graphs are never accumulated at once.
            for row_index, row in enumerate(rows):
                prompt_ids = make_prompt(row, tokenizer)
                teacher_importance = None
                teacher_importance_metrics = {}
                route_loss = visual.new_zeros((), dtype=torch.float32)
                feature_loss = visual.new_zeros((), dtype=torch.float32)
                av_routing_loss = visual.new_zeros((), dtype=torch.float32)
                route_metrics = {}
                feature_metrics = {}
                av_routing_metrics = {}
                teacher_feature_target = None
                use_route_distillation = (
                    args.training_objective == "phase5_1"
                    and args.route_loss_weight > 0
                )
                use_feature_distillation = (
                    args.training_objective == "phase5_1"
                    and args.feature_loss_weight > 0
                )
                use_av_routing = (
                    args.training_objective == "phase5_1"
                    and args.av_routing_loss_weight > 0
                )
                if use_route_distillation:
                    # Run the full-visual teacher first and release its decoder
                    # graph before the student graph is built.
                    teacher_importance, teacher_importance_metrics = (
                        frozen_teacher_visual_importance(
                            model,
                            tokenizer,
                            prompt_ids,
                            row["answer"],
                            visual,
                            topk=args.teacher_importance_topk,
                        )
                    )
                if use_feature_distillation:
                    teacher_feature_target = teacher_visual_feature_targets(
                        visual, args.num_queries
                    )
                av_loss = visual.new_zeros((), dtype=torch.float32)
                av_metrics = {}
                av_result = None
                if args.training_objective == "phase5" and args.av_loss_weight > 0:
                    av_result = temporal_av_contrastive_loss(
                        adbt,
                        audio_embeddings,
                        visual.float(),
                        audio_timestamps,
                        temperature=args.av_temperature,
                        min_temporal_offset=args.av_min_temporal_offset,
                        queue=av_queue,
                    )
                    av_loss = av_result.loss
                    av_metrics = av_result.metrics
                latents, attention = adbt(
                    visual.float(),
                    audio_embeddings=audio_embeddings,
                    end_timestamps=audio_timestamps,
                )
                positions = torch.as_tensor(
                    row_frame_positions[row_index], device=latents.device
                )
                latents = latents.index_select(0, positions)
                attention = attention.index_select(0, positions)
                latent_stats = tensor_stats(latents)
                adbt_stats = dict(adbt.last_diagnostics)
                attention_collapsed = (
                    adbt_stats.get("effective_support_mean", float("inf")) < 1.5
                    and adbt_stats.get("unique_argmax_mean", float("inf")) <= 2
                )
                queries_collapsed = (
                    adbt_stats.get("query_cosine_mean", -1.0) > 0.98
                )
                if attention_collapsed or queries_collapsed:
                    attention_collapse_streak += 1
                    if attention_collapse_streak == 10 or attention_collapse_streak % 100 == 0:
                        LOGGER.warning(
                            "ADBT collapse warning: streak=%d step=%d support=%.4f "
                            "unique_argmax=%.2f query_cosine=%.6f gap=%.4f",
                            attention_collapse_streak,
                            global_step,
                            adbt_stats.get("effective_support_mean", float("nan")),
                            adbt_stats.get("unique_argmax_mean", float("nan")),
                            adbt_stats.get("query_cosine_mean", float("nan")),
                            adbt_stats.get("top1_minus_top2_mean", float("nan")),
                        )
                else:
                    attention_collapse_streak = 0
                require_finite(
                    latents,
                    "latent",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={
                        "visual": visual,
                        **(
                            {"audio_embeddings": audio_embeddings}
                            if audio_embeddings is not None
                            else {}
                        ),
                    },
                )
                if use_route_distillation:
                    route_loss, route_metrics = teacher_routing_alignment_loss(
                        teacher_importance, attention
                    )
                if use_feature_distillation:
                    feature_loss, feature_metrics = (
                        visual_feature_distillation_loss(
                            latents, teacher_feature_target
                        )
                    )
                if use_av_routing:
                    av_routing_loss, av_routing_metrics = (
                        audio_visual_routing_contrastive_loss(
                            adbt,
                            audio_embeddings,
                            latents,
                            audio_timestamps,
                            temperature=args.av_routing_temperature,
                            min_temporal_offset=args.av_min_temporal_offset,
                        )
                    )
                embeds, labels, mask = build_inputs(model, tokenizer, prompt_ids, row["answer"], latents)
                teacher_option_logits = None
                if (
                    args.training_objective == "phase5"
                    and args.cf_loss_weight > 0
                    and args.cf_objective == "teacher_kd"
                ):
                    teacher_option_logits = frozen_teacher_option_logits(
                        model,
                        tokenizer,
                        prompt_ids,
                        row["answer"],
                        visual,
                        video_option_token_ids,
                    )
                wrong_latents = None
                wrong_audio_embeddings = None
                wrong_audio_cpu = None
                wrong_embeds = None
                wrong_labels = None
                wrong_mask = None
                option_logits = None
                answer_index = None
                negative_mode = "disabled"
                if args.training_objective == "phase5" and args.cf_loss_weight > 0:
                    wrong_audio_cpu, negative_mode = counterfactual_audio(
                        audio["embeddings"],
                        args.cf_negative,
                        previous_video_audio,
                        seed=args.seed + global_step + row_index,
                    )
                    wrong_audio_embeddings = wrong_audio_cpu.to(
                        visual.device, dtype=torch.float32
                    )
                    wrong_latents, _ = adbt(
                        visual.float(),
                        audio_embeddings=wrong_audio_embeddings,
                        end_timestamps=audio_timestamps,
                    )
                    wrong_embeds, wrong_labels, wrong_mask = build_inputs(
                        model,
                        tokenizer,
                        prompt_ids,
                        row["answer"],
                        wrong_latents,
                    )
                    model_embeds = torch.cat([embeds, wrong_embeds], dim=0)
                    model_labels = torch.cat([labels, wrong_labels], dim=0)
                    model_mask = torch.cat([mask, wrong_mask], dim=0)
                else:
                    model_embeds = embeds
                    model_labels = labels
                    model_mask = mask
                # Avoid full CausalLM logits [sequence, 152k] (which can cost
                # >9 GiB in float32). Only the answer-token positions are
                # supervised, so run the frozen backbone and project those
                # final hidden states through lm_head.
                backbone_outputs = model.model(
                    inputs_embeds=model_embeds,
                    attention_mask=model_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = backbone_outputs.last_hidden_state
                require_finite(
                    hidden,
                    "hidden",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "inputs_embeds": embeds},
                )
                # Causal LM alignment: the hidden state at position t predicts
                # the token at t+1.  The answer label is therefore selected
                # from labels[:, 1:] while its predictor is hidden[:, :-1].
                shifted_labels = model_labels[:, 1:].to(hidden.device)
                shifted_hidden = hidden[:, :-1]
                answer_mask = shifted_labels.ne(-100)
                answer_hidden = shifted_hidden[answer_mask]
                answer_labels = shifted_labels[answer_mask]
                lm_head = model.lm_head
                all_answer_logits = lm_head(answer_hidden)
                require_finite(
                    all_answer_logits,
                    "answer_logits",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "hidden": hidden},
                )
                qa_loss = F.cross_entropy(
                    all_answer_logits[:1].float(), answer_labels[:1]
                )
                cf_loss = qa_loss.new_zeros(())
                cf_metrics = {
                    "real_margin": float("nan"),
                    "wrong_margin": float("nan"),
                    "teacher_kl_real": float("nan"),
                    "teacher_kl_wrong": float("nan"),
                    "teacher_kl_advantage": float("nan"),
                    "violation_rate": 0.0,
                }
                if wrong_latents is not None:
                    if all_answer_logits.shape[0] != 2:
                        raise RuntimeError(
                            "Phase-5 counterfactual training expects one answer token "
                            f"per branch, got {all_answer_logits.shape[0]}"
                        )
                    option_ids = video_option_token_ids.to(all_answer_logits.device)
                    option_logits = all_answer_logits[:, option_ids].float()
                    answer_index = torch.tensor(
                        ["ABCD".index(str(row["answer"]).strip())],
                        device=option_logits.device,
                    )
                    if args.cf_objective == "teacher_kd":
                        if teacher_option_logits is None:
                            raise RuntimeError("teacher logits were not constructed")
                        kd_loss, kd_metrics = counterfactual_teacher_kd_loss(
                            teacher_option_logits,
                            option_logits[:1],
                            option_logits[1:],
                            margin=args.cf_margin,
                            temperature=args.cf_kd_temperature,
                        )
                        cf_loss = kd_loss
                        cf_metrics.update(kd_metrics)
                    else:
                        margin_loss, margin_metrics = counterfactual_qa_margin_loss(
                            option_logits[:1],
                            option_logits[1:],
                            answer_index,
                            margin=args.cf_margin,
                        )
                        cf_loss = margin_loss
                        cf_metrics.update(margin_metrics)
                if args.training_objective == "phase5_1":
                    raw_loss = (
                        args.qa_loss_weight * qa_loss
                        + args.route_loss_weight * route_loss
                        + args.feature_loss_weight * feature_loss
                        + args.av_routing_loss_weight * av_routing_loss
                    )
                else:
                    raw_loss = (
                        qa_loss
                        + args.av_loss_weight * av_loss
                        + args.cf_loss_weight * cf_loss
                    )
                loss = raw_loss / args.gradient_accumulation_steps
                require_finite(
                    loss,
                    "loss",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "answer_logits": all_answer_logits},
                )
                loss.backward()
                raw_loss_value = float(raw_loss.detach().cpu())
                qa_loss_value = float(qa_loss.detach().cpu())
                av_loss_value = float(av_loss.detach().cpu())
                cf_loss_value = float(cf_loss.detach().cpu())
                route_loss_value = float(route_loss.detach().cpu())
                feature_loss_value = float(feature_loss.detach().cpu())
                av_routing_loss_value = float(av_routing_loss.detach().cpu())
                video_loss_sum += raw_loss_value
                video_loss_count += 1
                if global_step == 0:
                    grad_norm = sum(
                        float(parameter.grad.detach().float().norm().cpu())
                        for parameter in adbt.parameters() if parameter.grad is not None
                    )
                    if not math.isfinite(grad_norm) or grad_norm == 0.0:
                        raise RuntimeError(f"ADBT received no finite gradient (norm={grad_norm})")
                    LOGGER.info("first-step ADBT gradient sum-norm=%.9e", grad_norm)
                clipped_grad_norm = None
                next_step = global_step + 1
                optimizer_boundary = next_step % args.gradient_accumulation_steps == 0
                log_modules = (
                    args.module_log_every > 0
                    and next_step % args.module_log_every == 0
                    and optimizer_boundary
                )
                module_metrics = {}
                group_snapshots = None
                if log_modules:
                    parameter_groups = phase41_parameter_groups(adbt)
                    group_grad_norms = parameter_group_grad_norms(parameter_groups)
                    group_snapshots = snapshot_parameter_groups(parameter_groups)
                    module_metrics.update(
                        {
                            f"adbt_grad/{name}": value
                            for name, value in group_grad_norms.items()
                        }
                    )
                    audio_grad = math.sqrt(
                        group_grad_norms["audio_projection"] ** 2
                        + group_grad_norms["audio_cross_attention"] ** 2
                    )
                    module_metrics["adbt_grad/audio_to_visual_key_ratio"] = (
                        audio_grad / max(group_grad_norms["visual_key"], 1e-30)
                    )
                rank_metrics = {}
                if args.rank_log_every > 0 and next_step % args.rank_log_every == 0:
                    attention_rank = batched_effective_rank(attention)
                    latent_rank = batched_effective_rank(latents)
                    rank_metrics = {
                        "adbt_rank/attention_effective_mean": float(attention_rank.mean().cpu()),
                        "adbt_rank/attention_effective_min": float(attention_rank.min().cpu()),
                        "adbt_rank/attention_effective_max": float(attention_rank.max().cpu()),
                        "adbt_rank/latent_effective_mean": float(latent_rank.mean().cpu()),
                        "adbt_rank/latent_effective_min": float(latent_rank.min().cpu()),
                        "adbt_rank/latent_effective_max": float(latent_rank.max().cpu()),
                    }
                if optimizer_boundary:
                    try:
                        clipped_grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                adbt.parameters(),
                                args.max_grad_norm,
                                error_if_nonfinite=True,
                            )
                        )
                    except RuntimeError:
                        save_debug_state(
                            args,
                            adbt,
                            optimizer,
                            epoch,
                            current_video_index,
                            global_step,
                            current_video,
                            reason="nonfinite_gradient_norm",
                            tensors={"latent": latents, "answer_logits": all_answer_logits},
                        )
                        raise
                    optimizer.step()
                    if group_snapshots is not None:
                        group_update_norms = parameter_group_update_norms(
                            parameter_groups, group_snapshots
                        )
                        module_metrics.update(
                            {
                                f"adbt_update/{name}": value
                                for name, value in group_update_norms.items()
                            }
                        )
                        audio_update = math.sqrt(
                            group_update_norms["audio_projection"] ** 2
                            + group_update_norms["audio_cross_attention"] ** 2
                        )
                        module_metrics["adbt_update/audio_to_visual_key_ratio"] = (
                            audio_update / max(group_update_norms["visual_key"], 1e-30)
                        )
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None and global_step % args.wandb_log_every == 0:
                    metrics = {
                        "train/step": global_step,
                        "train/loss": raw_loss_value,
                        "train/loss_scaled": float(loss.detach().cpu()),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                        "train/video_index": video_offset + video_idx + 1,
                        "train/video_progress": current_video_index / max(1, total_train_videos),
                        "train/answer_tokens": int(answer_labels.numel()),
                        "adbt_latent/max": latent_stats["max"],
                        "adbt_latent/min": latent_stats["min"],
                        "adbt_latent/mean": latent_stats["mean"],
                        "adbt_latent/std": latent_stats["std"],
                        "adbt_latent/max_abs": latent_stats["max_abs"],
                        "adbt_latent/nonfinite": latent_stats["nonfinite"],
                        "adbt_attention/entropy": adbt_stats["attention_entropy_mean"],
                        "adbt_attention/effective_support": adbt_stats["effective_support_mean"],
                        "adbt_attention/unique_argmax": adbt_stats["unique_argmax_mean"],
                        "adbt_attention/top1_minus_top2": adbt_stats["top1_minus_top2_mean"],
                        "adbt_query/mean_offdiag_cosine": adbt_stats["query_cosine_mean"],
                        "adbt_query/max_cosine": adbt_stats["query_cosine_max"],
                        "adbt_qk/q_norm_mean": adbt_stats["q_norm_mean"],
                        "adbt_qk/q_norm_max": adbt_stats["q_norm_max"],
                        "adbt_qk/k_norm_mean": adbt_stats["k_norm_mean"],
                        "adbt_qk/k_norm_max": adbt_stats["k_norm_max"],
                        "adbt_qk/score_min": adbt_stats["score_min"],
                        "adbt_qk/score_max": adbt_stats["score_max"],
                        "adbt_qk/score_std": adbt_stats["score_std"],
                    }
                    if args.training_objective == "phase5":
                        metrics.update(
                            {
                                "phase5/qa_loss": qa_loss_value,
                                "phase5/av_loss": av_loss_value,
                                "phase5/cf_loss": cf_loss_value,
                                "phase5/cf_real_margin": cf_metrics["real_margin"],
                                "phase5/cf_wrong_margin": cf_metrics["wrong_margin"],
                                "phase5/cf_teacher_kl_real": cf_metrics[
                                    "teacher_kl_real"
                                ],
                                "phase5/cf_teacher_kl_wrong": cf_metrics[
                                    "teacher_kl_wrong"
                                ],
                                "phase5/cf_teacher_kl_advantage": cf_metrics[
                                    "teacher_kl_advantage"
                                ],
                                "phase5/cf_violation_rate": cf_metrics[
                                    "violation_rate"
                                ],
                                "phase5/negative_shuffled": float(
                                    negative_mode == "shuffled"
                                ),
                                "phase5/negative_stale": float(
                                    negative_mode == "stale"
                                ),
                                "phase5/negative_crossvideo": float(
                                    negative_mode == "crossvideo"
                                ),
                            }
                        )
                        if av_metrics:
                            metrics.update(
                                {
                                    "phase5/av_recall_at_1": av_metrics[
                                        "recall_at_1"
                                    ],
                                    "phase5/av_recall_at_5": av_metrics[
                                        "recall_at_5"
                                    ],
                                    "phase5/av_temporal_hard_accuracy": av_metrics[
                                        "temporal_hard_accuracy"
                                    ],
                                    "phase5/av_queue_size": av_metrics["queue_size"],
                                }
                            )
                    elif args.training_objective == "phase5_1":
                        metrics.update(
                            {
                                "phase5_1/qa_loss": qa_loss_value,
                                "phase5_1/route_loss": route_loss_value,
                                "phase5_1/feature_loss": feature_loss_value,
                                "phase5_1/av_routing_loss": av_routing_loss_value,
                                "phase5_1/teacher_correct_logit": teacher_importance_metrics.get("teacher_correct_logit", float("nan")),
                                "phase5_1/teacher_visual_grad_norm": teacher_importance_metrics.get("teacher_visual_grad_norm", float("nan")),
                                "phase5_1/teacher_importance_nonzero": teacher_importance_metrics.get("teacher_importance_nonzero", float("nan")),
                                "phase5_1/teacher_importance_peak": route_metrics.get("teacher_importance_peak", float("nan")),
                                "phase5_1/student_mass_on_teacher_top1": route_metrics.get("student_mass_on_teacher_top1", float("nan")),
                                "phase5_1/feature_cosine": feature_metrics.get("feature_cosine", float("nan")),
                                "phase5_1/av_routing_recall_at_1": av_routing_metrics.get("recall_at_1", float("nan")),
                                "phase5_1/av_routing_recall_at_5": av_routing_metrics.get("recall_at_5", float("nan")),
                                "phase5_1/av_routing_temporal_hard_accuracy": av_routing_metrics.get("temporal_hard_accuracy", float("nan")),
                                "phase5_1/av_routing_positive_similarity": av_routing_metrics.get("positive_similarity", float("nan")),
                            }
                        )
                    if clipped_grad_norm is not None:
                        metrics["train/grad_norm"] = clipped_grad_norm
                    metrics.update(rank_metrics)
                    metrics.update(module_metrics)
                    wandb_run.log(metrics, step=global_step)
                if (
                    args.training_objective == "phase5"
                    and global_step % args.log_every == 0
                ):
                    if args.cf_objective == "teacher_kd":
                        LOGGER.info(
                            "phase5_kd step=%d qa=%.5f av=%.5f cf=%.5f "
                            "kl_real=%.4f kl_wrong=%.4f advantage=%.4f "
                            "violation=%.2f negative=%s av_r1=%.4f av_r5=%.4f "
                            "temporal=%.4f",
                            global_step,
                            qa_loss_value,
                            av_loss_value,
                            cf_loss_value,
                            cf_metrics["teacher_kl_real"],
                            cf_metrics["teacher_kl_wrong"],
                            cf_metrics["teacher_kl_advantage"],
                            cf_metrics["violation_rate"],
                            negative_mode,
                            av_metrics.get("recall_at_1", float("nan")),
                            av_metrics.get("recall_at_5", float("nan")),
                            av_metrics.get("temporal_hard_accuracy", float("nan")),
                        )
                    else:
                        LOGGER.info(
                            "phase5 step=%d qa=%.5f av=%.5f cf=%.5f "
                            "real_margin=%.4f wrong_margin=%.4f violation=%.2f "
                            "negative=%s av_r1=%.4f av_r5=%.4f temporal=%.4f",
                            global_step,
                            qa_loss_value,
                            av_loss_value,
                            cf_loss_value,
                            cf_metrics["real_margin"],
                            cf_metrics["wrong_margin"],
                            cf_metrics["violation_rate"],
                            negative_mode,
                            av_metrics.get("recall_at_1", float("nan")),
                            av_metrics.get("recall_at_5", float("nan")),
                            av_metrics.get("temporal_hard_accuracy", float("nan")),
                        )
                elif (
                    args.training_objective == "phase5_1"
                    and global_step % args.log_every == 0
                ):
                    LOGGER.info(
                        "phase5.1 step=%d qa=%.5f route=%.5f feat=%.5f "
                        "av_route=%.5f feat_cos=%.4f teacher_grad=%.5f "
                        "teacher_topk=%.1f student_teacher_mass=%.5f "
                        "av_r1=%.4f av_r5=%.4f temporal=%.4f",
                        global_step,
                        qa_loss_value,
                        route_loss_value,
                        feature_loss_value,
                        av_routing_loss_value,
                        feature_metrics.get("feature_cosine", float("nan")),
                        teacher_importance_metrics.get("teacher_visual_grad_norm", float("nan")),
                        teacher_importance_metrics.get("teacher_importance_nonzero", float("nan")),
                        route_metrics.get("student_mass_on_teacher_top1", float("nan")),
                        av_routing_metrics.get("recall_at_1", float("nan")),
                        av_routing_metrics.get("recall_at_5", float("nan")),
                        av_routing_metrics.get("temporal_hard_accuracy", float("nan")),
                    )
                if rank_metrics or module_metrics:
                    LOGGER.info(
                        "phase4.1 step=%d P_rank=%.3f Z_rank=%.3f grad_audio/key=%.6f "
                        "update_audio/key=%.6f",
                        global_step,
                        rank_metrics.get("adbt_rank/attention_effective_mean", float("nan")),
                        rank_metrics.get("adbt_rank/latent_effective_mean", float("nan")),
                        module_metrics.get("adbt_grad/audio_to_visual_key_ratio", float("nan")),
                        module_metrics.get("adbt_update/audio_to_visual_key_ratio", float("nan")),
                    )
                    if (
                        wandb_run is not None
                        and global_step % args.wandb_log_every != 0
                    ):
                        wandb_run.log(
                            {"train/step": global_step, **rank_metrics, **module_metrics},
                            step=global_step,
                        )
            if args.training_objective == "phase5":
                av_queue.enqueue(
                    audio_embeddings.mean(dim=1), visual.detach().float().mean(dim=1)
                )
                previous_video_audio = audio["embeddings"].detach().float().cpu()
            if global_step % args.log_every == 0:
                mean_video_loss = video_loss_sum / max(1, video_loss_count)
                LOGGER.info(
                    "epoch=%d video=%d/%d step=%d loss=%.5f entropy=%.4f "
                    "support=%.3f qcos=%.6f unique=%.2f gap=%.4f",
                    epoch,
                    current_video_index,
                    total_train_videos,
                    global_step,
                    mean_video_loss,
                    adbt_stats["attention_entropy_mean"],
                    adbt_stats["effective_support_mean"],
                    adbt_stats["query_cosine_mean"],
                    adbt_stats["unique_argmax_mean"],
                    adbt_stats["top1_minus_top2_mean"],
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/video_loss": mean_video_loss,
                            "train/video_frames": int(frames.shape[0]),
                            "train/video_duration_seconds": float(timestamps[-1]) if len(timestamps) else 0.0,
                        },
                        step=global_step,
                    )
            overall_video_idx = video_offset + video_idx + 1
            if (
                args.save_every_videos > 0
                and overall_video_idx % args.save_every_videos == 0
            ):
                progress = Path(args.output_dir) / f"adbt_epoch_{epoch + 1}_video_{overall_video_idx}.pt"
                progress.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "audio_bottleneck": adbt.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "video_idx": overall_video_idx,
                    "global_step": global_step,
                    "args": vars(args),
                    "init_adapter": args.init_adapter,
                    "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                    "dataset_name": args.dataset_name,
                }, progress)
                LOGGER.info("saved progress checkpoint %s", progress)
        out = Path(args.output_dir) / f"adbt_epoch_{epoch + 1}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "audio_bottleneck": adbt.state_dict(),
            "epoch": epoch,
            "video_idx": len(video_order) + video_offset,
            "global_step": global_step,
            "args": vars(args),
            "init_adapter": args.init_adapter,
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "dataset_name": args.dataset_name,
        }, out)
        LOGGER.info("saved %s", out)
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            LOGGER.info(
                "CUDA peak allocated cuda:%d=%.1f MiB peak reserved=%.1f MiB",
                device_index,
                torch.cuda.max_memory_allocated(device_index) / (1024 ** 2),
                torch.cuda.max_memory_reserved(device_index) / (1024 ** 2),
            )
    if wandb_run is not None:
        wandb_run.summary["final_global_step"] = global_step
        wandb_run.summary["completed_epochs"] = args.epochs
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--beats-checkpoint", default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help=(
            "External MCQA training manifest. Every row is used for training and must "
            "contain videoID, video_path, question, four options, answer, and optionally "
            "query_time_seconds. Relative video paths are resolved from the manifest."
        ),
    )
    parser.add_argument("--output-dir", default="./results/adbt-checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--train-max-frames", type=int, default=32)
    parser.add_argument(
        "--hard-example-result",
        default="",
        help=(
            "Optional prior eval JSON used to retain only incorrectly predicted "
            "training questions. Intended for explicitly disclosed in-sample curricula."
        ),
    )
    parser.add_argument(
        "--hard-example-temperature",
        type=float,
        default=0.05,
        help="Temperature key to select from a multi-temperature eval JSON.",
    )
    parser.add_argument(
        "--hard-example-max-margin",
        type=float,
        default=0.0,
        help="Keep wrong examples with top-2 option margin at most this value; <=0 keeps all.",
    )
    parser.add_argument(
        "--hard-example-repeats",
        type=int,
        default=1,
        help="Repeat each selected hard-example row this many times per curriculum epoch.",
    )
    parser.add_argument(
        "--skip-video",
        action="append",
        default=[],
        help=(
            "Explicit training-manifest videoID to skip after a documented media-decode "
            "failure. May be repeated."
        ),
    )
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--bottleneck-hidden", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--bottleneck-stage",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "ADBT stage. Stage 1 is the vision-only static-query control and "
            "does not instantiate or run BEATs; Stage 3 is audio-conditioned."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("legacy", "phase4"),
        default="phase4",
        help="Audio-query/attention architecture version.",
    )
    parser.add_argument(
        "--attention-temperature",
        type=float,
        default=0.2,
        help="Fixed cosine-attention temperature used by Phase 4.",
    )
    parser.add_argument(
        "--value-mode",
        choices=("projected", "native"),
        default="native",
        help=(
            "projected uses the legacy learned Value/output path; native uses "
            "the Phase-3 M=softmax(QK^T)V_original path exactly."
        ),
    )
    parser.add_argument(
        "--latent-norm",
        choices=("rmsnorm", "layernorm", "none"),
        default=None,
        help=(
            "Normalization applied to projected-value latents. Defaults to rmsnorm "
            "for projected mode and none for native mode; native requires none."
        ),
    )
    parser.add_argument(
        "--training-objective",
        choices=("phase4", "av_pretrain", "phase5", "phase5_1"),
        default="phase4",
        help=(
            "phase4 keeps pure QA training; av_pretrain runs Phase-5 Stage A "
            "alignment only; phase5 runs QA + AV + counterfactual loss; "
            "phase5_1 runs QA + teacher routing + feature + AV-routing."
        ),
    )
    parser.add_argument("--av-loss-weight", type=float, default=0.02)
    parser.add_argument("--av-temperature", type=float, default=0.07)
    parser.add_argument("--av-min-temporal-offset", type=float, default=5.0)
    parser.add_argument("--av-queue-size", type=int, default=256)
    parser.add_argument("--cf-loss-weight", type=float, default=0.2)
    parser.add_argument("--cf-margin", type=float, default=0.2)
    parser.add_argument(
        "--cf-objective",
        choices=("qa_margin", "teacher_kd"),
        default="qa_margin",
        help=(
            "qa_margin ranks correct-answer margins; teacher_kd ranks KL "
            "distance to an online frozen full-visual LLaVA teacher."
        ),
    )
    parser.add_argument("--cf-kd-temperature", type=float, default=1.0)
    parser.add_argument(
        "--qa-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the supervised MCQA loss in phase5_1. "
            "Set to zero only for the literal objective-only ablations in "
            "docs/experience.md; the historical Phase-5.1 default remains 1."
        ),
    )
    parser.add_argument("--route-loss-weight", type=float, default=0.1)
    parser.add_argument("--feature-loss-weight", type=float, default=0.05)
    parser.add_argument("--av-routing-loss-weight", type=float, default=0.05)
    parser.add_argument("--teacher-importance-topk", type=int, default=64)
    parser.add_argument("--av-routing-temperature", type=float, default=0.07)
    parser.add_argument(
        "--cf-negative",
        choices=("cycle", "shuffled", "stale", "crossvideo", "zero"),
        default="cycle",
        help="Wrong-audio source for Phase-5 counterfactual QA-margin training.",
    )
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--rank-log-every",
        type=int,
        default=30,
        help="Log effective ranks of attention P and latent Z every N question steps; 0 disables it.",
    )
    parser.add_argument(
        "--module-log-every",
        type=int,
        default=30,
        help="Log per-module gradient and optimizer-update norms every N question steps; 0 disables it.",
    )
    parser.add_argument("--save-every-videos", type=int, default=20)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default="",
        help="Resume weights, optimizer, epoch, video index, and global step from a progress checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-adapter",
        default="",
        help=(
            "Initialize adapter weights without restoring training progress. If only --num-queries "
            "changes, existing query slots are copied and new slots retain independent initialization."
        ),
    )
    parser.add_argument(
        "--override-resume-learning-rate",
        action="store_true",
        help=(
            "After restoring optimizer state with --resume, replace each parameter-group "
            "learning rate with --learning-rate while preserving optimizer moments."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("WANDB_TRAIN", "0").lower() in {"1", "true", "yes", "on"},
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "AudioRouter"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_NAME", ""))
    parser.add_argument("--wandb-run-id", default=os.getenv("WANDB_RUN_ID", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
        help="W&B transport mode; use offline on a compute node without network access.",
    )
    parser.add_argument(
        "--wandb-log-every",
        type=int,
        default=1,
        help="Log per-question metrics every N optimizer steps (video summaries use --log-every).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train(parse_args())

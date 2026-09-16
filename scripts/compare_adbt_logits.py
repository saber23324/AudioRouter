#!/usr/bin/env python3
"""Compare direct-latent and OQM-retrieval option logits on VideoMME.

This is a read-only diagnostic: it loads one trained ADBT checkpoint and the
frozen LLaVA/BEATs backbones, then evaluates the same visual/audio latents by
the direct path or, when ``--paths all`` is selected, by three paths:

1. ``direct_insert``: insert all latent embeddings at the image placeholder;
2. ``full_latent_kv``: prefill all latent KV without OQM quantization/retrieval;
3. ``oqm_retrieval``: use the production AudioRouter OQM path.

The middle control keeps the OQM query prompt and system-prefix convention, so
``full_latent_kv`` versus ``oqm_retrieval`` isolates the memory backend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from datasets import load_dataset
from transformers.cache_utils import DynamicCache

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import batched_effective_rank
from AudioRouter.main import AudioRouter
from AudioRouter.models.llava.llava_AudioRouter import create_frame_generator_llava
from AudioRouter.modules.AudioRouter_context import AudioRouterContext
from AudioRouter.pipeline import AudioRouterPipeline
from AudioRouter.train_adbt_videomme import encode_visual, sample_video, video_path


SYSTEM_PROMPT_IDS = [
    151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
    151644, 872, 198,
]
LETTERS = ("A", "B", "C", "D")


def official_context(row: Dict) -> str:
    instruction = (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return (
        instruction
        + "\n"
        + str(row["question"])
        + "\n"
        + "\n".join(row["options"])
        + "\n\nAnswer with the option's letter from the given choices directly."
    )


def official_input_ids(row: Dict, tokenizer) -> torch.Tensor:
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + official_context(row))
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )


def option_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Expected one token for {letter!r}, got {encoded}")
        ids[letter] = int(encoded[0])
    return ids


def summarize_option_logits(logits: torch.Tensor, ids: Dict[str, int]) -> Dict:
    values = torch.tensor(
        [float(logits[ids[letter]].float().cpu()) for letter in LETTERS],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(values, dim=0)
    winner = LETTERS[int(values.argmax())]
    ordered = torch.sort(values, descending=True).values
    return {
        "prediction": winner,
        "margin": float(ordered[0] - ordered[1]),
        "logits": {letter: float(value) for letter, value in zip(LETTERS, values)},
        "option_probabilities": {
            letter: float(value) for letter, value in zip(LETTERS, probabilities)
        },
    }


def direct_insert_logits(model, prompt_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    prompt_ids = prompt_ids.reshape(-1)
    positions = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
    if positions.numel() != 1:
        raise RuntimeError(f"Expected one image marker, got {positions.tolist()}")
    marker = int(positions.item())
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    prefix = embedding(prompt_ids[:marker].to(device).unsqueeze(0))
    suffix = embedding(prompt_ids[marker + 1 :].to(device).unsqueeze(0))
    latent_embeds = latents.reshape(1, -1, latents.shape[-1]).to(device, dtype=dtype)
    inputs_embeds = torch.cat((prefix, latent_embeds, suffix), dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    AudioRouterContext.get_instance().clear_mode()
    with torch.inference_mode():
        output = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return model.lm_head(output.last_hidden_state[0, -1]).float().cpu()


def prefill_full_latent_kv(model, latent_batches: Iterable[torch.Tensor]) -> DynamicCache:
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    context = AudioRouterContext.get_instance()
    context.clear_mode()
    context.should_store_keys = False

    system_ids = torch.tensor([SYSTEM_PROMPT_IDS], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.model(
            inputs_embeds=embedding(system_ids),
            attention_mask=torch.ones_like(system_ids),
            position_ids=torch.arange(system_ids.shape[1], device=device).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        for latent_batch in latent_batches:
            latent_embeds = latent_batch.reshape(1, -1, latent_batch.shape[-1]).to(
                device=device, dtype=dtype
            )
            past_length = cache.get_seq_length()
            sequence_length = latent_embeds.shape[1]
            output = model.model(
                inputs_embeds=latent_embeds,
                attention_mask=torch.ones(
                    (1, past_length + sequence_length), dtype=torch.long, device=device
                ),
                position_ids=torch.arange(
                    past_length, past_length + sequence_length, device=device
                ).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
    return cache


def logits_with_past(model, tokenizer, query_task, prompt_ids, past_key_values) -> torch.Tensor:
    device = model.get_input_embeddings().weight.device
    formatted_ids = query_task._format_input_ids_for_generation(prompt_ids, tokenizer, device)
    inputs_embeds = model.get_input_embeddings()(formatted_ids)
    # Qwen2 updates a supplied DynamicCache in place during attention even
    # when this diagnostic does not request the returned cache.  Clone it so
    # one question cannot contaminate the next question's control path.
    query_cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(
        zip(past_key_values.key_cache, past_key_values.value_cache)
    ):
        query_cache.update(key.clone(), value.clone(), layer_idx)
    past_length = query_cache.get_seq_length()
    query_length = inputs_embeds.shape[1]
    with torch.inference_mode():
        output = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(
                (1, past_length + query_length), dtype=torch.long, device=device
            ),
            position_ids=torch.arange(
                past_length, past_length + query_length, device=device
            ).unsqueeze(0),
            past_key_values=query_cache,
            use_cache=False,
            return_dict=True,
        )
        return model.lm_head(output.last_hidden_state[0, -1]).float().cpu()


def fill_oqm(pipeline: AudioRouterPipeline, video_id: str, latent_batches: List[torch.Tensor]):
    pipeline.reset_AudioRouter_state()
    state = {}
    for batch_idx, latent_batch in enumerate(latent_batches):
        frame_info = {
            "batch_type": {
                "is_first": batch_idx == 0,
                "is_last": batch_idx == len(latent_batches) - 1,
                "batch_idx": batch_idx,
            },
            "local_batch_idx": batch_idx,
            "encoding_start_frame": None,
        }
        state = pipeline.AudioRouter_core.process_vision_batch(
            video_id,
            latent_batch,
            state,
            None,
            batch_idx,
            pipeline.model,
            frame_info,
        )
    pipeline.AudioRouter_state = state


def oqm_logits(model, tokenizer, pipeline, video_id: str, prompt_ids: torch.Tensor):
    query_task = pipeline.AudioRouter_core.query_task
    query_task.prepare_retrieval_context(video_id, prompt_ids, model, tokenizer)
    device = model.get_input_embeddings().weight.device
    prepared, _ = query_task._prepare_input_ids(prompt_ids, device)
    retrieval_ids = query_task._get_retrieval_input_ids(prepared, device)
    retrieved = query_task._perform_retrieval_forward(model, retrieval_ids)
    retrieved_length = int(retrieved.get_seq_length())
    logits = logits_with_past(model, tokenizer, query_task, prompt_ids, retrieved)
    return logits, retrieved_length


def select_rows(dataset, split: Dict, duration: str, video_id: str, num_videos: int):
    allowed = set(str(item) for item in split["test"])
    rows = [
        row for row in dataset
        if str(row["videoID"]) in allowed and str(row["duration"]) == duration
    ]
    if video_id:
        selected_ids = [video_id]
    else:
        selected_ids = []
        for row in rows:
            current = str(row["videoID"])
            if current not in selected_ids:
                selected_ids.append(current)
            if len(selected_ids) >= num_videos:
                break
    return selected_ids, {
        current: [row for row in rows if str(row["videoID"]) == current]
        for current in selected_ids
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-file", default="results/adbt_videomme_split.json")
    parser.add_argument("--duration", choices=("short", "medium", "long"), default="short")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32, help="0 uses official FPS without a cap")
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument(
        "--attention-temperature",
        type=float,
        default=None,
        help=(
            "Inference-only cosine temperature override. Supplying this flag "
            "also enables the explicit checkpoint override guard."
        ),
    )
    parser.add_argument(
        "--paths",
        choices=("all", "direct"),
        default="all",
        help="Use 'direct' to skip the CTR/OQM comparison for direct-backend diagnostics.",
    )
    parser.add_argument(
        "--skip-logits",
        action="store_true",
        help="Collect latent/attention/QK diagnostics without running question logits.",
    )
    parser.add_argument("--output", default="results/adbt-logit-comparison.json")
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--hf-cache", default="/home/yxd/.cache/huggingface")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["AUDIO_BOTTLENECK_CHECKPOINT"] = args.checkpoint
    if args.attention_temperature is not None:
        if args.attention_temperature <= 0:
            raise ValueError("--attention-temperature must be > 0")
        os.environ["AUDIO_BOTTLENECK_TEMPERATURE"] = str(
            args.attention_temperature
        )
        os.environ["AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE"] = "1"
    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
    )
    model.tokenizer = tokenizer
    model = AudioRouter(model, "llava")
    model.eval()

    dataset = load_dataset(
        "lmms-lab/Video-MME",
        cache_dir=args.hf_cache,
        download_mode="reuse_dataset_if_exists",
    )["test"]
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    selected_ids, rows_by_video = select_rows(
        dataset, split, args.duration, args.video_id, args.num_videos
    )
    if not selected_ids:
        raise RuntimeError("No matching held-out videos")

    ids = option_token_ids(tokenizer)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "duration": args.duration,
        "max_frames": args.max_frames,
        "attention_temperature": float(
            model.audio_bottleneck.attention_temperature
        ),
        "paths": args.paths,
        "skip_logits": args.skip_logits,
        "option_token_ids": ids,
        "videos": [],
    }
    stream_batch_size = int(model.AudioRouter_config.get("streaming_encoder_batch_size"))

    for current_video in selected_ids:
        path = video_path(current_video)
        frames, timestamps = sample_video(path, fps="auto", max_frames=args.max_frames)
        visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
        audio = model.encode_audio_from_video(path, timestamps)
        phase4 = getattr(model.audio_bottleneck, "architecture", "legacy") == "phase4"
        bottleneck_dtype = torch.float32 if phase4 else visual.dtype
        bottleneck = model.audio_bottleneck.to(
            device=visual.device, dtype=bottleneck_dtype
        )
        latent_batches = []
        attention_batches = []
        q_norm_batches = []
        k_norm_batches = []
        score_batches = []
        top1_batches = []
        top2_batches = []
        query_cosine_batches = []
        attention_rank_batches = []
        latent_rank_batches = []
        with torch.inference_mode():
            for start in range(0, visual.shape[0], stream_batch_size):
                end = min(start + stream_batch_size, visual.shape[0])
                audio_embeddings = audio["embeddings"][start:end].to(
                    visual.device, dtype=bottleneck_dtype
                )
                end_timestamps = audio["end_timestamps"][start:end].to(visual.device)
                visual_batch = visual[start:end].to(dtype=bottleneck_dtype)
                latent, attention = bottleneck(
                    visual_batch,
                    audio_embeddings=audio_embeddings,
                    end_timestamps=end_timestamps,
                )
                latent_batches.append(latent)
                attention_batches.append(attention)

                # Recompute Q/K scores only for diagnostics. This follows the
                # exact Stage-3 equations but does not replace or alter the
                # latent returned by the production bottleneck above.
                queries = bottleneck.query_generator(
                    audio_embeddings, end_timestamps
                )
                visual_keys = bottleneck.visual_key(visual_batch)
                scores = bottleneck.compute_attention_scores(queries, visual_keys)
                top_scores = scores.float().topk(k=2, dim=-1).values
                query_norms = queries.float().norm(dim=-1)
                key_norms = visual_keys.float().norm(dim=-1)
                normalized_queries = queries.float() / query_norms.clamp_min(
                    1e-12
                ).unsqueeze(-1)
                query_cosine = torch.matmul(
                    normalized_queries, normalized_queries.transpose(-1, -2)
                )
                off_diagonal = ~torch.eye(
                    queries.shape[1], dtype=torch.bool, device=queries.device
                )
                q_norm_batches.append(query_norms.cpu())
                k_norm_batches.append(key_norms.cpu())
                score_batches.append(scores.detach().float().cpu())
                top1_batches.append(top_scores[..., 0].cpu())
                top2_batches.append(top_scores[..., 1].cpu())
                query_cosine_batches.append(query_cosine[:, off_diagonal].cpu())
                attention_rank_batches.append(
                    batched_effective_rank(attention).cpu()
                )
                latent_rank_batches.append(batched_effective_rank(latent).cpu())

        all_latents = torch.cat(latent_batches, dim=0)
        latent_cpu = all_latents.detach().float().cpu().contiguous()
        attention_cpu = torch.cat(attention_batches, dim=0).detach().float().cpu()
        max_attention = attention_cpu.max(dim=-1).values
        attention_entropy = -(
            attention_cpu * attention_cpu.clamp_min(1e-30).log()
        ).sum(dim=-1)
        unique_argmax = torch.tensor(
            [
                len(torch.unique(frame_attention.argmax(dim=-1)))
                for frame_attention in attention_cpu
            ],
            dtype=torch.float32,
        )
        q_norms = torch.cat(q_norm_batches).flatten()
        k_norms = torch.cat(k_norm_batches).flatten()
        score_values = torch.cat(score_batches).flatten()
        top1_scores = torch.cat(top1_batches).flatten()
        top2_scores = torch.cat(top2_batches).flatten()
        top1_minus_top2 = top1_scores - top2_scores
        query_cosines = torch.cat(query_cosine_batches).flatten()
        attention_ranks = torch.cat(attention_rank_batches).flatten()
        latent_ranks = torch.cat(latent_rank_batches).flatten()
        full_cache = None
        pipeline = None
        memory_id = f"diagnostic_{current_video}"
        if args.paths == "all":
            full_cache = prefill_full_latent_kv(model, latent_batches)
            pipeline = AudioRouterPipeline(model=model, tokenizer=tokenizer)
            fill_oqm(pipeline, memory_id, latent_batches)
        video_report = {
            "video_id": current_video,
            "video_path": path,
            "sampled_frames": int(len(frames)),
            "latent_tokens": int(all_latents.shape[0] * all_latents.shape[1]),
            "latent_summary": {
                "shape": list(all_latents.shape),
                "source_dtype": str(all_latents.dtype),
                "sha256_float32": hashlib.sha256(
                    latent_cpu.numpy().tobytes()
                ).hexdigest(),
                "mean": float(latent_cpu.mean()),
                "std": float(latent_cpu.std(unbiased=False)),
                "max_abs": float(latent_cpu.abs().max()),
            },
            "attention_summary": {
                "mean_max_probability": float(max_attention.mean()),
                "min_max_probability": float(max_attention.min()),
                "fraction_max_probability_ge_0_999": float(
                    (max_attention >= 0.999).float().mean()
                ),
                "mean_entropy": float(attention_entropy.mean()),
                "mean_effective_support": float(attention_entropy.exp().mean()),
                "mean_unique_argmax_per_frame": float(unique_argmax.mean()),
            },
            "qk_norms": {
                "q_norm_mean": float(q_norms.mean()),
                "q_norm_max": float(q_norms.max()),
                "k_norm_mean": float(k_norms.mean()),
                "k_norm_max": float(k_norms.max()),
            },
            "score_distribution": {
                "score_min": float(score_values.min()),
                "score_max": float(score_values.max()),
                "score_std": float(score_values.std(unbiased=False)),
                "score_dtype": str(score_values.dtype),
            },
            "score_ranking": {
                "top1_score_mean": float(top1_scores.mean()),
                "top1_score_min": float(top1_scores.min()),
                "top1_score_max": float(top1_scores.max()),
                "top2_score_mean": float(top2_scores.mean()),
                "top2_score_min": float(top2_scores.min()),
                "top2_score_max": float(top2_scores.max()),
                "top1_minus_top2_mean": float(top1_minus_top2.mean()),
                "top1_minus_top2_min": float(top1_minus_top2.min()),
                "top1_minus_top2_max": float(top1_minus_top2.max()),
                "top1_minus_top2_p50": float(
                    torch.quantile(top1_minus_top2, 0.50)
                ),
                "top1_minus_top2_p95": float(
                    torch.quantile(top1_minus_top2, 0.95)
                ),
            },
            "query_cosine": {
                "mean_off_diagonal_query_cosine_similarity": float(
                    query_cosines.mean()
                ),
                "max_query_cosine": float(query_cosines.max()),
                "min_query_cosine": float(query_cosines.min()),
            },
            "effective_rank": {
                "definition": "exp(entropy(normalized_singular_values))",
                "attention_mean": float(attention_ranks.mean()),
                "attention_min": float(attention_ranks.min()),
                "attention_max": float(attention_ranks.max()),
                "latent_mean": float(latent_ranks.mean()),
                "latent_min": float(latent_ranks.min()),
                "latent_max": float(latent_ranks.max()),
            },
            "full_kv_tokens": (
                int(full_cache.get_seq_length()) if full_cache is not None else None
            ),
            "questions": [],
        }
        for row in (() if args.skip_logits else rows_by_video[current_video]):
            prompt_ids = official_input_ids(row, tokenizer)
            direct = direct_insert_logits(model, prompt_ids, all_latents)
            summaries = {
                "direct_insert": summarize_option_logits(direct, ids),
            }
            retrieved_tokens = None
            if args.paths == "all":
                full_kv = logits_with_past(
                    model,
                    tokenizer,
                    pipeline.AudioRouter_core.query_task,
                    prompt_ids,
                    full_cache,
                )
                oqm, retrieved_tokens = oqm_logits(
                    model, tokenizer, pipeline, memory_id, prompt_ids
                )
                summaries.update(
                    full_latent_kv=summarize_option_logits(full_kv, ids),
                    oqm_retrieval=summarize_option_logits(oqm, ids),
                )
                summaries["full_vs_oqm_max_abs_option_logit_delta"] = max(
                    abs(summaries["full_latent_kv"]["logits"][letter]
                        - summaries["oqm_retrieval"]["logits"][letter])
                    for letter in LETTERS
                )
            video_report["questions"].append(
                {
                    "question_id": row.get("question_id", ""),
                    "answer": row["answer"],
                    "retrieved_kv_tokens": retrieved_tokens,
                    **summaries,
                }
            )
        report["videos"].append(video_report)
        if pipeline is not None:
            pipeline.AudioRouter_core.clear_cache(memory_id)
        del (
            full_cache,
            all_latents,
            latent_cpu,
            attention_cpu,
            attention_batches,
            q_norm_batches,
            k_norm_batches,
            score_batches,
            top1_batches,
            top2_batches,
            query_cosine_batches,
            attention_rank_batches,
            latent_rank_batches,
            q_norms,
            k_norms,
            score_values,
            top1_scores,
            top2_scores,
            top1_minus_top2,
            query_cosines,
            attention_ranks,
            latent_ranks,
            latent_batches,
            visual,
            audio,
            frames,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

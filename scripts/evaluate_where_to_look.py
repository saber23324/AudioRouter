"""Compare vision-only and audio-conditioned ADBT Where-to-Look maps.

The delayed question/answer is used only to construct a frozen full-visual
Grad x Activation reference after both query-independent memories have been
built. It never enters either bottleneck forward pass.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder
from AudioRouter.train_adbt_videomme import (
    encode_visual,
    frozen_teacher_visual_importance,
    make_prompt,
    row_video_path,
    rows_end_time,
    sample_video,
    video_path,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--audio-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", default="results/adbt_videomme_split.json")
    parser.add_argument(
        "--dataset-manifest",
        default="",
        help="Normalized MCQA manifest. Empty keeps the legacy VideoMME loader.",
    )
    parser.add_argument("--split-name", choices=("train", "test"), default="test")
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=-1,
        help="Shuffle eligible video IDs deterministically before truncation; -1 preserves split order.",
    )
    parser.add_argument(
        "--question-policy",
        choices=("auto", "all", "first", "middle", "latest"),
        default="auto",
        help="Teacher questions per video. Auto uses all delayed-QA rows and the middle causal query for real-time rows.",
    )
    parser.add_argument(
        "--max-teacher-questions",
        type=int,
        default=0,
        help="Cap delayed-QA teacher questions per video using evenly spaced rows; 0 keeps all.",
    )
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--teacher-topk", type=int, default=64)
    parser.add_argument("--metric-topk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument(
        "--beats-checkpoint",
        default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
    )
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--hf-cache", default="/home/yxd/.cache/huggingface")
    return parser.parse_args()


def select_teacher_rows(rows, policy: str):
    """Select teacher supervision without violating real-time query boundaries."""
    if not rows:
        raise ValueError("cannot select teacher rows from an empty video")
    timed = [row for row in rows if row.get("query_time_seconds") is not None]
    if policy == "auto":
        policy = "middle" if timed else "all"
    if policy == "all":
        distinct_boundaries = {row.get("query_time_seconds") for row in rows}
        if len(distinct_boundaries) > 1:
            raise ValueError(
                "question-policy=all would mix real-time queries with different causal "
                "boundaries; use first, middle, or latest"
            )
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            float("inf")
            if row.get("query_time_seconds") is None
            else float(row["query_time_seconds"]),
            str(row.get("question_id", "")),
        ),
    )
    if policy == "first":
        return [ordered[0]]
    if policy == "middle":
        return [ordered[len(ordered) // 2]]
    if policy == "latest":
        return [ordered[-1]]
    raise ValueError(f"unsupported question policy: {policy}")


def safe_figure_stem(video_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(video_id)).strip("._")
    return stem or "video"


def limit_teacher_rows(rows, maximum: int):
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def load_adbt(path: str, model, stage: int, temperature: float):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    saved_stage = int(saved.get("bottleneck_stage", 3))
    if saved_stage != stage:
        raise ValueError(f"checkpoint {path} has stage={saved_stage}, expected {stage}")
    adbt = AudioConditionedBottleneck(
        visual_dim=int(model.config.hidden_size),
        num_queries=int(saved.get("num_queries", 64)),
        hidden_size=int(saved.get("bottleneck_hidden", 256)),
        num_heads=int(saved.get("num_heads", 8)),
        stage=stage,
        latent_norm=saved.get("latent_norm", "none"),
        value_mode=saved.get("value_mode", "native"),
        architecture=saved.get("architecture", "phase4"),
        attention_temperature=temperature,
    )
    adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=True)
    return adbt


def topk_recall(reference: torch.Tensor, prediction: torch.Tensor, k: int):
    k = min(k, reference.shape[-1])
    target = reference.topk(k, dim=-1).indices
    selected = prediction.topk(k, dim=-1).indices
    matches = (selected.unsqueeze(-1) == target.unsqueeze(-2)).any(-1).float().sum(-1)
    return matches / k


def distribution_metrics(reference: torch.Tensor, prediction: torch.Tensor, k: int):
    eps = 1e-8
    reference = reference.float().clamp_min(eps)
    prediction = prediction.float().clamp_min(eps)
    cross_entropy = -(reference * prediction.log()).sum(-1)
    midpoint = 0.5 * (reference + prediction)
    js = 0.5 * (
        (reference * (reference.log() - midpoint.log())).sum(-1)
        + (prediction * (prediction.log() - midpoint.log())).sum(-1)
    )
    return cross_entropy, js, topk_recall(reference, prediction, k)


def overlay(ax, frame: np.ndarray, importance: np.ndarray, title: str):
    ax.imshow(frame)
    heat = torch.from_numpy(importance).reshape(1, 1, 14, 14)
    heat = F.interpolate(
        heat, size=frame.shape[:2], mode="bilinear", align_corners=False
    )[0, 0].numpy()
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
    ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
        trust_remote_code=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.get_vision_tower().parameters()).device

    vision_adbt = load_adbt(
        args.vision_checkpoint, model, stage=1, temperature=args.temperature
    ).to(device=device, dtype=torch.float32).eval()
    audio_adbt = load_adbt(
        args.audio_checkpoint, model, stage=3, temperature=args.temperature
    ).to(device=device, dtype=torch.float32).eval()
    if vision_adbt.num_queries != audio_adbt.num_queries:
        raise ValueError("Where-to-Look comparison requires equal query budgets")
    audio_encoder = BEATsAudioEncoder(
        args.beats_checkpoint, batch_size=args.beats_batch_size, output_mode="temporal"
    )

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    if args.dataset_manifest:
        manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
        dataset = manifest["rows"]
        dataset_name = str(manifest.get("dataset_name", Path(args.dataset_manifest).stem))
    else:
        manifest = None
        dataset_name = "videomme"
        dataset = load_dataset(
            "lmms-lab/Video-MME",
            cache_dir=args.hf_cache,
            download_mode="reuse_dataset_if_exists",
        )["test"]
    rows_by_video = {}
    for row in dataset:
        rows_by_video.setdefault(str(row["videoID"]), []).append(row)
    video_ids = [
        video_id for video_id in split[args.split_name] if video_id in rows_by_video
    ]
    if args.selection_seed >= 0:
        random.Random(args.selection_seed).shuffle(video_ids)
    video_ids = video_ids[: args.max_videos]

    records = []
    for video_id in video_ids:
        video_rows = rows_by_video[video_id]
        teacher_rows = limit_teacher_rows(
            select_teacher_rows(video_rows, args.question_policy),
            args.max_teacher_questions,
        )
        path = row_video_path(teacher_rows[0]) if manifest is not None else video_path(video_id)
        query_boundary = rows_end_time(teacher_rows)
        frames, timestamps = sample_video(
            path,
            fps="auto",
            max_frames=args.max_frames,
            end_time_seconds=query_boundary,
        )
        with torch.inference_mode():
            visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
            audio = audio_encoder.encode_video(
                path, timestamps, visual.device, ablation="real"
            )
            _, vision_attention = vision_adbt(
                visual.float(), None, torch.as_tensor(timestamps, device=visual.device)
            )
            _, audio_attention = audio_adbt(
                visual.float(),
                audio["embeddings"].to(visual.device, dtype=torch.float32),
                audio["end_timestamps"].to(visual.device),
            )
            vision_importance = vision_attention.float().mean(dim=1)
            audio_importance = audio_attention.float().mean(dim=1)

        teacher_maps = []
        for row in teacher_rows:
            teacher, _ = frozen_teacher_visual_importance(
                model,
                tokenizer,
                make_prompt(row, tokenizer),
                row["answer"],
                visual,
                topk=args.teacher_topk,
            )
            teacher_maps.append(teacher)
        teacher_importance = torch.stack(teacher_maps).mean(0)
        teacher_importance = teacher_importance / teacher_importance.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        vision_ce, vision_js, vision_recall = distribution_metrics(
            teacher_importance, vision_importance, args.metric_topk
        )
        audio_ce, audio_js, audio_recall = distribution_metrics(
            teacher_importance, audio_importance, args.metric_topk
        )
        delta = vision_ce - audio_ce
        representative = int(delta.argmax().item())
        record = {
            "dataset": dataset_name,
            "split": args.split_name,
            "video_id": video_id,
            "question_ids": [str(row.get("question_id", "")) for row in teacher_rows],
            "task_types": [str(row.get("task_type", "")) for row in teacher_rows],
            "query_boundary_seconds": query_boundary,
            "frames": int(len(frames)),
            "teacher_questions": len(teacher_maps),
            "vision_teacher_cross_entropy": float(vision_ce.mean().cpu()),
            "audio_teacher_cross_entropy": float(audio_ce.mean().cpu()),
            "cross_entropy_improvement": float(delta.mean().cpu()),
            "vision_teacher_js": float(vision_js.mean().cpu()),
            "audio_teacher_js": float(audio_js.mean().cpu()),
            "vision_teacher_topk_recall": float(vision_recall.mean().cpu()),
            "audio_teacher_topk_recall": float(audio_recall.mean().cpu()),
            "topk_recall_improvement": float(
                (audio_recall - vision_recall).mean().cpu()
            ),
            "representative_frame": representative,
            "representative_timestamp": float(timestamps[representative]),
        }
        records.append(record)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        axes[0].imshow(frames[representative])
        task_label = ", ".join(sorted(set(record["task_types"])))
        boundary_label = (
            "end of video"
            if query_boundary is None
            else f"query <= {query_boundary:.1f}s"
        )
        axes[0].set_title(f"frame t={timestamps[representative]:.1f}s")
        axes[0].axis("off")
        overlay(
            axes[1], frames[representative],
            vision_importance[representative].detach().cpu().numpy(), "Vision-only"
        )
        overlay(
            axes[2], frames[representative],
            audio_importance[representative].detach().cpu().numpy(), "Audio-conditioned"
        )
        overlay(
            axes[3], frames[representative],
            teacher_importance[representative].detach().cpu().numpy(), "Frozen teacher"
        )
        fig.suptitle(
            f"{dataset_name}/{args.split_name}/{video_id}: "
            f"{boundary_label}; task={task_label}; "
            f"CE improvement={record['cross_entropy_improvement']:.4f}",
            y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(figures_dir / f"{safe_figure_stem(video_id)}.png", dpi=160)
        plt.close(fig)

        del visual, audio, vision_attention, audio_attention, teacher_importance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "dataset": dataset_name,
        "dataset_manifest": (
            str(Path(args.dataset_manifest).resolve()) if args.dataset_manifest else None
        ),
        "split_file": str(Path(args.split_file).resolve()),
        "split": args.split_name,
        "selection_seed": args.selection_seed,
        "question_policy": args.question_policy,
        "max_teacher_questions": args.max_teacher_questions,
        "vision_checkpoint": str(Path(args.vision_checkpoint).resolve()),
        "audio_checkpoint": str(Path(args.audio_checkpoint).resolve()),
        "temperature": args.temperature,
        "videos": len(records),
        "frames": sum(item["frames"] for item in records),
        "teacher_questions": sum(item["teacher_questions"] for item in records),
        "mean_vision_teacher_cross_entropy": float(
            np.mean([item["vision_teacher_cross_entropy"] for item in records])
        ),
        "mean_audio_teacher_cross_entropy": float(
            np.mean([item["audio_teacher_cross_entropy"] for item in records])
        ),
        "mean_cross_entropy_improvement": float(
            np.mean([item["cross_entropy_improvement"] for item in records])
        ),
        "mean_vision_teacher_js": float(
            np.mean([item["vision_teacher_js"] for item in records])
        ),
        "mean_audio_teacher_js": float(
            np.mean([item["audio_teacher_js"] for item in records])
        ),
        "mean_vision_teacher_topk_recall": float(
            np.mean([item["vision_teacher_topk_recall"] for item in records])
        ),
        "mean_audio_teacher_topk_recall": float(
            np.mean([item["audio_teacher_topk_recall"] for item in records])
        ),
        "mean_topk_recall_improvement": float(
            np.mean([item["topk_recall_improvement"] for item in records])
        ),
        "per_video": records,
    }
    output = output_dir / "metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()

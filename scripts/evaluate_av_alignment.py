"""Evaluate Phase-5 audio/visual alignment without invoking the LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder
from AudioRouter.train_adbt_videomme import encode_visual, sample_video, video_path


def retrieval(similarities: torch.Tensor) -> tuple[float, float]:
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    ranking = similarities.argsort(dim=-1, descending=True)
    recall1 = (ranking[:, 0] == targets).float().mean()
    k = min(5, similarities.shape[1])
    recall5 = (ranking[:, :k] == targets[:, None]).any(-1).float().mean()
    return float(recall1.cpu()), float(recall5.cpu())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-file", default="results/adbt_videomme_split.json")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--min-temporal-offset", type=float, default=5.0)
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument(
        "--beats-checkpoint",
        default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
    )
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--hf-cache", default="/home/yxd/.cache/huggingface")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model_name = get_model_name_from_path(args.pretrained)
    _, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, model_name, device_map="auto", multimodal=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.get_vision_tower().parameters()).device
    adbt = AudioConditionedBottleneck(
        visual_dim=int(model.config.hidden_size),
        num_queries=int(checkpoint_args.get("num_queries", 64)),
        hidden_size=int(checkpoint_args.get("bottleneck_hidden", 256)),
        num_heads=int(checkpoint_args.get("num_heads", 8)),
        stage=3,
        latent_norm=checkpoint_args.get("latent_norm", "none"),
        value_mode=checkpoint_args.get("value_mode", "native"),
        architecture=checkpoint_args.get("architecture", "phase4"),
        attention_temperature=float(
            checkpoint_args.get("attention_temperature", 0.05)
        ),
    ).to(device=device, dtype=torch.float32)
    adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=True)
    adbt.eval()
    audio_encoder = BEATsAudioEncoder(
        args.beats_checkpoint,
        batch_size=args.beats_batch_size,
        output_mode="temporal",
    )

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    dataset = load_dataset(
        "lmms-lab/Video-MME",
        cache_dir=args.hf_cache,
        download_mode="reuse_dataset_if_exists",
    )["test"]
    available = {str(row["videoID"]) for row in dataset}
    video_ids = [video_id for video_id in split["test"] if video_id in available]
    video_ids = video_ids[: args.max_videos]

    all_audio = []
    all_visual = []
    per_video = []
    with torch.inference_mode():
        for video_id in video_ids:
            path = video_path(video_id)
            frames, timestamps = sample_video(path, max_frames=args.max_frames)
            visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
            audio = audio_encoder.encode_video(path, timestamps, visual.device, ablation="real")
            raw_audio = audio["embeddings"].to(device).float().mean(dim=1)
            raw_visual = visual.detach().to(device).float().mean(dim=1)
            aligned_audio = F.normalize(
                adbt.query_generator.audio_projection(raw_audio), dim=-1
            )
            aligned_visual = F.normalize(adbt.visual_key(raw_visual), dim=-1)
            similarities = aligned_audio @ aligned_visual.transpose(0, 1)
            recall1, recall5 = retrieval(similarities)
            times = torch.as_tensor(timestamps, device=device).float()
            hard_mask = (
                (times[:, None] - times[None, :]).abs()
                >= args.min_temporal_offset
            )
            hard_mask.fill_diagonal_(False)
            valid = hard_mask.any(-1)
            hard_max = similarities.masked_fill(~hard_mask, float("-inf")).max(-1).values
            temporal = float(
                (similarities.diagonal()[valid] > hard_max[valid]).float().mean().cpu()
            ) if valid.any() else float("nan")
            per_video.append(
                {
                    "video_id": video_id,
                    "frames": int(len(frames)),
                    "recall_at_1": recall1,
                    "recall_at_5": recall5,
                    "temporal_hard_accuracy": temporal,
                    "positive_similarity": float(similarities.diagonal().mean().cpu()),
                }
            )
            all_audio.append(aligned_audio.cpu())
            all_visual.append(aligned_visual.cpu())
            del visual, audio, raw_audio, raw_visual, aligned_audio, aligned_visual

    audio_matrix = torch.cat(all_audio)
    visual_matrix = torch.cat(all_visual)
    global_similarities = audio_matrix @ visual_matrix.transpose(0, 1)
    global_r1, global_r5 = retrieval(global_similarities)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "videos": len(per_video),
        "frames": int(audio_matrix.shape[0]),
        "global_recall_at_1": global_r1,
        "global_recall_at_5": global_r5,
        "mean_same_video_recall_at_1": sum(x["recall_at_1"] for x in per_video)
        / max(1, len(per_video)),
        "mean_same_video_recall_at_5": sum(x["recall_at_5"] for x in per_video)
        / max(1, len(per_video)),
        "mean_temporal_hard_accuracy": sum(
            x["temporal_hard_accuracy"] for x in per_video
        ) / max(1, len(per_video)),
        "mean_positive_similarity": float(global_similarities.diagonal().mean()),
        "per_video": per_video,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()

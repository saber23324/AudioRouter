"""Reconstruct an exact VideoMME FPS/frame-count audit without decoding frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset
from decord import VideoReader, cpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", default="auto")
    parser.add_argument(
        "--duration", choices=("all", "short", "medium", "long"), default="all"
    )
    parser.add_argument("--split-file", default="results/adbt_videomme_split.json")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--video-root", default="/home/yxd/.cache/huggingface/videomme/data"
    )
    parser.add_argument("--hf-cache", default="/home/yxd/.cache/huggingface")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_video(root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".MP4", ".mkv"):
        candidate = root / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing VideoMME media for {video_id} below {root}")


def actual_fps(setting: str, duration: float) -> float:
    if setting.startswith("auto"):
        scale_text = setting[len("auto") :]
        scale = float(scale_text) if scale_text else 1.0
        if scale <= 0:
            raise ValueError(f"FPS scale must be positive, got {setting}")
        return (0.2 if duration > 1800 else 0.5) * scale
    value = float(setting)
    if value <= 0:
        raise ValueError(f"FPS must be positive, got {setting}")
    return value


def main():
    args = parse_args()
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    allowed = set(split[args.split])
    dataset = load_dataset(
        "lmms-lab/Video-MME",
        cache_dir=args.hf_cache,
        download_mode="reuse_dataset_if_exists",
    )["test"]
    videos = {}
    for row in dataset:
        video_id = str(row["videoID"])
        duration_class = str(row["duration"])
        if video_id not in allowed:
            continue
        if args.duration != "all" and duration_class != args.duration:
            continue
        videos[video_id] = duration_class

    records = []
    root = Path(args.video_root)
    for video_id in sorted(videos):
        path = resolve_video(root, video_id)
        reader = VideoReader(str(path), ctx=cpu(0))
        source_fps = float(reader.get_avg_fps())
        total_source_frames = len(reader)
        duration_seconds = total_source_frames / source_fps
        fps = actual_fps(args.fps, duration_seconds)
        sampled_frames = max(
            1, min(total_source_frames, math.ceil(duration_seconds * fps))
        )
        records.append(
            {
                "video_id": video_id,
                "duration_class": videos[video_id],
                "path": str(path.resolve()),
                "duration_seconds": duration_seconds,
                "source_fps": source_fps,
                "actual_fps": fps,
                "sampled_frames": sampled_frames,
            }
        )

    report = {
        "fps_setting": args.fps,
        "split_file": str(Path(args.split_file).resolve()),
        "split": args.split,
        "duration_filter": args.duration,
        "unique_videos": len(records),
        "total_sampled_frames": sum(row["sampled_frames"] for row in records),
        "sampled_frames_min": min((row["sampled_frames"] for row in records), default=None),
        "sampled_frames_max": max((row["sampled_frames"] for row in records), default=None),
        "totals_by_duration": {
            duration_class: {
                "unique_videos": sum(
                    row["duration_class"] == duration_class for row in records
                ),
                "total_sampled_frames": sum(
                    row["sampled_frames"]
                    for row in records
                    if row["duration_class"] == duration_class
                ),
            }
            for duration_class in ("short", "medium", "long")
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

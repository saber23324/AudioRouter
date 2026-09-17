#!/usr/bin/env python3
"""Select high-confidence correct examples and render an inference gallery."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        default="results/repro-validation-20260917/streamingbench-realtime-seedall-epoch3-tau005.json",
    )
    parser.add_argument(
        "--manifest",
        default="results/dataset_manifests/streamingbench_realtime.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="ckpt/streamingbench_realtime_adbt_epoch3.pt",
    )
    parser.add_argument("--output-dir", default="assets/demo_gallery")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-query-time", type=float, default=12.0)
    parser.add_argument("--exclude-video-id", action="append", default=[])
    parser.add_argument("--attention-temperature", type=float, default=0.05)
    parser.add_argument("--render-fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sample"


def select_examples(args: argparse.Namespace) -> list[dict]:
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / args.evaluation).read_text(encoding="utf-8"))
    rows = {
        (str(row["videoID"]), str(row.get("question_id"))): row
        for row in manifest["rows"]
    }
    excluded = set(args.exclude_video_id)
    candidates = []
    seen_videos = set()
    for record in evaluation["records"]:
        if not record.get("option_correct"):
            continue
        row = rows.get((str(record["videoID"]), str(record.get("question_id"))))
        if row is None or str(row["videoID"]) in excluded:
            continue
        query_time = row.get("query_time_seconds")
        if query_time is None or float(query_time) > args.max_query_time:
            continue
        if not Path(row["video_path"]).is_file():
            continue
        target = str(row["answer"]).strip().upper()
        confidence = float(record["option_probabilities"][target])
        candidates.append((confidence, row, record))

    selected = []
    for confidence, row, record in sorted(candidates, key=lambda item: item[0], reverse=True):
        video_id = str(row["videoID"])
        if video_id in seen_videos:
            continue
        seen_videos.add(video_id)
        selected.append(
            {
                "selection_confidence": confidence,
                "row": row,
                "evaluation_record": record,
            }
        )
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"found only {len(selected)} eligible examples, requested {args.count}")
    return selected


def make_preview(video: Path, output: Path, query_time: float) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, query_time * 0.75) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode preview frame from {video}")
    if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"could not write preview: {output}")


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_examples(args)
    gallery = []

    for index, item in enumerate(selected, start=1):
        row = item["row"]
        stem = f"{index:02d}_{safe_stem(str(row['videoID']))}"
        video_output = output_dir / f"{stem}.mp4"
        sidecar = video_output.with_suffix(".json")
        preview = output_dir / f"{stem}.jpg"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "render_inference_demo.py"),
            "--video",
            str(row["video_path"]),
            "--question",
            str(row["question"]),
            "--options",
            *[str(option) for option in row["options"]],
            "--checkpoint",
            str(ROOT / args.checkpoint),
            "--attention-temperature",
            str(args.attention_temperature),
            "--query-time",
            str(row["query_time_seconds"]),
            "--fps",
            "auto",
            "--max-frames",
            "0",
            "--render-fps",
            str(args.render_fps),
            "--output",
            str(video_output),
            "--include-audio",
        ]
        if args.overwrite or not video_output.is_file() or not sidecar.is_file():
            print(
                f"[{index}/{len(selected)}] {row['videoID']} "
                f"selection_confidence={item['selection_confidence']:.6f}",
                flush=True,
            )
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        target = str(row["answer"]).strip().upper()
        metadata.update(
            {
                "dataset": "StreamingBench Real-Time",
                "videoID": row["videoID"],
                "question_id": row.get("question_id"),
                "task_type": row.get("task_type"),
                "target": target,
                "correct": metadata["prediction"] == target,
                "selection_confidence": item["selection_confidence"],
            }
        )
        sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        make_preview(video_output, preview, float(row["query_time_seconds"]))
        gallery.append(
            {
                "rank": index,
                "videoID": row["videoID"],
                "task_type": row.get("task_type"),
                "question": row["question"],
                "options": row["options"],
                "target": target,
                "prediction": metadata["prediction"],
                "correct": metadata["correct"],
                "option_probabilities": metadata["option_probabilities"],
                "query_time_seconds": row["query_time_seconds"],
                "video": video_output.name,
                "preview": preview.name,
                "metadata": sidecar.name,
            }
        )
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(gallery, indent=2, ensure_ascii=False), encoding="utf-8")
    correct = sum(int(item["correct"]) for item in gallery)
    print(f"rendered {len(gallery)} demos; correct={correct}/{len(gallery)}; index={index_path}")


if __name__ == "__main__":
    main()

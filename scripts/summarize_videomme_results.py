"""Summarize official VideoMME result directories without averaging splits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SPLITS = ("short", "medium", "long")
METRIC = "videomme_perception_score,none"
PEAK_PATTERN = re.compile(r"Peak allocated: cuda:\d+=([0-9.]+) MiB")
SAMPLING_PATTERN = re.compile(
    r"\[VIDEOMME SAMPLING\] path=(?P<path>\S+).*?"
    r"actual_fps=(?P<fps>[0-9.]+) sampled_frames=(?P<frames>\d+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("SPLIT", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--sampling-audit",
        default="",
        help="Optional output of audit_videomme_sampling.py used when a run log is unavailable.",
    )
    return parser.parse_args()


def load_one(split: str, directory: Path):
    result_files = sorted(directory.rglob("*_results.json"))
    sample_files = sorted(directory.rglob(f"*_samples_videomme_{split}.jsonl"))
    if len(result_files) != 1 or len(sample_files) != 1:
        raise ValueError(
            f"{directory}: expected one result and one {split} sample file, got "
            f"{len(result_files)} and {len(sample_files)}"
        )
    result = json.loads(result_files[0].read_text(encoding="utf-8"))
    score = float(result["results"][f"videomme_{split}"][METRIC])
    with sample_files[0].open(encoding="utf-8") as handle:
        samples = sum(1 for _ in handle)
    correct = round(score * samples / 100.0)
    log_path = Path(str(directory) + ".log")
    peaks = []
    sampling = []
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        peaks = [float(value) for value in PEAK_PATTERN.findall(log_text)]
        sampling = [
            {
                "path": match.group("path"),
                "actual_fps": float(match.group("fps")),
                "frames": int(match.group("frames")),
            }
            for match in SAMPLING_PATTERN.finditer(log_text)
        ]
    # VideoMME evaluates three questions for each video.  The evaluator emits a
    # sampling audit line for every question, so summing all records would count
    # the same decoded frame sequence three times.  Keep both accounting views,
    # but use the unique-video total for the experiment's sampled-frame budget.
    sampling_by_video = {}
    for row in sampling:
        previous = sampling_by_video.get(row["path"])
        signature = (row["actual_fps"], row["frames"])
        if previous is not None and previous != signature:
            raise ValueError(
                f"{directory}: inconsistent sampling for {row['path']}: "
                f"{previous} versus {signature}"
            )
        sampling_by_video[row["path"]] = signature
    unique_frame_counts = [value[1] for value in sampling_by_video.values()]
    return {
        "split": split,
        "directory": str(directory.resolve()),
        "result_file": str(result_files[0].resolve()),
        "sample_file": str(sample_files[0].resolve()),
        "samples": samples,
        "correct": correct,
        "accuracy": score,
        "peak_allocated_mib": max(peaks) if peaks else None,
        "sampling_records": len(sampling),
        "unique_videos": len(sampling_by_video),
        "total_sampled_frames": sum(unique_frame_counts) if sampling else None,
        "total_sampled_frames_across_qa": (
            sum(row["frames"] for row in sampling) if sampling else None
        ),
        "sampled_frames_min": min(unique_frame_counts, default=None),
        "sampled_frames_max": max(unique_frame_counts, default=None),
        "actual_fps_values": sorted({row["actual_fps"] for row in sampling}),
    }


def main():
    args = parse_args()
    audit_records = None
    if args.sampling_audit:
        audit = json.loads(Path(args.sampling_audit).read_text(encoding="utf-8"))
        audit_records = audit.get("records", [])
    seen = set()
    rows = []
    for split, directory in args.result:
        split = split.lower()
        if split not in SPLITS or split in seen:
            raise ValueError(f"invalid or duplicate split: {split}")
        seen.add(split)
        row = load_one(split, Path(directory))
        if row["total_sampled_frames"] is None and audit_records is not None:
            records = [
                record for record in audit_records
                if record.get("duration_class") == split
            ]
            frame_counts = [int(record["sampled_frames"]) for record in records]
            row.update(
                {
                    "unique_videos": len(records),
                    "total_sampled_frames": sum(frame_counts),
                    "sampled_frames_min": min(frame_counts, default=None),
                    "sampled_frames_max": max(frame_counts, default=None),
                    "actual_fps_values": sorted(
                        {float(record["actual_fps"]) for record in records}
                    ),
                    "sampling_source": "metadata_audit",
                }
            )
        elif row["total_sampled_frames"] is not None:
            row["sampling_source"] = "evaluation_log"
        rows.append(row)
    total = sum(row["samples"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    report = {
        "splits": rows,
        "total_samples": total,
        "total_correct": correct,
        "overall_accuracy": 100.0 * correct / total,
        "peak_allocated_mib": max(
            (
                row["peak_allocated_mib"]
                for row in rows
                if row["peak_allocated_mib"] is not None
            ),
            default=None,
        ),
        "total_sampled_frames": (
            sum(row["total_sampled_frames"] for row in rows)
            if all(row["total_sampled_frames"] is not None for row in rows)
            else None
        ),
        "unique_videos": (
            sum(row["unique_videos"] for row in rows)
            if all(row["unique_videos"] for row in rows)
            else None
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

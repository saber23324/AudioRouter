"""Build deterministic, video-grouped manifests for the local MCQA subsets.

Only rows whose media exists locally are retained.  The normalized schema is
consumed by ``AudioRouter.train_adbt_videomme`` through ``--dataset-manifest``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from decord import VideoReader, cpu


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MLVU_MCQA_TASKS = {
    "1_plotQA": "plotQA",
    "2_needle": "needle",
    "3_ego": "ego",
    "4_count": "count",
    "5_order": "order",
    "6_anomaly_reco": "anomaly_reco",
    "7_topic_reasoning": "topic_reasoning",
}


def answer_letter(candidates: list[str], answer: str) -> str:
    normalized = str(answer).strip().casefold()
    for index, candidate in enumerate(candidates):
        if str(candidate).strip().casefold() == normalized:
            return LETTERS[index]
    raise ValueError(f"answer {answer!r} is not one of {candidates!r}")


def labeled_options(candidates: list[str]) -> list[str]:
    if len(candidates) > len(LETTERS):
        raise ValueError(f"too many options: {len(candidates)}")
    return [f"{LETTERS[index]}. {str(value).strip()}" for index, value in enumerate(candidates)]


def parse_timestamp(value: str) -> float:
    parts = [float(part) for part in str(value).strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unsupported timestamp: {value!r}")


def build_mlvu(snapshot: Path) -> dict:
    annotation_path = snapshot / "MLVU/json/3_ego.json"
    video_root = snapshot / "MLVU/video/3_ego"
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    missing = []
    per_video_index: dict[str, int] = {}
    for record in source:
        video_id = Path(record["video"]).stem
        path = video_root / record["video"]
        if not path.is_file():
            missing.append(record["video"])
            continue
        candidates = [str(value) for value in record["candidates"]]
        per_video_index[video_id] = per_video_index.get(video_id, 0) + 1
        rows.append(
            {
                "videoID": video_id,
                "video_path": str(path.resolve()),
                "question_id": f"mlvu_ego_{video_id}_{per_video_index[video_id]}",
                "question": str(record["question"]).strip(),
                "options": labeled_options(candidates),
                "answer": answer_letter(candidates, record["answer"]),
                "query_time_seconds": None,
                "task_type": "ego",
                "source_duration_seconds": float(record["duration"]),
            }
        )
    return {
        "dataset_name": "mlvu_ego_partial",
        "source_readme": str((snapshot / "README.md").resolve()),
        "source_annotation": str(annotation_path.resolve()),
        "rows": rows,
        "source_rows": len(source),
        "missing_rows": len(missing),
        "missing_video_names": sorted(set(missing)),
    }


def build_mlvu_full_mcqa(snapshot: Path) -> dict:
    """Normalize all seven scored MLVU-dev multiple-choice tasks.

    MLVU's ``8_sub_scene`` and ``9_summary`` annotations are free-form
    generation tasks and therefore intentionally remain outside this accuracy
    manifest. Task-prefixed video IDs keep same-named files in different source
    directories distinct.
    """
    rows = []
    missing = []
    task_stats = {}
    source_annotations = []
    for directory_name, task_type in MLVU_MCQA_TASKS.items():
        annotation_path = snapshot / "MLVU/json" / f"{directory_name}.json"
        video_root = snapshot / "MLVU/video" / directory_name
        source = json.loads(annotation_path.read_text(encoding="utf-8"))
        source_annotations.append(str(annotation_path.resolve()))
        per_video_index: dict[str, int] = {}
        available_rows = 0
        available_videos = set()
        for record in source:
            source_video_name = str(record["video"])
            source_video_stem = Path(source_video_name).stem
            video_id = f"{task_type}::{source_video_stem}"
            path = video_root / source_video_name
            if not path.is_file():
                missing.append(f"{directory_name}/{source_video_name}")
                continue
            candidates = [str(value) for value in record["candidates"]]
            if len(candidates) != 4:
                raise ValueError(
                    f"MLVU-dev MCQA row must have four candidates: "
                    f"{annotation_path} {source_video_name}"
                )
            per_video_index[video_id] = per_video_index.get(video_id, 0) + 1
            rows.append(
                {
                    "videoID": video_id,
                    "video_path": str(path.resolve()),
                    "question_id": (
                        f"mlvu_{task_type}_{source_video_stem}_"
                        f"{per_video_index[video_id]}"
                    ),
                    "question": str(record["question"]).strip(),
                    "options": labeled_options(candidates),
                    "answer": answer_letter(candidates, record["answer"]),
                    "query_time_seconds": None,
                    "task_type": task_type,
                    "source_duration_seconds": float(record["duration"]),
                    "source_task": directory_name,
                }
            )
            available_rows += 1
            available_videos.add(video_id)
        task_stats[task_type] = {
            "source_rows": len(source),
            "available_rows": available_rows,
            "available_videos": len(available_videos),
        }
    return {
        "dataset_name": "mlvu_dev_mcqa_full",
        "source_readme": str((snapshot / "README.md").resolve()),
        "source_annotations": source_annotations,
        "excluded_generation_tasks": ["8_sub_scene", "9_summary"],
        "task_stats": task_stats,
        "rows": rows,
        "source_rows": sum(value["source_rows"] for value in task_stats.values()),
        "missing_rows": len(missing),
        "missing_video_names": sorted(set(missing)),
    }


def build_mlvu_full_split(
    rows: list[dict],
    *,
    seed: int = 1234,
    train_fraction: float = 0.8,
    ego_split: dict | None = None,
) -> dict:
    """Create a deterministic task-stratified, media-grouped MLVU split.

    When supplied, the historical Ego split is preserved exactly so a
    checkpoint previously trained on that split does not contaminate the new
    Ego validation subset. Video IDs backed by the same resolved media file are
    assigned together even when that file is reused by multiple tasks.
    """
    task_videos: dict[str, set[str]] = defaultdict(set)
    video_task = {}
    video_path = {}
    for row in rows:
        task_type = str(row["task_type"])
        video_id = str(row["videoID"])
        path = str(Path(row["video_path"]).resolve())
        task_videos[task_type].add(video_id)
        if video_id in video_task and video_task[video_id] != task_type:
            raise ValueError(f"video ID belongs to multiple MLVU tasks: {video_id}")
        if video_id in video_path and video_path[video_id] != path:
            raise ValueError(f"video ID maps to multiple media files: {video_id}")
        video_task[video_id] = task_type
        video_path[video_id] = path

    path_groups: dict[str, set[str]] = defaultdict(set)
    for video_id, path in video_path.items():
        path_groups[path].add(video_id)

    target_test = {
        task_type: len(ids) - int(len(ids) * train_fraction)
        for task_type, ids in task_videos.items()
    }
    assignment = {}
    if ego_split is not None:
        ego_train = {f"ego::{value}" for value in ego_split["train"]}
        ego_test = {f"ego::{value}" for value in ego_split["test"]}
        if ego_train | ego_test != task_videos.get("ego", set()):
            raise ValueError("historical Ego split does not cover full Ego videos")
        if ego_train & ego_test:
            raise ValueError("historical Ego split contains train/test overlap")
        for side, video_ids in (("train", ego_train), ("test", ego_test)):
            for video_id in video_ids:
                path = video_path[video_id]
                previous = assignment.get(path)
                if previous is not None and previous != side:
                    raise ValueError("historical Ego split divides reused media")
                assignment[path] = side

    test_counts = defaultdict(int)
    for path, side in assignment.items():
        if side == "test":
            for video_id in path_groups[path]:
                test_counts[video_task[video_id]] += 1
    for task_type, count in test_counts.items():
        if count > target_test[task_type]:
            raise ValueError(f"fixed split exceeds test target for {task_type}")

    remaining = [path for path in sorted(path_groups) if path not in assignment]
    random.Random(seed).shuffle(remaining)
    # Prefer multi-task media groups first; single-task groups can then fill any
    # remaining per-task deficits exactly.
    remaining.sort(key=lambda path: -len({video_task[v] for v in path_groups[path]}))
    for path in remaining:
        additions = defaultdict(int)
        for video_id in path_groups[path]:
            additions[video_task[video_id]] += 1
        fits = all(
            test_counts[task_type] + count <= target_test[task_type]
            for task_type, count in additions.items()
        )
        needed = any(
            test_counts[task_type] < target_test[task_type]
            for task_type in additions
        )
        side = "test" if fits and needed else "train"
        assignment[path] = side
        if side == "test":
            for task_type, count in additions.items():
                test_counts[task_type] += count

    if dict(test_counts) != target_test:
        raise ValueError(
            f"could not satisfy task-stratified test targets: "
            f"actual={dict(test_counts)} expected={target_test}"
        )

    train = []
    test = []
    for path, video_ids in path_groups.items():
        destination = test if assignment[path] == "test" else train
        destination.extend(video_ids)

    if set(train) & set(test):
        raise ValueError("generated full MLVU split contains train/test overlap")
    return {"train": sorted(train), "test": sorted(test)}


def streaming_video_id(question_id: str) -> str:
    match = re.search(r"_sample_(\d+)_\d+$", question_id)
    if not match:
        raise ValueError(f"cannot parse StreamingBench question id: {question_id!r}")
    return f"sample_{int(match.group(1))}"


def build_streaming(root: Path) -> dict:
    annotation_path = root / "StreamingBench/Real_Time_Visual_Understanding.csv"
    video_root = root / "Real_Time_Visual_Understanding_videos"
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    missing = []
    invalid_timestamps = []
    duration_cache: dict[str, float] = {}
    for record in source:
        video_id = streaming_video_id(record["question_id"])
        path = video_root / video_id / "video.mp4"
        if not path.is_file():
            missing.append(video_id)
            continue
        candidates = ast.literal_eval(record["options"])
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"invalid options for {record['question_id']}: {candidates!r}")
        query_time = parse_timestamp(record["time_stamp"])
        if video_id not in duration_cache:
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            duration_cache[video_id] = len(reader) / float(reader.get_avg_fps())
        if query_time > duration_cache[video_id] + 1.0:
            invalid_timestamps.append(
                {
                    "question_id": record["question_id"],
                    "time_stamp": record["time_stamp"],
                    "parsed_seconds": query_time,
                    "video_duration_seconds": duration_cache[video_id],
                }
            )
            continue
        # The CSV already prefixes options with A./B./C./D.; retain it exactly.
        rows.append(
            {
                "videoID": video_id,
                "video_path": str(path.resolve()),
                "question_id": record["question_id"],
                "question": record["question"].strip(),
                "options": [str(value).strip() for value in candidates],
                "answer": record["answer"].strip().upper(),
                "query_time_seconds": query_time,
                "task_type": record["task_type"],
                "frames_required": record["frames_required"],
                "temporal_clue_type": record["temporal_clue_type"],
            }
        )
    return {
        "dataset_name": "streamingbench_realtime_partial",
        "source_readme": str((root / "README.md").resolve()),
        "source_annotation": str(annotation_path.resolve()),
        "rows": rows,
        "source_rows": len(source),
        "missing_rows": len(missing),
        "missing_video_names": sorted(set(missing)),
        "invalid_timestamp_rows": invalid_timestamps,
    }


def write_manifest(payload: dict, path: Path) -> None:
    video_ids = sorted({str(row["videoID"]) for row in payload["rows"]})
    payload = dict(payload)
    payload["available_rows"] = len(payload["rows"])
    payload["available_videos"] = len(video_ids)
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload["rows"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{payload['dataset_name']}: rows={payload['available_rows']}/"
        f"{payload['source_rows']} videos={payload['available_videos']} "
        f"missing_rows={payload['missing_rows']} "
        f"invalid_rows={len(payload.get('invalid_timestamp_rows', []))} -> {path}"
    )


def write_split(payload: dict, path: Path) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"refusing to overwrite a different split: {path}")
        print(f"verified unchanged split -> {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"split: train={len(payload['train'])} test={len(payload['test'])} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlvu-snapshot",
        default="/home/yxd/pkt/huggingface/hub/datasets--MLVU--MVLU/"
        "snapshots/06ddc388aa34746b3abba77972b7a7dfd977f7a3",
    )
    parser.add_argument("--streaming-root", default="/home/yxd/pkt/streaming/dataset")
    parser.add_argument("--output-dir", default="results/dataset_manifests")
    parser.add_argument("--split-output-dir", default="results/dataset_splits")
    parser.add_argument("--mlvu-full-seed", type=int, default=1234)
    parser.add_argument(
        "--mlvu-ego-split",
        default="results/dataset_splits/mlvu_ego_seed1234_80_20.json",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "mlvu", "mlvu-full", "streaming"),
        default="all",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.dataset in {"all", "mlvu"}:
        write_manifest(build_mlvu(Path(args.mlvu_snapshot)), output_dir / "mlvu_ego.json")
    if args.dataset in {"all", "mlvu-full"}:
        full_manifest = build_mlvu_full_mcqa(Path(args.mlvu_snapshot))
        write_manifest(full_manifest, output_dir / "mlvu_full_mcqa.json")
        ego_split = json.loads(Path(args.mlvu_ego_split).read_text(encoding="utf-8"))
        full_split = build_mlvu_full_split(
            full_manifest["rows"], seed=args.mlvu_full_seed, ego_split=ego_split
        )
        write_split(
            full_split,
            Path(args.split_output_dir)
            / f"mlvu_full_mcqa_seed{args.mlvu_full_seed}_pathgroup_80_20.json",
        )
    if args.dataset in {"all", "streaming"}:
        write_manifest(
            build_streaming(Path(args.streaming_root)),
            output_dir / "streamingbench_realtime.json",
        )


if __name__ == "__main__":
    main()

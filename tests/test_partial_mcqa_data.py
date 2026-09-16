import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.prepare_partial_mcqa_datasets import (
    answer_letter,
    build_mlvu_full_split,
    parse_timestamp,
)
from AudioRouter.train_adbt_videomme import causal_prefix_length, make_split, rows_end_time


class PartialMCQADataTest(unittest.TestCase):
    def test_timestamp_parser_supports_benchmark_formats(self):
        self.assertEqual(parse_timestamp("09:11"), 551.0)
        self.assertEqual(parse_timestamp("00:09:11"), 551.0)

    def test_answer_is_normalized_to_candidate_letter(self):
        self.assertEqual(answer_letter(["one", "Two", "three"], " two "), "B")

    def test_query_uses_only_sampled_causal_prefix(self):
        timestamps = np.asarray([0.0, 2.0, 4.0, 6.0])
        self.assertEqual(causal_prefix_length(timestamps, {"query_time_seconds": 4}), 3)
        self.assertEqual(causal_prefix_length(timestamps, {"query_time_seconds": 1}), 1)
        self.assertEqual(causal_prefix_length(timestamps, {}), 4)

    def test_video_decode_boundary_is_latest_question(self):
        rows = [
            {"query_time_seconds": 3.0},
            {"query_time_seconds": 9.0},
            {"query_time_seconds": 5.0},
        ]
        self.assertEqual(rows_end_time(rows), 9.0)
        self.assertIsNone(rows_end_time([{"query_time_seconds": None}]))

    def test_existing_split_is_loaded_without_overwrite(self):
        dataset = [{"videoID": value} for value in ("a", "b", "c")]
        payload = {"train": ["a", "a", "b"], "test": ["c"]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = make_split(dataset, path)
            self.assertEqual(loaded, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_overlap_requires_explicit_opt_in(self):
        dataset = [{"videoID": value} for value in ("a", "b")]
        payload = {"train": ["a", "b"], "test": ["b"]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train/test-overlap"):
                make_split(dataset, path)
            self.assertEqual(
                make_split(dataset, path, allow_train_test_overlap=True), payload
            )

    def test_full_mlvu_split_is_stratified_and_preserves_ego(self):
        rows = []
        for task_type in ("ego", "plotQA"):
            for video in ("a", "b", "c", "d", "e"):
                rows.append(
                    {
                        "task_type": task_type,
                        "videoID": f"{task_type}::{video}",
                        "video_path": f"/{task_type}/{video}.mp4",
                    }
                )
        split = build_mlvu_full_split(
            rows,
            ego_split={"train": ["a", "b", "c", "d"], "test": ["e"]},
        )
        self.assertEqual(
            {value for value in split["train"] if value.startswith("ego::")},
            {"ego::a", "ego::b", "ego::c", "ego::d"},
        )
        self.assertEqual(
            {value for value in split["test"] if value.startswith("ego::")},
            {"ego::e"},
        )
        self.assertEqual(len(split["train"]), 8)
        self.assertEqual(len(split["test"]), 2)
        self.assertFalse(set(split["train"]) & set(split["test"]))

    def test_full_mlvu_split_keeps_reused_media_on_one_side(self):
        rows = []
        for index in range(5):
            path = f"/media/{index}.mp4"
            rows.append(
                {
                    "task_type": "plotQA",
                    "videoID": f"plotQA::{index}",
                    "video_path": path,
                }
            )
            rows.append(
                {
                    "task_type": "topic_reasoning",
                    "videoID": f"topic_reasoning::{index}",
                    "video_path": path,
                }
            )
        split = build_mlvu_full_split(rows)
        side = {video_id: name for name in ("train", "test") for video_id in split[name]}
        for index in range(5):
            self.assertEqual(
                side[f"plotQA::{index}"], side[f"topic_reasoning::{index}"]
            )


if __name__ == "__main__":
    unittest.main()

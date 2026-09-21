import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.prepare_partial_mcqa_datasets import (
    answer_letter,
    parse_timestamp,
)
from AudioRouter.train_adbt_videomme import (
    causal_prefix_length,
    load_training_manifest,
    rows_end_time,
)


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

    def test_training_manifest_loads_every_row_and_resolves_relative_media(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.touch()
            payload = {
                "dataset_name": "external-demo",
                "rows": [
                    {
                        "videoID": "sample",
                        "video_path": "sample.mp4",
                        "question": "What happens?",
                        "options": ["A. One", "B. Two", "C. Three", "D. Four"],
                        "answer": "b",
                    },
                    {
                        "videoID": "sample",
                        "video_path": "sample.mp4",
                        "question": "When?",
                        "options": ["A. Now", "B. Later", "C. Before", "D. Never"],
                        "answer": "A",
                        "query_time_seconds": 3,
                    },
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset_name, rows = load_training_manifest(path)
            self.assertEqual(dataset_name, "external-demo")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["video_path"], str(video.resolve()))
            self.assertEqual(rows[0]["answer"], "B")
            self.assertIsNone(rows[0]["query_time_seconds"])
            self.assertEqual(rows[1]["query_time_seconds"], 3.0)

    def test_training_manifest_rejects_non_mcqa_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.touch()
            path = root / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "videoID": "sample",
                                "video_path": "sample.mp4",
                                "question": "What happens?",
                                "options": ["one", "two", "three"],
                                "answer": "A",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly four options"):
                load_training_manifest(path)


if __name__ == "__main__":
    unittest.main()

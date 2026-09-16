"""Compare two lmms-eval VideoMME sample JSONL files by question id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    matches = sorted(candidate.rglob("*_samples_videomme_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"{candidate}: expected one VideoMME sample JSONL, got {len(matches)}")
    return matches[0]


def load(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metric = row["videomme_perception_score"]
            rows[metric["question_id"]] = {
                "prediction": metric["pred_answer"],
                "answer": metric["answer"],
            }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left_path, right_path = resolve(args.left), resolve(args.right)
    left, right = load(left_path), load(right_path)
    if set(left) != set(right):
        raise ValueError("question-id sets differ")
    prediction_changes = 0
    left_to_right_gain = 0
    left_to_right_loss = 0
    for question_id in left:
        lhs, rhs = left[question_id], right[question_id]
        prediction_changes += lhs["prediction"] != rhs["prediction"]
        left_correct = lhs["prediction"] == lhs["answer"]
        right_correct = rhs["prediction"] == rhs["answer"]
        left_to_right_gain += (not left_correct) and right_correct
        left_to_right_loss += left_correct and (not right_correct)
    report = {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "samples": len(left),
        "prediction_changes": prediction_changes,
        "left_to_right_gains": left_to_right_gain,
        "left_to_right_losses": left_to_right_loss,
        "net_correct_delta": left_to_right_gain - left_to_right_loss,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

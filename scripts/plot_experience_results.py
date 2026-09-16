"""Render the figures required by docs/experience.md from saved JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/experience_figures")
    parser.add_argument("--full-summary", default="results/exp01_full196_summary.json")
    parser.add_argument("--q32-summary", default="results/exp04_audio_q32_summary.json")
    parser.add_argument("--q64-summary", default="results/exp04_audio_q64_summary.json")
    parser.add_argument("--q96-summary", default="results/exp04_audio_q96_summary.json")
    parser.add_argument("--q128-summary", default="results/exp04_audio_q128_summary.json")
    parser.add_argument("--full-profile", default="results/exp06_profile_full_auto.json")
    parser.add_argument("--audio-profile", default="results/exp06_profile_audio64_auto.json")
    return parser.parse_args()


def save_figure(fig, output: Path):
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = [32, 64, 96, 128, 196]
    summaries = [
        load(args.q32_summary),
        load(args.q64_summary),
        load(args.q96_summary),
        load(args.q128_summary),
        load(args.full_summary),
    ]
    accuracies = [row["overall_accuracy"] for row in summaries]
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(budgets, accuracies, marker="o", linewidth=2)
    for budget, accuracy in zip(budgets, accuracies):
        axis.annotate(f"{accuracy:.2f}", (budget, accuracy), xytext=(0, 7),
                      textcoords="offset points", ha="center", fontsize=8)
    axis.set_xlabel("Visual tokens per frame")
    axis.set_ylabel("VideoMME held-out accuracy (%)")
    axis.set_xticks(budgets)
    axis.grid(alpha=0.25)
    save_figure(fig, output_dir / "exp04_accuracy_vs_tokens.png")

    full = load(args.full_profile)
    audio = load(args.audio_profile)
    full_by_frames = {row["frames"]: row for row in full["scaling"]}
    audio_by_frames = {row["frames"]: row for row in audio["scaling"]}
    frames = sorted(set(full_by_frames) & set(audio_by_frames))
    full_prefill = [full_by_frames[n]["llm_prefill_sec"] for n in frames]
    audio_prefill = [audio_by_frames[n]["llm_prefill_sec"] for n in frames]

    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(frames, full_prefill, marker="o", label="Full-196")
    axis.plot(frames, audio_prefill, marker="o", label="Audio-64")
    axis.set_xlabel("Number of frames")
    axis.set_ylabel("Cold LLM prefill time (s)")
    axis.set_xticks(frames)
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, output_dir / "exp06_prefill_vs_frames.png")

    speedups = [dense / compact for dense, compact in zip(full_prefill, audio_prefill)]
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(frames, speedups, marker="o", linewidth=2)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Number of frames")
    axis.set_ylabel("Prefill speedup (Full-196 / Audio-64)")
    axis.set_xticks(frames)
    axis.grid(alpha=0.25)
    save_figure(fig, output_dir / "exp06_prefill_speedup_vs_frames.png")

    manifest = {
        "budget": {"tokens_per_frame": budgets, "overall_accuracy": accuracies},
        "scaling": {
            "frames": frames,
            "full196_prefill_sec": full_prefill,
            "audio64_prefill_sec": audio_prefill,
            "speedup": speedups,
        },
    }
    (output_dir / "figure_data.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

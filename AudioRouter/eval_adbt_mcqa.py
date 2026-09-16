"""Evaluate a trained ADBT adapter on a normalized held-out MCQA split."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder
from AudioRouter.train_adbt_videomme import (
    build_inputs,
    encode_visual,
    make_prompt,
    option_token_ids,
    row_video_path,
    sample_video_for_rows,
)


LOGGER = logging.getLogger("eval_adbt_mcqa")


def ratio(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def summarize(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("task_type") or "all")].append(record)

    def one(rows: list[dict]) -> dict:
        return {
            "examples": len(rows),
            "greedy_correct": sum(int(row["greedy_correct"]) for row in rows),
            "greedy_accuracy": ratio(
                sum(int(row["greedy_correct"]) for row in rows), len(rows)
            ),
            "option_correct": sum(int(row["option_correct"]) for row in rows),
            "option_accuracy": ratio(
                sum(int(row["option_correct"]) for row in rows), len(rows)
            ),
            "mean_loss": sum(float(row["loss"]) for row in rows) / max(1, len(rows)),
        }

    return {
        "overall": one(records),
        "by_task_type": {name: one(rows) for name, rows in sorted(groups.items())},
    }


def evaluate(args) -> dict:
    manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    allowed = {str(value) for value in split["test"]}
    rows = [row for row in manifest["rows"] if str(row["videoID"]) in allowed]
    rows_by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_video[str(row["videoID"])].append(row)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    pretrained = args.pretrained or saved_args.get(
        "pretrained", "lmms-lab/llava-onevision-qwen2-7b-ov"
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained,
        None,
        get_model_name_from_path(pretrained),
        device_map="auto",
        multimodal=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    visual_dim = int(model.config.hidden_size)
    if args.attention_temperatures:
        temperatures = [
            float(value.strip())
            for value in args.attention_temperatures.split(",")
            if value.strip()
        ]
        if not temperatures:
            raise ValueError("--attention-temperatures must contain at least one value")
        if args.attention_temperature is not None:
            raise ValueError(
                "use either --attention-temperature or --attention-temperatures, not both"
            )
    else:
        temperatures = [
            float(
                args.attention_temperature
                if args.attention_temperature is not None
                else saved_args.get("attention_temperature", 0.05)
            )
        ]
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("--attention-temperatures contains duplicate values")
    if any(value <= 0 for value in temperatures):
        raise ValueError("attention temperatures must be positive")

    adbt = AudioConditionedBottleneck(
        visual_dim=visual_dim,
        num_queries=int(saved_args.get("num_queries", 64)),
        hidden_size=int(saved_args.get("bottleneck_hidden", 256)),
        num_heads=int(saved_args.get("num_heads", 8)),
        stage=int(saved_args.get("bottleneck_stage", 3)),
        latent_norm=saved_args.get("latent_norm", "none"),
        value_mode=saved_args.get("value_mode", "native"),
        architecture=saved_args.get("architecture", "phase4"),
        attention_temperature=temperatures[0],
    )
    missing, unexpected = adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    adapter_device = next(model.get_vision_tower().parameters()).device
    adbt.to(device=adapter_device, dtype=torch.float32).eval()
    beats_checkpoint = args.beats_checkpoint or saved_args.get("beats_checkpoint")
    audio_encoder = BEATsAudioEncoder(
        beats_checkpoint,
        batch_size=args.beats_batch_size,
        output_mode="temporal",
    )
    answer_option_ids = option_token_ids(tokenizer)
    records_by_temperature = {temperature: [] for temperature in temperatures}
    frames_seen = 0

    with torch.no_grad():
        for video_index, (video_id, video_rows) in enumerate(
            sorted(rows_by_video.items()), start=1
        ):
            path = row_video_path(video_rows[0])
            frames, timestamps, row_frame_positions = sample_video_for_rows(
                path, video_rows, fps=args.fps, max_frames=args.max_frames
            )
            visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
            audio = audio_encoder.encode_video(
                path, timestamps, visual.device, ablation=args.audio_ablation
            )
            audio_embeddings = audio["embeddings"].to(visual.device, dtype=torch.float32)
            audio_timestamps = audio["end_timestamps"].to(visual.device)
            frames_seen += len(timestamps)
            for temperature in temperatures:
                adbt.attention_temperature = temperature
                latents, _ = adbt(
                    visual.float(),
                    audio_embeddings=audio_embeddings,
                    end_timestamps=audio_timestamps,
                )
                records = records_by_temperature[temperature]
                for row_index, row in enumerate(video_rows):
                    positions = torch.as_tensor(
                        row_frame_positions[row_index], device=latents.device
                    )
                    prefix_frames = int(positions.numel())
                    prompt_ids = make_prompt(row, tokenizer)
                    embeds, labels, mask = build_inputs(
                        model,
                        tokenizer,
                        prompt_ids,
                        row["answer"],
                        latents.index_select(0, positions),
                    )
                    output = model.model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    shifted_labels = labels[:, 1:].to(output.last_hidden_state.device)
                    answer_mask = shifted_labels.ne(-100)
                    answer_hidden = output.last_hidden_state[:, :-1][answer_mask]
                    answer_labels = shifted_labels[answer_mask]
                    logits = model.lm_head(answer_hidden)[0].float()
                    loss = F.cross_entropy(logits[None], answer_labels[:1])
                    greedy_token = int(logits.argmax().item())
                    greedy_text = tokenizer.decode([greedy_token]).strip()
                    selected_option_logits = logits[
                        answer_option_ids.to(logits.device)
                    ].float()
                    option_probabilities = selected_option_logits.softmax(dim=0)
                    option_index = int(selected_option_logits.argmax().item())
                    option_prediction = "ABCD"[option_index]
                    top_values = selected_option_logits.topk(k=2).values
                    target = str(row["answer"]).strip().upper()
                    records.append(
                        {
                            "dataset": manifest.get("dataset_name"),
                            "videoID": video_id,
                            "question_id": row.get("question_id"),
                            "task_type": row.get("task_type"),
                            "query_time_seconds": row.get("query_time_seconds"),
                            "sampled_prefix_frames": prefix_frames,
                            "target": target,
                            "greedy_prediction": greedy_text,
                            "greedy_correct": greedy_text == target,
                            "option_prediction": option_prediction,
                            "option_correct": option_prediction == target,
                            "option_logits": {
                                letter: float(value)
                                for letter, value in zip(
                                    "ABCD", selected_option_logits.detach().cpu().tolist()
                                )
                            },
                            "option_probabilities": {
                                letter: float(value)
                                for letter, value in zip(
                                    "ABCD", option_probabilities.detach().cpu().tolist()
                                )
                            },
                            "option_top2_margin": float(
                                (top_values[0] - top_values[1]).detach().cpu()
                            ),
                            "loss": float(loss.cpu()),
                        }
                    )
                del latents
            LOGGER.info(
                "video=%d/%d id=%s frames=%d questions=%d",
                video_index,
                len(rows_by_video),
                video_id,
                len(timestamps),
                len(video_rows),
            )
            del visual, audio, audio_embeddings, audio_timestamps, frames
            torch.cuda.empty_cache()

    metadata = {
        "dataset": manifest.get("dataset_name"),
        "manifest": str(Path(args.dataset_manifest).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_videos": len(rows_by_video),
        "sampled_frames": frames_seen,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "audio_ablation": args.audio_ablation,
        "cuda_peak_allocated_mib": {
            str(index): torch.cuda.max_memory_allocated(index) / 1024**2
            for index in range(torch.cuda.device_count())
        },
        "cuda_peak_reserved_mib": {
            str(index): torch.cuda.max_memory_reserved(index) / 1024**2
            for index in range(torch.cuda.device_count())
        },
    }
    if len(temperatures) == 1:
        records = records_by_temperature[temperatures[0]]
        summary = summarize(records)
        summary.update(metadata)
        summary["attention_temperature"] = temperatures[0]
        summary["records"] = records
    else:
        summary = dict(metadata)
        summary["attention_temperatures"] = temperatures
        summary["temperature_results"] = {}
        for temperature in temperatures:
            records = records_by_temperature[temperature]
            result = summarize(records)
            result["records"] = records
            summary["temperature_results"][str(temperature)] = result
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--beats-checkpoint", default="")
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument("--vision-batch-size", type=int, default=8)
    parser.add_argument("--fps", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--audio-ablation", choices=("real", "zero", "shuffled", "stale"), default="real")
    parser.add_argument("--attention-temperature", type=float, default=None)
    parser.add_argument(
        "--attention-temperatures",
        default="",
        help="comma-separated inference temperatures; reuses visual/audio encoding",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = evaluate(parse_args())
    printable = {key: value for key, value in result.items() if key != "records"}
    if "temperature_results" in printable:
        printable["temperature_results"] = {
            temperature: {
                key: value for key, value in temperature_result.items() if key != "records"
            }
            for temperature, temperature_result in printable["temperature_results"].items()
        }
    print(json.dumps(printable, indent=2))

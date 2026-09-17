#!/usr/bin/env python3
"""Render an AudioRouter MCQA inference as an annotated MP4.

The video shows the query-independent audio-conditioned routing map used to
compress each sampled frame.  Audio is used only to choose visual tokens; it
is never inserted into the language-model token stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger("render_inference_demo")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEATS = "/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AudioRouter MCQA inference and render its routing map."
    )
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--options",
        nargs=4,
        required=True,
        metavar=("A", "B", "C", "D"),
        help="Exactly four answer options; letter prefixes are optional.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "ckpt" / "videomme_adbt_epoch3.pt"),
    )
    parser.add_argument("--output", default=str(ROOT / "results" / "demo" / "audiorouter_inference.mp4"))
    parser.add_argument("--pretrained", default="")
    parser.add_argument(
        "--beats-checkpoint",
        default=os.getenv("BEATS_CHECKPOINT", DEFAULT_BEATS),
    )
    parser.add_argument("--attention-temperature", type=float, default=None)
    parser.add_argument(
        "--query-time",
        type=float,
        default=None,
        help="Only use and render media up to this causal boundary in seconds.",
    )
    parser.add_argument("--fps", default="auto", help="Model sampling rate.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--vision-batch-size", type=int, default=8)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument("--render-fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--final-card-seconds",
        type=float,
        default=2.5,
        help="Duration of the final prediction card.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Copy source audio into the rendered MP4 when FFmpeg is available.",
    )
    return parser.parse_args()


def normalized_options(values: list[str]) -> list[str]:
    options = []
    for letter, value in zip("ABCD", values):
        clean = re.sub(rf"^\s*{letter}\s*[.):\-]\s*", "", str(value), flags=re.I)
        options.append(f"{letter}. {clean.strip()}")
    return options


def load_adapter(
    checkpoint_path: Path,
    model,
    temperature_override: float | None,
) -> tuple[torch.nn.Module, dict, float]:
    from AudioRouter.audio_bottleneck import AudioConditionedBottleneck

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    temperature = float(
        temperature_override
        if temperature_override is not None
        else saved.get("attention_temperature", 0.05)
    )
    adapter = AudioConditionedBottleneck(
        visual_dim=int(model.config.hidden_size),
        num_queries=int(saved.get("num_queries", 64)),
        hidden_size=int(saved.get("bottleneck_hidden", 256)),
        num_heads=int(saved.get("num_heads", 8)),
        stage=int(saved.get("bottleneck_stage", 3)),
        latent_norm=saved.get("latent_norm", "none"),
        value_mode=saved.get("value_mode", "native"),
        architecture=saved.get("architecture", "phase4"),
        attention_temperature=temperature,
    )
    adapter.load_state_dict(checkpoint["audio_bottleneck"], strict=True)
    return adapter, saved, temperature


def run_inference(args: argparse.Namespace) -> dict:
    video = Path(args.video).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if args.query_time is not None and args.query_time <= 0:
        raise ValueError("--query-time must be positive")
    if args.render_fps <= 0 or args.width < 640 or args.height < 360:
        raise ValueError("render FPS must be positive and canvas must be at least 640x360")

    # Keep heavyweight imports lazy so `--help` and argument validation never
    # initialize tokenizers or touch a model registry.
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    from AudioRouter.audio_bottleneck import BEATsAudioEncoder
    from AudioRouter.train_adbt_videomme import (
        build_inputs,
        encode_visual,
        make_prompt,
        option_token_ids,
        sample_video,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    pretrained = args.pretrained or saved.get(
        "pretrained", "lmms-lab/llava-onevision-qwen2-7b-ov"
    )
    LOGGER.info("loading frozen VLM: %s", pretrained)
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

    adapter, saved, temperature = load_adapter(
        checkpoint_path, model, args.attention_temperature
    )
    adapter_device = next(model.get_vision_tower().parameters()).device
    adapter.to(device=adapter_device, dtype=torch.float32).eval()

    beats_checkpoint = args.beats_checkpoint or saved.get("beats_checkpoint", "")
    audio_encoder = BEATsAudioEncoder(
        beats_checkpoint,
        batch_size=args.beats_batch_size,
        beats_root=str(ROOT / "unilm" / "beats"),
        output_mode="temporal",
    )
    if audio_encoder.source != "beats":
        raise FileNotFoundError(
            "a real BEATs checkpoint is required for the inference demo; set "
            "BEATS_CHECKPOINT or pass --beats-checkpoint"
        )

    LOGGER.info("sampling causal video prefix")
    frames, timestamps = sample_video(
        str(video),
        fps=args.fps,
        max_frames=args.max_frames,
        end_time_seconds=args.query_time,
    )
    options = normalized_options(args.options)
    row = {"question": args.question, "options": options}

    with torch.inference_mode():
        visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
        audio = audio_encoder.encode_video(
            str(video), timestamps, visual.device, ablation="real"
        )
        audio_embeddings = audio["embeddings"].to(visual.device, dtype=torch.float32)
        audio_timestamps = audio["end_timestamps"].to(visual.device)
        latents, attention = adapter(
            visual.float(),
            audio_embeddings=audio_embeddings,
            end_timestamps=audio_timestamps,
        )

        prompt_ids = make_prompt(row, tokenizer)
        embeds, labels, mask = build_inputs(
            model, tokenizer, prompt_ids, "A", latents
        )
        output = model.model(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        shifted_labels = labels[:, 1:].to(output.last_hidden_state.device)
        answer_hidden = output.last_hidden_state[:, :-1][shifted_labels.ne(-100)]
        logits = model.lm_head(answer_hidden)[0].float()
        selected = logits[option_token_ids(tokenizer).to(logits.device)]
        probabilities = selected.softmax(dim=0).detach().cpu().numpy()

    prediction_index = int(np.argmax(probabilities))
    prediction = "ABCD"[prediction_index]
    # Average the query rows: each value is the amount of routed visual mass
    # assigned to a native patch in that sampled frame.
    importance = attention.detach().float().mean(dim=1).cpu().numpy()
    metadata = {
        "video": str(video),
        "checkpoint": str(checkpoint_path),
        "pretrained": pretrained,
        "question": args.question,
        "options": options,
        "prediction": prediction,
        "predicted_option": options[prediction_index],
        "option_probabilities": {
            letter: float(value) for letter, value in zip("ABCD", probabilities)
        },
        "attention_temperature": temperature,
        "query_time_seconds": args.query_time,
        "model_sampling_fps": args.fps,
        "model_max_frames": args.max_frames,
        "sampled_frames": int(len(timestamps)),
        "sampled_timestamps": [float(value) for value in timestamps],
        "audio_encoder": audio["source"],
        "audio_enters_llm": False,
        "adapter_diagnostics": adapter.last_diagnostics,
        "routing_importance": importance,
    }
    del model, adapter, visual, latents, attention
    torch.cuda.empty_cache()
    return metadata


def resize_with_letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - size[0]) // 2
    y = (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas


def routing_overlay(frame: np.ndarray, weights: np.ndarray) -> np.ndarray:
    token_count = int(weights.size)
    grid = int(round(math.sqrt(token_count)))
    if grid * grid == token_count:
        heat = weights.reshape(grid, grid)
    else:
        heat = weights.reshape(1, token_count)
    span = float(np.ptp(heat))
    normalized = np.zeros_like(heat, dtype=np.float32) if span < 1e-12 else (heat - heat.min()) / span
    normalized = cv2.resize(
        normalized.astype(np.float32),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.62, color, 0.38, 0)


def draw_text_block(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    width_chars: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> int:
    line_height = max(20, int(30 * scale))
    for line in textwrap.wrap(str(text), width=max(12, width_chars)) or [""]:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return y


def compose_frame(
    source: np.ndarray,
    weights: np.ndarray,
    timestamp: float,
    duration: float,
    metadata: dict,
    width: int,
    height: int,
) -> np.ndarray:
    panel_width = max(390, int(width * 0.34))
    video_width = width - panel_width
    video = resize_with_letterbox(source, video_width, height)
    video = routing_overlay(video, weights)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :video_width] = video
    canvas[:, video_width:] = (22, 24, 28)

    cv2.rectangle(canvas, (0, 0), (video_width, 58), (10, 10, 10), -1)
    cv2.putText(canvas, "AudioRouter | audio-guided visual routing", (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    x, y = video_width + 24, 44
    cv2.putText(canvas, "MODEL INFERENCE", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (93, 220, 255), 2, cv2.LINE_AA)
    y = draw_text_block(canvas, metadata["question"], x, y + 42, 42, 0.57, (245, 245, 245), 1)
    y += 15
    for letter, option in zip("ABCD", metadata["options"]):
        probability = metadata["option_probabilities"][letter]
        selected = letter == metadata["prediction"]
        color = (94, 235, 138) if selected else (205, 205, 205)
        prefix = "> " if selected else "  "
        y = draw_text_block(canvas, f"{prefix}{option}", x, y, 39, 0.53, color, 2 if selected else 1)
        cv2.rectangle(canvas, (x, y + 2), (x + int((panel_width - 52) * probability), y + 9), color, -1)
        y += 28

    boundary = metadata["query_time_seconds"]
    label = f"causal prefix: {timestamp:6.2f}s / {duration:6.2f}s"
    cv2.putText(canvas, label, (x, height - 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (190, 190, 190), 1, cv2.LINE_AA)
    if boundary is not None:
        cv2.putText(canvas, f"query boundary: {boundary:.2f}s", (x, height - 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (93, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "audio routes; visual tokens answer", (x, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    progress = min(1.0, timestamp / max(duration, 1e-6))
    cv2.rectangle(canvas, (0, height - 8), (width, height), (45, 45, 45), -1)
    cv2.rectangle(canvas, (0, height - 8), (int(width * progress), height), (93, 220, 255), -1)
    return canvas


def mux_source_audio(
    silent_video: Path,
    source_video: Path,
    output: Path,
    prefix_duration: float,
    output_duration: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(silent_video),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            f"atrim=duration={prefix_duration:.6f},apad",
            "-t",
            f"{output_duration:.6f}",
            str(output),
        ]
        try:
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            LOGGER.warning("FFmpeg audio mux failed (%s); trying PyAV", exc)

    # PyAV is already required by the evaluation environment. Remux the AAC
    # packets without decoding, and stop exactly at the causal query boundary.
    try:
        import av

        with (
            av.open(str(silent_video)) as video_input,
            av.open(str(source_video)) as audio_input,
            av.open(str(output), mode="w") as muxed_output,
        ):
            video_stream = video_input.streams.video[0]
            audio_stream = next(
                (stream for stream in audio_input.streams if stream.type == "audio"),
                None,
            )
            if audio_stream is None:
                LOGGER.warning("source video has no audio stream; keeping output silent")
                return False
            output_video = muxed_output.add_stream_from_template(video_stream)
            output_audio = muxed_output.add_stream_from_template(audio_stream)
            for packet in video_input.demux(video_stream):
                if packet.dts is None:
                    continue
                packet.stream = output_video
                muxed_output.mux(packet)
            for packet in audio_input.demux(audio_stream):
                if packet.dts is None:
                    continue
                timestamp = float(packet.pts * packet.time_base) if packet.pts is not None else 0.0
                if timestamp >= prefix_duration:
                    break
                packet.stream = output_audio
                muxed_output.mux(packet)
        return True
    except Exception as exc:  # pragma: no cover - codec/container dependent
        output.unlink(missing_ok=True)
        LOGGER.warning("PyAV audio mux failed (%s); keeping output silent", exc)
        return False


def render_video(args: argparse.Namespace, metadata: dict) -> dict:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(metadata["video"])
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"invalid video metadata: fps={source_fps}, frames={total_frames}")
    full_duration = total_frames / source_fps
    duration = min(full_duration, metadata["query_time_seconds"]) if metadata["query_time_seconds"] else full_duration
    render_fps = min(float(args.render_fps), source_fps)

    handle = tempfile.NamedTemporaryFile(
        prefix=".audiorouter_demo_", suffix=".mp4", dir=output.parent, delete=False
    )
    silent_path = Path(handle.name)
    handle.close()
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        capture.release()
        silent_path.unlink(missing_ok=True)
        raise RuntimeError("OpenCV could not create the output MP4")

    timestamps = np.asarray(metadata["sampled_timestamps"], dtype=np.float32)
    importance = metadata.pop("routing_importance")
    next_output_time = 0.0
    last_canvas = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp > duration + 1e-6:
                break
            if timestamp + 0.5 / source_fps < next_output_time:
                continue
            routing_index = int(np.abs(timestamps - timestamp).argmin())
            last_canvas = compose_frame(
                frame,
                importance[routing_index],
                timestamp,
                duration,
                metadata,
                args.width,
                args.height,
            )
            writer.write(last_canvas)
            next_output_time += 1.0 / render_fps

        if last_canvas is None:
            raise RuntimeError("no video frames were rendered")
        card = last_canvas.copy()
        overlay = card.copy()
        cv2.rectangle(overlay, (0, 0), (args.width, args.height), (8, 10, 14), -1)
        card = cv2.addWeighted(overlay, 0.82, card, 0.18, 0)
        result = f"Prediction: {metadata['predicted_option']}"
        cv2.putText(card, result, (55, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.15, (94, 235, 138), 3, cv2.LINE_AA)
        cv2.putText(card, "AudioRouter", (55, args.height // 2 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (93, 220, 255), 2, cv2.LINE_AA)
        for _ in range(max(0, round(args.final_card_seconds * render_fps))):
            writer.write(card)
    finally:
        capture.release()
        writer.release()

    final_duration = duration + max(0.0, args.final_card_seconds)
    audio_included = False
    if args.include_audio:
        audio_included = mux_source_audio(
            silent_path,
            source,
            output,
            prefix_duration=duration,
            output_duration=final_duration,
        )
    if not audio_included:
        os.replace(silent_path, output)
    else:
        silent_path.unlink(missing_ok=True)
    metadata.update(
        {
            "output_video": str(output),
            "render_fps": render_fps,
            "rendered_prefix_seconds": duration,
            "audio_included": audio_included,
        }
    )
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("prediction=%s video=%s metadata=%s", metadata["prediction"], output, sidecar)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = run_inference(args)
    metadata = render_video(args, metadata)
    print(
        json.dumps(
            {
                "prediction": metadata["prediction"],
                "predicted_option": metadata["predicted_option"],
                "option_probabilities": metadata["option_probabilities"],
                "output_video": metadata["output_video"],
                "audio_included": metadata["audio_included"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()

"""Measure the actual direct-backend costs requested by docs/experience.md.

This is deliberately separate from accuracy evaluation.  It runs the same
frozen SigLIP projector and (for ``audio``) BEATs + ADBT path, then measures a
cold decoder prefill for fixed stream lengths.  Audio never enters the decoder;
only the visual-derived ``Z = P V_native`` latents do.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers.generation.utils import GenerationMixin

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from AudioRouter.audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder
from AudioRouter.tasks.vision import VisionTask
from AudioRouter.train_adbt_videomme import sample_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("full", "audio"), required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", default="auto", choices=("auto", "auto2", "auto3"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument("--bottleneck-batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--text-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument(
        "--lengths", default="16,32,64,128,256,512",
        help="Comma-separated frame counts used for cold-prefill scaling.",
    )
    parser.add_argument(
        "--beats-checkpoint",
        default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
    )
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    return parser.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call):
    sync()
    start = time.perf_counter()
    value = call()
    sync()
    return value, time.perf_counter() - start


def load_adbt(path: str, model, queries: int, temperature: float):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    if int(saved.get("bottleneck_stage", 3)) != 3:
        raise ValueError("efficiency profiling requires a Stage-3 audio checkpoint")
    saved_queries = int(saved.get("num_queries", queries))
    if saved_queries != queries:
        raise ValueError(f"checkpoint has {saved_queries} queries, requested {queries}")
    adbt = AudioConditionedBottleneck(
        visual_dim=int(model.config.hidden_size),
        num_queries=saved_queries,
        hidden_size=int(saved.get("bottleneck_hidden", 256)),
        num_heads=int(saved.get("num_heads", 8)),
        stage=3,
        latent_norm=saved.get("latent_norm", "none"),
        value_mode=saved.get("value_mode", "native"),
        architecture=saved.get("architecture", "phase4"),
        attention_temperature=temperature,
    )
    adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=True)
    return adbt


def encode_visual_timed(model, image_processor, frames: np.ndarray, batch_size: int):
    processed, preprocess_sec = timed(
        lambda: image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
    )
    tower = model.get_vision_tower()
    projector = model.get_model().mm_projector
    task = VisionTask(None, None)
    chunks = []
    tower_sec = 0.0
    projector_sec = 0.0
    pool_sec = 0.0
    with torch.inference_mode():
        for start in range(0, len(processed), batch_size):
            batch = processed[start : start + batch_size].to(tower.device)

            def tower_call():
                output = tower(batch)
                return output[0] if isinstance(output, (tuple, list)) else output

            raw, elapsed = timed(tower_call)
            tower_sec += elapsed
            projected, elapsed = timed(lambda: projector(raw))
            projector_sec += elapsed
            pooled, elapsed = timed(lambda: task._apply_2d_pool(projected, stride=2))
            pool_sec += elapsed
            chunks.append(pooled)
            del batch, raw, projected
    return torch.cat(chunks), {
        "image_preprocess_sec": preprocess_sec,
        "vision_tower_sec": tower_sec,
        "visual_projector_sec": projector_sec,
        "spatial_pool_sec": pool_sec,
    }


def encode_adbt_timed(adbt, visual, audio, timestamps, batch_size: int):
    outputs = []
    total = 0.0
    with torch.inference_mode():
        for start in range(0, visual.shape[0], batch_size):
            end = min(start + batch_size, visual.shape[0])

            def call():
                latents, _ = adbt(
                    visual[start:end].float(),
                    audio[start:end].to(visual.device, dtype=torch.float32),
                    timestamps[start:end].to(visual.device),
                )
                return latents

            latents, elapsed = timed(call)
            total += elapsed
            outputs.append(latents)
    return torch.cat(outputs), total


def prefill_once(model, visual_tokens, text_tokens: int):
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    token_ids = torch.full((1, text_tokens), 1, dtype=torch.long, device=device)
    text_embeds = embedding(token_ids)
    inputs = torch.cat(
        (visual_tokens.reshape(1, -1, visual_tokens.shape[-1]).to(device, dtype), text_embeds),
        dim=1,
    )
    positions = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    sync()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.get_model()(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
    sync()
    prefill_sec = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model.lm_head(output.last_hidden_state[:, -1])
        _ = logits.argmax(-1)
    sync()
    first_projection_sec = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    del output, logits, inputs, positions, mask, text_embeds, token_ids
    sync()
    torch.cuda.empty_cache()
    return {
        "llm_prefill_sec": prefill_sec,
        "first_token_projection_sec": first_projection_sec,
        "ttft_sec": prefill_sec + first_projection_sec,
        "peak_allocated_mib": peak_mib,
    }


def generate_timed(model, visual_tokens, text_tokens: int, decode_tokens: int):
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    token_ids = torch.full((1, text_tokens), 1, dtype=torch.long, device=device)
    text_embeds = embedding(token_ids)
    inputs = torch.cat(
        (visual_tokens.reshape(1, -1, visual_tokens.shape[-1]).to(device, dtype), text_embeds),
        dim=1,
    )
    positions = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    sync()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = GenerationMixin.generate(
            model,
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            do_sample=False,
            num_beams=1,
            max_new_tokens=decode_tokens,
            use_cache=True,
        )
    sync()
    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    produced = int(generated.shape[-1])
    del generated, inputs, positions, mask, text_embeds, token_ids
    sync()
    torch.cuda.empty_cache()
    return elapsed, peak_mib, produced


def main():
    args = parse_args()
    if args.setting == "audio" and not Path(args.checkpoint).is_file():
        raise FileNotFoundError("--checkpoint is required for setting=audio")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted({int(value) for value in args.lengths.split(",") if value})

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        get_model_name_from_path(args.pretrained),
        device_map="auto",
        multimodal=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    del tokenizer
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frames, timestamps_np = sample_video(
        args.video, fps=args.fps, max_frames=args.max_frames
    )
    visual, components = encode_visual_timed(
        model, image_processor, frames, args.vision_batch_size
    )
    timestamps = torch.as_tensor(timestamps_np, dtype=torch.float32)
    components["audio_encoder_sec"] = 0.0
    components["bottleneck_sec"] = 0.0

    if args.setting == "audio":
        adbt = load_adbt(
            args.checkpoint, model, args.num_queries, args.temperature
        ).to(device=visual.device, dtype=torch.float32).eval()
        # A run-specific cache records the exact embeddings but cannot turn a
        # first-run BEATs measurement into an accidental historical cache hit.
        cache_dir = output.parent / f".{output.stem}_beats_cache"
        audio_encoder = BEATsAudioEncoder(
            args.beats_checkpoint,
            batch_size=args.beats_batch_size,
            output_mode="temporal",
            cache_dir=str(cache_dir),
        )
        audio, components["audio_encoder_sec"] = timed(
            lambda: audio_encoder.encode_video(
                args.video, timestamps_np, visual.device, ablation="real"
            )
        )
        latent, components["bottleneck_sec"] = encode_adbt_timed(
            adbt,
            visual,
            audio["embeddings"],
            audio["end_timestamps"],
            args.bottleneck_batch_size,
        )
        tokens_per_frame = args.num_queries
        audio_cache_hit = audio_encoder._cache_hits > 0
        del audio_encoder, audio, adbt
    else:
        latent = visual
        tokens_per_frame = int(visual.shape[1])
        audio_cache_hit = None

    scaling = []
    available_lengths = [value for value in lengths if value <= len(frames)]
    if not available_lengths:
        raise ValueError(f"only {len(frames)} frames sampled; none of {lengths} is available")
    # Warm up kernels without including it in any reported measurement.
    _ = prefill_once(model, latent[: min(16, len(latent))], args.text_tokens)
    for length in available_lengths:
        metrics = prefill_once(model, latent[:length], args.text_tokens)
        metrics.update(
            {
                "frames": length,
                "tokens_per_frame": tokens_per_frame,
                "visual_tokens_entering_llm": length * tokens_per_frame,
            }
        )
        scaling.append(metrics)

    full_metrics = prefill_once(model, latent, args.text_tokens)
    generation_sec, generation_peak_mib, generated_tokens = generate_timed(
        model, latent, args.text_tokens, args.decode_tokens
    )
    measured_decode_sec = max(0.0, generation_sec - full_metrics["ttft_sec"])
    full_metrics.update(
        {
            "frames": len(frames),
            "tokens_per_frame": tokens_per_frame,
            "visual_tokens_entering_llm": len(frames) * tokens_per_frame,
            "generation_sec": generation_sec,
            "generated_tokens": generated_tokens,
            "measured_decode_sec": measured_decode_sec,
            "generation_throughput_tokens_per_sec": (
                generated_tokens / measured_decode_sec if measured_decode_sec > 0 else None
            ),
            "generation_peak_allocated_mib": generation_peak_mib,
        }
    )
    components["front_end_total_sec"] = sum(
        components[key]
        for key in (
            "image_preprocess_sec", "vision_tower_sec", "visual_projector_sec",
            "spatial_pool_sec", "audio_encoder_sec", "bottleneck_sec",
        )
    )
    full_metrics["measured_e2e_to_first_token_sec"] = (
        components["front_end_total_sec"] + full_metrics["ttft_sec"]
    )
    full_metrics["measured_e2e_generation_sec"] = (
        components["front_end_total_sec"] + generation_sec
    )
    report = {
        "setting": args.setting,
        "video": str(Path(args.video).resolve()),
        "fps": args.fps,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "sampled_frames": len(frames),
        "temperature": args.temperature if args.setting == "audio" else None,
        "components": components,
        "full_stream": full_metrics,
        "scaling": scaling,
        "notes": {
            "cold_prefill": True,
            "audio_cache_hit": audio_cache_hit,
            "ttft_definition": "cold decoder prefill + final-position LM-head projection",
            "decode_time": (
                f"generation wall time minus separately measured TTFT for "
                f"{args.decode_tokens} greedy tokens"
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

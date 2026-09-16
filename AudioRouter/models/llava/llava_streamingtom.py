import os
import types
import uuid
from typing import Dict, Iterator

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

from llava.constants import IMAGE_TOKEN_INDEX

from ...core import AudioRouterCore
from ...pipeline import AudioRouterPipeline
from .modeling_qwen2_revise import Qwen2Attention
from ...audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder

def setup_llava_with_AudioRouter(model: nn.Module, AudioRouter_config) -> nn.Module:
    model.AudioRouter_config = AudioRouter_config
    model._AudioRouter_core_obj = AudioRouterCore()
    stage = AudioRouter_config.get('audio_bottleneck_stage', 0)
    if stage > 0:
        architecture = AudioRouter_config.get('audio_bottleneck_architecture', 'phase4')
        model_instance = model.get_model() if hasattr(model, 'get_model') else model
        visual_dim = getattr(getattr(model, 'config', None), 'hidden_size', None)
        if visual_dim is None:
            visual_dim = getattr(getattr(model_instance, 'config', None), 'hidden_size', None)
        if visual_dim is None:
            raise ValueError('Cannot infer LLaVA hidden size for audio bottleneck')
        model.audio_encoder = None
        if stage >= 2:
            model.audio_encoder = BEATsAudioEncoder(
                checkpoint_path=AudioRouter_config.get('beats_checkpoint', ''),
                sample_rate=AudioRouter_config.get('audio_sample_rate', 16000),
                window_seconds=AudioRouter_config.get('audio_window_seconds', 2.0),
                batch_size=AudioRouter_config.get('beats_batch_size', 8),
                output_mode='temporal' if architecture == 'phase4' else 'mean',
            )
        model.audio_bottleneck = AudioConditionedBottleneck(
            visual_dim=int(visual_dim),
            num_queries=AudioRouter_config.get('audio_bottleneck_num_queries', 32),
            hidden_size=AudioRouter_config.get('audio_bottleneck_hidden_size', 256),
            num_heads=AudioRouter_config.get('audio_bottleneck_num_heads', 8),
            stage=stage,
            latent_norm=AudioRouter_config.get('audio_bottleneck_latent_norm', 'rmsnorm'),
            value_mode=AudioRouter_config.get('audio_bottleneck_value_mode', 'projected'),
            architecture=architecture,
            attention_temperature=AudioRouter_config.get(
                'audio_bottleneck_temperature', 0.2
            ),
        )
        adapter_checkpoint = AudioRouter_config.get('audio_bottleneck_checkpoint', '')
        if adapter_checkpoint:
            checkpoint = torch.load(adapter_checkpoint, map_location='cpu', weights_only=False)
            checkpoint_args = checkpoint.get('args', {})
            checkpoint_stage = int(
                checkpoint_args.get('bottleneck_stage', 3)
                if isinstance(checkpoint_args, dict) else 3
            )
            if checkpoint_stage != stage:
                raise RuntimeError(
                    'ADBT checkpoint stage does not match runtime config: '
                    f'checkpoint={checkpoint_stage} runtime={stage}'
                )
            checkpoint_architecture = (
                checkpoint_args.get('architecture', 'legacy')
                if isinstance(checkpoint_args, dict) else 'legacy'
            )
            if checkpoint_architecture != architecture:
                raise RuntimeError(
                    'ADBT checkpoint architecture does not match runtime config: '
                    f'checkpoint={checkpoint_architecture} runtime={architecture}'
                )
            if architecture == 'phase4':
                checkpoint_temperature = float(
                    checkpoint_args.get('attention_temperature', 0.2)
                )
                runtime_temperature = float(
                    AudioRouter_config.get('audio_bottleneck_temperature', 0.2)
                )
                if checkpoint_temperature != runtime_temperature:
                    if not AudioRouter_config.get(
                        'audio_bottleneck_allow_temperature_override', False
                    ):
                        raise RuntimeError(
                            'ADBT checkpoint temperature does not match runtime config: '
                            f'checkpoint={checkpoint_temperature} runtime={runtime_temperature}. '
                            'Set AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE=1 only for an '
                            'explicit inference-only temperature sweep.'
                        )
                    print(
                        '[AudioRouter] inference-only temperature override: '
                        f'checkpoint={checkpoint_temperature}, runtime={runtime_temperature}'
                    )
            checkpoint_value_mode = (
                checkpoint_args.get('value_mode', 'projected')
                if isinstance(checkpoint_args, dict) else 'projected'
            )
            requested_value_mode = AudioRouter_config.get(
                'audio_bottleneck_value_mode', 'projected'
            )
            if checkpoint_value_mode != requested_value_mode:
                raise RuntimeError(
                    'ADBT checkpoint value mode does not match runtime config: '
                    f'checkpoint={checkpoint_value_mode} runtime={requested_value_mode}'
                )
            state_dict = checkpoint.get('audio_bottleneck', checkpoint)
            missing, unexpected = model.audio_bottleneck.load_state_dict(state_dict, strict=False)
            allowed_missing = {'latent_norm.weight'}
            invalid_missing = sorted(set(missing) - allowed_missing)
            if invalid_missing or unexpected:
                raise RuntimeError(
                    f'Invalid ADBT checkpoint: missing={invalid_missing}, unexpected={unexpected}'
                )
            if missing:
                print(
                    '[AudioRouter] initialized new latent stability parameters: '
                    f'{missing}'
                )
            print(f'[AudioRouter] loaded audio bottleneck checkpoint: {adapter_checkpoint}')
        # Audio extraction is requested by lmms-eval before generate() is
        # entered.  Place the adapter on the vision device now so the BEATs
        # wrapper also uses that GPU instead of silently running on CPU.
        vision_device = model.get_vision_tower().device
        model.audio_bottleneck.to(device=vision_device, dtype=torch.float32)
        model._audio_bottleneck_enabled = True
        model._audio_bottleneck_stage = stage
        model._audio_bottleneck_requires_audio = stage >= 2
        model.encode_audio_from_video = types.MethodType(_encode_audio_from_video, model)
        print(
            f'[AudioRouter] enabled audio bottleneck stage {stage}: '
            f'{model.audio_bottleneck.num_queries} latent slots, backend='
            f'{AudioRouter_config.get("audio_bottleneck_backend")}, value_mode='
            f'{AudioRouter_config.get("audio_bottleneck_value_mode")}, architecture='
            f'{architecture}, temperature='
            f'{AudioRouter_config.get("audio_bottleneck_temperature")}'
        )
    else:
        model._audio_bottleneck_enabled = False
        model._audio_bottleneck_requires_audio = False
    model._AudioRouter_direct_backend = (
        AudioRouter_config.get('audio_bottleneck_backend') == 'direct'
    )
    # The direct ADBT accuracy path never constructs or retrieves an OQM KV
    # cache.  Keep the checkpoint's native attention implementation so this
    # comparison changes only the visual representation supplied to LLaVA.
    if AudioRouter_config.get('audio_bottleneck_backend') != 'direct':
        _replace_attention_layers(model)
    model.original_generate = model.generate
    model.generate = types.MethodType(generate_with_AudioRouter, model)
    model.generate_with_AudioRouter_streaming = types.MethodType(generate_with_AudioRouter_streaming, model)
    model._is_AudioRouter_patched = True
    return model

def create_frame_generator_llava(video_input, config, audio_info=None) -> Iterator[Dict]:
    assert isinstance(video_input, torch.Tensor), f"Expected tensor input, got {type(video_input)}"

    if video_input.shape[0] == 0:
        return

    assert video_input.dim() == 4, f"Expected 4D tensor [T, C, H, W], got {video_input.dim()}D"
    batch_size = config.get('streaming_encoder_batch_size')
    total_frames = video_input.shape[0]

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        batch = {'frames': list(video_input[start:end].unbind(0)), 'grid_thw': None}
        if audio_info is not None:
            embeddings = audio_info['embeddings']
            timestamps = audio_info['end_timestamps']
            batch['audio_embeddings'] = embeddings[start:end]
            batch['audio_timestamps'] = timestamps[start:end]
        yield batch


def _encode_audio_from_video(self, video_path, end_timestamps):
    """Encode only causal windows ending at the sampled visual timestamps."""
    timestamps = tuple(float(value) for value in end_timestamps)
    ablation = self.AudioRouter_config.get('audio_ablation', 'real')
    crossvideo_path = self.AudioRouter_config.get('audio_crossvideo_path', '')
    cache_key = (str(video_path), timestamps, ablation, crossvideo_path)
    audio_cache = getattr(self, '_direct_audio_cache', None)
    if audio_cache is None:
        audio_cache = {}
        self._direct_audio_cache = audio_cache
    if cache_key in audio_cache:
        return audio_cache[cache_key]

    device = torch.device('cpu')
    if self.audio_bottleneck is not None:
        device = next(self.audio_bottleneck.parameters()).device
    audio_info = self.audio_encoder.encode_video(
        video_path,
        timestamps,
        device=device,
        ablation=ablation,
        crossvideo_path=crossvideo_path,
    )
    # VideoMME has three questions per video.  Reusing causal BEATs embeddings
    # for those questions is exact and avoids decoding the same waveform three
    # times. lmms-eval sorts requests by prompt length, so keep a bounded CPU
    # cache rather than assuming the three questions remain adjacent.
    audio_info['video_path'] = str(video_path)
    if len(audio_cache) >= 256:
        audio_cache.pop(next(iter(audio_cache)))
    audio_cache[cache_key] = audio_info
    return audio_info

def generate_with_AudioRouter(self, *args, **kwargs):
    input_ids = args[0]
    images = kwargs.get('images')

    if images is None:
        return self.original_generate(*args, **kwargs)
    images = images[0]

    if self.AudioRouter_config.get('audio_bottleneck_backend') == 'direct':
        measure_memory = os.getenv('AudioRouter_MEASURE_MEMORY', '0') == '1'
        direct_kwargs = dict(kwargs)
        direct_kwargs.pop('images', None)
        if measure_memory:
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)
        result = _generate_with_direct_adbt(self, input_ids, images, **direct_kwargs)
        if measure_memory:
            peaks = []
            for device_index in range(torch.cuda.device_count()):
                peak_mib = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
                peaks.append(f'cuda:{device_index}={peak_mib:.1f} MiB')
            print('[AudioRouter DIRECT MEMORY] Peak allocated: ' + ', '.join(peaks))
        return result

    if not hasattr(self, '_AudioRouter_pipeline'):
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else kwargs.get('tokenizer')
        assert tokenizer is not None, "tokenizer is required for AudioRouter inference"
        self._AudioRouter_pipeline = AudioRouterPipeline(model=self, tokenizer=tokenizer)
    else:
        self._AudioRouter_pipeline.reset_AudioRouter_state()

    video_id = f"video_{uuid.uuid4().hex[:8]}"
    audio_info = kwargs.pop('audio_info', None)
    frame_batches = list(create_frame_generator_llava(images, self.AudioRouter_config, audio_info))
    questions = [{'batch_idx': -1, 'input_ids': input_ids}]
    answers = self._AudioRouter_pipeline.process_video(video_id, frame_batches, questions, **kwargs)
    self._AudioRouter_pipeline.AudioRouter_core.clear_cache(video_id)

    if os.getenv('AudioRouter_PROFILE', '0') == '1':
        from ...utils.profiler import get_profiler
        profiler = get_profiler()
        profiler.print_summary()

    return answers[0]['answer'] if answers else None


def _encode_direct_adbt_latents(self, images, audio_info):
    """Encode ordered frames into direct visual tokens without CTR or OQM."""
    if audio_info is None and getattr(self, '_audio_bottleneck_requires_audio', False):
        raise ValueError('Direct ADBT stage >=2 requires audio_info')

    vision_tower = self.get_vision_tower()
    model_instance = self.get_model() if hasattr(self, 'get_model') else self
    mm_projector = getattr(model_instance, 'mm_projector', None)
    latent_batches = []
    for frame_batch in create_frame_generator_llava(
        images, self.AudioRouter_config, audio_info
    ):
        processed_features, _ = self._AudioRouter_core_obj.encode_vision_batch(
            frame_batch, vision_tower, mm_projector
        )
        if getattr(self, '_audio_bottleneck_enabled', False):
            audio_embeddings = frame_batch.get('audio_embeddings')
            timestamps = frame_batch.get('audio_timestamps')
            phase4 = getattr(self.audio_bottleneck, 'architecture', 'legacy') == 'phase4'
            bottleneck_dtype = torch.float32 if phase4 else processed_features.dtype
            self.audio_bottleneck.to(
                device=processed_features.device, dtype=bottleneck_dtype
            )
            if audio_embeddings is not None:
                audio_embeddings = audio_embeddings.to(
                    device=processed_features.device, dtype=bottleneck_dtype
                )
            if timestamps is not None:
                timestamps = timestamps.to(processed_features.device)
            with torch.inference_mode():
                latents, _ = self.audio_bottleneck(
                    processed_features.to(dtype=bottleneck_dtype),
                    audio_embeddings=audio_embeddings,
                    end_timestamps=timestamps,
                )
            latent_batches.append(latents)
        else:
            latent_batches.append(processed_features)

    if not latent_batches:
        raise ValueError('Direct ADBT received no video frames')
    direct_features = torch.cat(latent_batches, dim=0).reshape(
        -1, latent_batches[0].shape[-1]
    )
    if not getattr(self, '_audio_bottleneck_enabled', False):
        # Match LLaVA-OneVision's spatial_unpad + one_token video contract.
        merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        newline_position = getattr(self.config, 'mm_newline_position', 'one_token')
        if merge_type.startswith('spatial') and newline_position == 'one_token' and 'unpad' in merge_type:
            direct_features = torch.cat(
                (direct_features, self.get_model().image_newline[None].to(direct_features)),
                dim=0,
            )
    return direct_features


def _generate_with_direct_adbt(self, input_ids, images, **kwargs):
    """Generate from visual-derived ADBT latents using native LLaVA attention."""
    audio_info = kwargs.pop('audio_info', None)
    direct_video_path = kwargs.pop('direct_video_path', None)
    direct_timestamps = kwargs.pop('direct_video_timestamps', None)
    kwargs.pop('images', None)
    kwargs.pop('image_sizes', None)
    kwargs.pop('modalities', None)
    kwargs.pop('tokenizer', None)
    kwargs.pop('attention_mask', None)
    position_ids = kwargs.pop('position_ids', None)

    # Release the previous video's full-precision prefix KV before allocating
    # features for a new video. Holding both very long prefixes concurrently
    # can otherwise double the transient peak and OOM on VideoMME-Long.
    if direct_video_path is not None:
        next_video_identity = (
            str(direct_video_path),
            tuple(float(value) for value in direct_timestamps),
            tuple(images.shape),
        )
        previous_prefix_key = getattr(self, '_direct_prefix_cache_key', None)
        if previous_prefix_key is not None and previous_prefix_key[:3] != next_video_identity:
            self._direct_prefix_cache_key = None
            self._direct_prefix_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cache_key = None
    if audio_info is not None and audio_info.get('video_path') is not None:
        timestamps = audio_info.get('end_timestamps')
        cache_key = (
            str(audio_info['video_path']),
            tuple(float(value) for value in timestamps),
            tuple(images.shape),
        )
    elif direct_video_path is not None:
        cache_key = (
            str(direct_video_path),
            tuple(float(value) for value in direct_timestamps),
            tuple(images.shape),
        )
    latent_cache = getattr(self, '_direct_latent_cache', None)
    if latent_cache is None:
        latent_cache = {}
        self._direct_latent_cache = latent_cache
    if cache_key is not None and cache_key in latent_cache:
        latents = latent_cache[cache_key]
    else:
        latents = _encode_direct_adbt_latents(self, images, audio_info)
        if cache_key is not None:
            # VideoMME's three questions for a video are contiguous. Retaining
            # older videos can consume tens of GiB of host RAM at high FPS and
            # provides no reuse, so this cache is deliberately one-video-only.
            # Move the live tensor to CPU as well; otherwise the first question
            # keeps both the float32 source and float16 decoder copy on GPU.
            latents = latents.detach().cpu()
            latent_cache.clear()
            latent_cache[cache_key] = latents
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError('Direct ADBT generation currently requires batch size 1')
    marker_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if marker_positions.numel() != 1:
        raise ValueError(
            f'Direct ADBT expects one image marker, got {marker_positions.numel()}'
        )
    marker = int(marker_positions.item())
    embedding = self.get_input_embeddings()
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    prefix = embedding(input_ids[:, :marker].to(device))
    suffix = embedding(input_ids[:, marker + 1:].to(device))
    latent_embeds = latents.unsqueeze(0).to(device=device, dtype=dtype)
    inputs_embeds = torch.cat((prefix, latent_embeds, suffix), dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=device
    )
    if position_ids is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)

    # Cache the identical visual prefix shared by the three VideoMME questions.
    # Full Vision keeps its established default.  ADBT must be explicitly
    # enabled after an answer-equivalence smoke test because its long-video KV
    # is large. This is exact prefix reuse: latents, prompt suffix, decoding,
    # and metric are unchanged; audio still never enters the LLM.
    prefix_cache_enabled = (
        not getattr(self, '_audio_bottleneck_enabled', False)
        or os.getenv('AudioRouter_PREFIX_KV_CACHE', '0') == '1'
    )
    if prefix_cache_enabled and cache_key is not None:
        prefix_cache_key = cache_key + (tuple(int(value) for value in input_ids[0, :marker]),)
        prefix_past = None
        if getattr(self, '_direct_prefix_cache_key', None) == prefix_cache_key:
            prefix_past = self._direct_prefix_cache
        else:
            common_embeds = torch.cat((prefix, latent_embeds), dim=1)
            common_length = common_embeds.shape[1]
            prefill_chunk_size = int(
                os.getenv('AudioRouter_PREFIX_PREFILL_CHUNK_SIZE', '0')
            )
            # Call the decoder backbone, not the CausalLM head. The prefix
            # needs only KV; projecting every visual position to the vocabulary
            # would allocate tens of GiB of logits. Optional chronological
            # chunking further bounds decoder activations while constructing
            # exactly the same causal KV prefix.
            if 0 < prefill_chunk_size < common_length:
                prefix_past = None
                with torch.inference_mode():
                    for start in range(0, common_length, prefill_chunk_size):
                        end = min(start + prefill_chunk_size, common_length)
                        prefix_outputs = self.get_model()(
                            inputs_embeds=common_embeds[:, start:end],
                            attention_mask=torch.ones(
                                (1, end), dtype=torch.long, device=device
                            ),
                            position_ids=torch.arange(
                                start, end, dtype=torch.long, device=device
                            ).unsqueeze(0),
                            past_key_values=prefix_past,
                            use_cache=True,
                            return_dict=True,
                        )
                        prefix_past = prefix_outputs.past_key_values
                        del prefix_outputs
            else:
                common_mask = torch.ones(
                    (1, common_length), dtype=torch.long, device=device
                )
                common_positions = torch.arange(
                    common_length, dtype=torch.long, device=device
                ).unsqueeze(0)
                with torch.inference_mode():
                    prefix_outputs = self.get_model()(
                        inputs_embeds=common_embeds,
                        attention_mask=common_mask,
                        position_ids=common_positions,
                        use_cache=True,
                        return_dict=True,
                    )
                prefix_past = prefix_outputs.past_key_values
            if hasattr(prefix_past, 'to_legacy_cache'):
                prefix_past = prefix_past.to_legacy_cache()
            self._direct_prefix_cache_key = prefix_cache_key
            self._direct_prefix_cache = prefix_past

        common_length = prefix.shape[1] + latent_embeds.shape[1]
        full_mask = torch.ones(
            (1, common_length + suffix.shape[1]), dtype=torch.long, device=device
        )
        full_positions = torch.arange(
            common_length + suffix.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0)
        return GenerationMixin.generate(
            self,
            # GenerationMixin derives cache_position from the supplied embed
            # length and then removes the already cached prefix. Supplying the
            # full prompt here therefore selects exactly the uncached suffix.
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=full_positions,
            past_key_values=DynamicCache.from_legacy_cache(prefix_past),
            **kwargs,
        )

    # LlavaQwenForCausalLM.generate rejects externally supplied inputs_embeds;
    # invoke the standard HF generation implementation after multimodal
    # preparation has been performed explicitly above.
    return GenerationMixin.generate(
        self,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **kwargs,
    )

def generate_with_AudioRouter_streaming(self, images, questions, video_id=None, is_start=False, **kwargs):
    assert video_id is not None, "video_id must be provided"

    if not hasattr(self, '_video_pipelines'):
        self._video_pipelines = {}

    needs_new_pipeline = is_start or video_id not in self._video_pipelines

    if needs_new_pipeline:
        if is_start:
            for vid, pipeline in list(self._video_pipelines.items()):
                pipeline.AudioRouter_core.clear_cache(vid)
            self._video_pipelines.clear()

        tokenizer = getattr(self, 'tokenizer', None) or kwargs.get('tokenizer')
        assert tokenizer is not None, "tokenizer is required for AudioRouter inference"
        self._video_pipelines[video_id] = AudioRouterPipeline(model=self, tokenizer=tokenizer)

    pipeline = self._video_pipelines[video_id]

    if images is not None and images.numel() > 0:
        frame_batches = list(create_frame_generator_llava(images, self.AudioRouter_config, kwargs.pop('audio_info', None)))
    else:
        frame_batches = []

    answers = pipeline.process_video(video_id, frame_batches, questions, **kwargs)
    return answers

def _replace_attention_layers(model: nn.Module):
    if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
        raise ValueError(f"Cannot find model.model.layers, model type: {type(model)}")

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'self_attn'):
            continue
        old_attn = layer.self_attn
        new_attn = Qwen2Attention(config=old_attn.config, layer_idx=layer_idx).to(
            dtype=old_attn.q_proj.weight.dtype, device=old_attn.q_proj.weight.device)
        new_attn.load_state_dict(old_attn.state_dict())
        layer.self_attn = new_attn
        layer.self_attn._AudioRouter_context = None

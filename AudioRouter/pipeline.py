from typing import Any, Dict, Tuple

import torch


class AudioRouterPipeline:

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.model_forward = self.model.forward
        self.model_type = getattr(self.model, '_AudioRouter_model_type')
        assert self.model_type == 'llava', f"Unsupported model type {self.model_type}, only 'llava' is supported"
        self.AudioRouter_core = getattr(self.model, '_AudioRouter_core_obj')
        self.config = getattr(self.model, 'AudioRouter_config').config
        self.AudioRouter_state = {}
        self.AudioRouter_core.clear_cache()

    def process_video(self, video_id: str, frame_batches: list, questions: list = None, **kwargs):
        encoding_start_frame = kwargs.pop('encoding_start_frame', None)

        batch_info = self.AudioRouter_core.oqm.get_batch_info(video_id)
        cumulative_batch_offset = batch_info.get('batch_idx', 0)

        total_batches = len(frame_batches)

        for local_batch_idx, frame_batch in enumerate(frame_batches):
            batch_idx = local_batch_idx + cumulative_batch_offset
            is_first = (batch_idx == 0)

            frame_info = {
                'batch_type': {
                    'is_first': is_first,
                    'is_last': local_batch_idx == total_batches - 1,
                    'batch_idx': batch_idx
                },
                'local_batch_idx': local_batch_idx,
                'encoding_start_frame': encoding_start_frame
            }

            frame_batch['frame_info'] = frame_info
            self._process_single_batch(video_id, batch_idx, frame_batch, frame_info)

        if questions is None:
            return None

        answers = []
        for question in questions:
            q_kwargs = {k: v for k, v in question.items() if k not in ['batch_idx', 'input_ids']}
            q_kwargs.update(kwargs)
            answers.append({
                'answer': self._process_query(video_id, question['input_ids'], **q_kwargs)
            })

        return answers

    def reset_AudioRouter_state(self):
        self.AudioRouter_state = {}
        self.AudioRouter_core.clear_cache()

    def _process_single_batch(self, video_id: str, batch_idx: int, frame_batch: Dict, frame_info: Dict[str, Any]):
        processed_features, attention_scores = self._encode_frame_batch(frame_batch)
        self.AudioRouter_state = self.AudioRouter_core.process_vision_batch(
            video_id,
            processed_features,
            self.AudioRouter_state,
            attention_scores,
            batch_idx,
            self.model,
            frame_info
        )

    def _get_vision_tower(self):
        if hasattr(self.model, 'get_vision_tower'):
            return self.model.get_vision_tower()
        if hasattr(self.model, 'visual'):
            return self.model.visual
        return None

    def _encode_frame_batch(self, frame_batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_tower = self._get_vision_tower()
        mm_projector = None

        if self.model_type == "llava":
            model_instance = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
            mm_projector = getattr(model_instance, 'mm_projector', None)

        processed_features, attention_scores = self.AudioRouter_core.encode_vision_batch(
            frame_batch,
            vision_tower,
            mm_projector
        )
        # ADBT is deliberately upstream of CTR/OQM.  Its output has one fixed
        # latent group per frame and remains visual-derived (audio only changes
        # the query/attention weights).
        bottleneck = getattr(self.model, 'audio_bottleneck', None)
        stage = getattr(self.model, '_audio_bottleneck_stage', 0)
        if bottleneck is not None and stage > 0:
            audio_embeddings = frame_batch.get('audio_embeddings')
            timestamps = frame_batch.get('audio_timestamps')
            if stage >= 2 and audio_embeddings is None:
                raise ValueError('Audio bottleneck stage >=2 requires audio_embeddings in frame batch')
            if timestamps is not None:
                timestamps = timestamps.to(processed_features.device)
            phase4 = getattr(bottleneck, 'architecture', 'legacy') == 'phase4'
            bottleneck_dtype = torch.float32 if phase4 else processed_features.dtype
            if audio_embeddings is not None:
                audio_embeddings = audio_embeddings.to(
                    device=processed_features.device, dtype=bottleneck_dtype
                )
            # Accelerate may shard the frozen LLaVA modules across GPUs.  The
            # newly attached adapter is intentionally moved to the vision
            # tower's device on first use, avoiding cross-device matmul errors
            # while keeping decoded frames CPU-side.
            bottleneck.to(device=processed_features.device, dtype=bottleneck_dtype)
            with torch.no_grad():
                processed_features, _ = bottleneck(
                    processed_features.to(dtype=bottleneck_dtype),
                    audio_embeddings=audio_embeddings,
                    end_timestamps=timestamps,
                )
            # CTR consumes the original 196-token grid and is not implicitly
            # applied to ADBT.  The explicit ctr_oqm mode remains available as
            # a separately named ablation (and requires compatible dimensions).
            if self.config.get('audio_bottleneck_backend', 'oqm') == 'oqm':
                attention_scores = None
        return processed_features, attention_scores

    def _process_query(self, video_id: str, input_ids, **kwargs):
        self.AudioRouter_core.prepare_retrieval_context(
            video_id,
            input_ids,
            self.model,
            self.tokenizer
        )
        return self.AudioRouter_core.retrieve_and_generate(
            input_ids,
            self.model,
            self.tokenizer,
            **kwargs
        )

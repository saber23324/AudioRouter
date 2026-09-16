import os
from typing import Any, Dict


class AudioRouterConfig:

    def __init__(self):
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        value_mode = os.getenv('AUDIO_BOTTLENECK_VALUE_MODE', 'projected')
        default_latent_norm = 'none' if value_mode == 'native' else 'rmsnorm'
        config = {
            'ctr_similarity_threshold': float(os.environ['CTR_SIMILARITY_THRESHOLD']),
            'ctr_retain_tokens': int(os.environ['CTR_RETAIN_TOKENS']),
            'ctr_k': int(os.environ['CTR_K']),
            'ctr_beta': float(os.environ['CTR_BETA']),
            'oqm_retrieval_max_tokens': int(os.environ['OQM_RETRIEVAL_MAX_TOKENS']),
            'oqm_enable_quantization': os.environ['OQM_ENABLE_QUANTIZATION'] == '1',
            'oqm_quantization_bits': int(os.environ['OQM_QUANTIZATION_BITS']),
            'oqm_group_size': int(os.environ['OQM_GROUP_SIZE']),
            'oqm_init_token_count': int(os.environ['OQM_INIT_TOKEN_COUNT']),
            'oqm_sliding_window_size': int(os.environ['OQM_SLIDING_WINDOW_SIZE']),
            'streaming_encoder_batch_size': int(os.environ['STREAMING_ENCODER_BATCH_SIZE']),
            # Staged audio-conditioned bottleneck.  These options are opt-in so
            # the frozen AudioRouter reproduction path is unchanged.
            'audio_bottleneck_stage': int(os.getenv('AUDIO_BOTTLENECK_STAGE', '0')),
            'audio_bottleneck_num_queries': int(os.getenv('AUDIO_BOTTLENECK_NUM_QUERIES', '32')),
            'audio_bottleneck_hidden_size': int(os.getenv('AUDIO_BOTTLENECK_HIDDEN_SIZE', '256')),
            'audio_bottleneck_num_heads': int(os.getenv('AUDIO_BOTTLENECK_NUM_HEADS', '8')),
            'audio_bottleneck_backend': os.getenv('AUDIO_BOTTLENECK_BACKEND', 'oqm'),
            'audio_bottleneck_value_mode': value_mode,
            'audio_bottleneck_latent_norm': os.getenv('AUDIO_BOTTLENECK_LATENT_NORM', default_latent_norm),
            'audio_bottleneck_architecture': os.getenv('AUDIO_BOTTLENECK_ARCHITECTURE', 'phase4'),
            'audio_bottleneck_temperature': float(os.getenv('AUDIO_BOTTLENECK_TEMPERATURE', '0.2')),
            'audio_bottleneck_allow_temperature_override': os.getenv(
                'AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE', '0'
            ).lower() in {'1', 'true', 'yes', 'on'},
            'audio_window_seconds': float(os.getenv('AUDIO_WINDOW_SECONDS', '2.0')),
            'audio_sample_rate': int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
            'audio_ablation': os.getenv('AUDIO_ABLATION', 'real'),
            'audio_crossvideo_path': os.getenv('AUDIO_CROSSVIDEO_PATH', ''),
            'beats_checkpoint': os.getenv('BEATS_CHECKPOINT', ''),
            'beats_batch_size': int(os.getenv('BEATS_BATCH_SIZE', '8')),
            'audio_bottleneck_checkpoint': os.getenv('AUDIO_BOTTLENECK_CHECKPOINT', ''),
        }
        return config

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def _validate_config(self):
        assert self.config['ctr_retain_tokens'] == self.config['oqm_group_size'], \
            "ctr_retain_tokens must equal oqm_group_size"
        assert self.config['ctr_retain_tokens'] > 0, "ctr_retain_tokens must be > 0"
        assert 0 <= self.config['ctr_similarity_threshold'] <= 1, "ctr_similarity_threshold must be in [0, 1]"
        assert self.config['ctr_k'] > 0, "ctr_k must be > 0"
        assert 0 <= self.config['ctr_beta'] <= 1, "ctr_beta must be in [0, 1]"
        assert self.config['oqm_retrieval_max_tokens'] > 0, "oqm_retrieval_max_tokens must be > 0"
        assert isinstance(self.config['oqm_enable_quantization'], bool), "oqm_enable_quantization must be bool"
        assert self.config['oqm_quantization_bits'] in [2, 4], "oqm_quantization_bits must be 2 or 4"
        assert self.config['oqm_group_size'] > 0, "oqm_group_size must be > 0"
        assert self.config['oqm_init_token_count'] >= 0, "oqm_init_token_count must be >= 0"
        assert self.config['oqm_sliding_window_size'] > 0, "oqm_sliding_window_size must be > 0"
        assert self.config['streaming_encoder_batch_size'] > 0, "streaming_encoder_batch_size must be > 0"
        assert self.config['oqm_sliding_window_size'] % self.config['oqm_group_size'] == 0, \
            "oqm_sliding_window_size must be a multiple of oqm_group_size"
        assert self.config['audio_bottleneck_stage'] in [0, 1, 2, 3], \
            "audio_bottleneck_stage must be one of 0, 1, 2, 3"
        assert self.config['audio_bottleneck_num_queries'] > 0
        assert self.config['audio_bottleneck_hidden_size'] > 0
        assert self.config['audio_bottleneck_hidden_size'] % self.config['audio_bottleneck_num_heads'] == 0
        assert self.config['audio_bottleneck_backend'] in ['direct', 'oqm', 'ctr_oqm']
        assert self.config['audio_bottleneck_value_mode'] in ['projected', 'native']
        assert self.config['audio_bottleneck_latent_norm'] in ['none', 'rmsnorm', 'layernorm']
        assert self.config['audio_bottleneck_architecture'] in ['legacy', 'phase4']
        assert self.config['audio_bottleneck_temperature'] > 0
        assert isinstance(
            self.config['audio_bottleneck_allow_temperature_override'], bool
        )
        if self.config['audio_bottleneck_value_mode'] == 'native':
            assert self.config['audio_bottleneck_latent_norm'] == 'none', \
                "native value mode requires AUDIO_BOTTLENECK_LATENT_NORM=none"
        assert self.config['audio_window_seconds'] > 0
        assert self.config['audio_sample_rate'] == 16000, "BEATs expects 16 kHz audio"
        assert self.config['audio_ablation'] in [
            'real', 'zero', 'shuffled', 'stale', 'crossvideo'
        ]
        if self.config['audio_ablation'] == 'crossvideo':
            assert self.config['audio_crossvideo_path'], \
                "AUDIO_CROSSVIDEO_PATH is required for crossvideo ablation"

        if self.config['audio_bottleneck_stage'] > 0 and self.config['audio_bottleneck_backend'] == 'oqm':
            assert self.config['audio_bottleneck_num_queries'] == self.config['oqm_group_size'], \
                "ADBT-to-OQM currently stores one latent group per frame; num_queries must equal oqm_group_size"


_AudioRouter_CONFIG_INSTANCE = None


def AudioRouter(model, model_type: str):
    from .models.llava.llava_AudioRouter import setup_llava_with_AudioRouter
    config = reload_AudioRouter_config()
    _print_initialization_info(model_type, config)
    model_type = model_type.lower()
    if model_type == "llava":
        return _patch_model(model, config, setup_llava_with_AudioRouter, model_type)
    raise ValueError(f"Unsupported model type {model_type}, only 'llava' is supported")


def get_AudioRouter_config():
    global _AudioRouter_CONFIG_INSTANCE
    if _AudioRouter_CONFIG_INSTANCE is None:
        _AudioRouter_CONFIG_INSTANCE = AudioRouterConfig()
    return _AudioRouter_CONFIG_INSTANCE


def reload_AudioRouter_config():
    global _AudioRouter_CONFIG_INSTANCE
    _AudioRouter_CONFIG_INSTANCE = AudioRouterConfig()
    return _AudioRouter_CONFIG_INSTANCE


def _patch_model(model, config, setup_func, model_type: str):
    if getattr(model, '_is_AudioRouter_patched', False):
        return model
    model._AudioRouter_model_type = model_type
    setup_func(model, config)
    return model


def _print_initialization_info(model_type: str, config):
    print(f"\n{'='*60}")
    print(f"Model Type: {model_type}")
    for key, value in sorted(config.config.items()):
        print(f"  {key}: {value}")
    print("="*60 + "\n")

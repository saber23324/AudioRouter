"""Causal VideoMME audio-window extraction with an optional frozen BEATs model."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


class BEATsAudioEncoder(nn.Module):
    """Frozen BEATs wrapper with a deterministic smoke-test fallback.

    The wrapper auto-discovers the downloaded checkpoint in the local HF cache;
    ``BEATS_CHECKPOINT`` can override that location. When no checkpoint is
    available, it emits a 768-D deterministic acoustic-statistics embedding so
    Stage 2--4 plumbing can be tested without pretending that the fallback is a
    pretrained BEATs representation.
    """

    output_dim = 768

    def __init__(
        self,
        checkpoint_path: str = "",
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        batch_size: int = 8,
        beats_root: str = "/home/yxd/AudioRouter/unilm/beats",
        output_mode: str = "temporal",
        cache_dir: str = "",
    ):
        super().__init__()
        if output_mode not in ("temporal", "mean"):
            raise ValueError(
                f"output_mode must be temporal or mean, got {output_mode}"
            )
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.batch_size = batch_size
        self.output_mode = output_mode
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.beats_root = Path(beats_root)
        configured_cache = cache_dir or os.getenv("BEATS_EMBEDDING_CACHE", "")
        self.cache_dir = Path(configured_cache).expanduser() if configured_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
        self.beats = None
        self.source = "diagnostic_fallback"

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)
        else:
            LOGGER.warning(
                "No BEATs checkpoint found; using deterministic acoustic-statistics "
                "features for plumbing-only smoke testing."
            )

    @staticmethod
    def _resolve_checkpoint(checkpoint_path: str) -> str:
        """Resolve the downloaded HF module path without hard-coding one cache."""
        filename = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
        candidates = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path).expanduser())
        candidates.extend([
            Path("/nvme_data/pkt/huggingface/modules") / filename,
            Path("/home/yxd/hub/modules") / filename,
            Path("/home/yxd/AudioRouter/huggingface/modules/transformers_modules") / filename,
        ])
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return ""

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"BEATs checkpoint does not exist: {path}")

        # The vendored BEATs source uses top-level imports (``from backbone``),
        # so its own directory must be present while importing it.
        beats_root = str(self.beats_root)
        if beats_root not in sys.path:
            sys.path.insert(0, beats_root)
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = BEATs(BEATsConfig(checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.beats = model
        self.output_dim = int(model.cfg.encoder_embed_dim)
        self.source = "beats"

    @staticmethod
    def decode_audio(video_path: str, sample_rate: int = 16000) -> torch.Tensor:
        """Decode mono PCM on CPU. Feature windows remain causally truncated."""
        command = [
            "ffmpeg", "-v", "error", "-i", str(video_path), "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            pcm = np.frombuffer(completed.stdout, dtype=np.int16).copy()
            return torch.from_numpy(pcm).float().div_(32768.0)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            # PyAV is already part of the lmms-eval environment and provides a
            # portable fallback when a system ffmpeg executable is unavailable.
            try:
                import av
                container = av.open(str(video_path))
                stream = next((s for s in container.streams if s.type == "audio"), None)
                if stream is None:
                    return torch.zeros(0, dtype=torch.float32)
                resampler = av.audio.resampler.AudioResampler(
                    format="s16", layout="mono", rate=sample_rate
                )
                chunks = []
                for frame in container.decode(stream):
                    for resampled in resampler.resample(frame):
                        chunks.append(resampled.to_ndarray().reshape(-1))
                # Flush delayed samples in the resampler.
                for resampled in resampler.resample(None):
                    chunks.append(resampled.to_ndarray().reshape(-1))
                if not chunks:
                    return torch.zeros(0, dtype=torch.float32)
                pcm = np.concatenate(chunks).astype(np.float32, copy=False)
                return torch.from_numpy(pcm).div_(32768.0)
            except Exception as av_exc:
                LOGGER.warning("Could not decode audio from %s (%s; PyAV: %s)", video_path, exc, av_exc)
                return torch.zeros(0, dtype=torch.float32)

    def _causal_windows(
        self, waveform: torch.Tensor, end_timestamps: torch.Tensor
    ) -> torch.Tensor:
        window_samples = int(round(self.window_seconds * self.sample_rate))
        windows: List[torch.Tensor] = []
        for timestamp in end_timestamps.tolist():
            # Never read a sample whose timestamp is later than the visual
            # timestamp associated with this embedding.
            end = min(max(int(round(float(timestamp) * self.sample_rate)), 0), waveform.numel())
            start = max(0, end - window_samples)
            window = waveform[start:end]
            if window.numel() < window_samples:
                window = torch.nn.functional.pad(window, (window_samples - window.numel(), 0))
            windows.append(window)
        if not windows:
            return torch.empty(0, window_samples, dtype=torch.float32)
        return torch.stack(windows)

    def _fallback_features(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.numel() == 0:
            shape = (0, 1, self.output_dim) if self.output_mode == "temporal" else (0, self.output_dim)
            return torch.empty(*shape)
        eps = 1e-6
        signs = torch.sign(windows)
        zcr = (signs[:, 1:] != signs[:, :-1]).float().mean(dim=1)
        base = torch.stack(
            [
                windows.mean(dim=1),
                windows.std(dim=1),
                windows.square().mean(dim=1).add(eps).sqrt(),
                windows.abs().mean(dim=1),
                windows.amax(dim=1),
                windows.amin(dim=1),
                zcr,
                windows[:, -1],
            ],
            dim=-1,
        )
        features = torch.zeros(windows.shape[0], self.output_dim, dtype=torch.float32)
        features[:, : base.shape[1]] = base
        # The fallback is only for plumbing smoke tests. Preserve the temporal
        # rank expected by Phase 4 without pretending that hand-written audio
        # statistics are a BEATs token sequence.
        return features.unsqueeze(1) if self.output_mode == "temporal" else features

    def _embedding_cache_path(
        self, audio_source_path: str, timestamps: torch.Tensor
    ) -> Path | None:
        cache_dir = getattr(self, "cache_dir", None)
        if cache_dir is None or self.beats is None:
            return None
        source_path = Path(audio_source_path).resolve()
        source_stat = source_path.stat()
        checkpoint_path = Path(self.checkpoint_path).resolve()
        checkpoint_stat = checkpoint_path.stat()
        digest = hashlib.sha256()
        digest.update(str(source_path).encode("utf-8"))
        digest.update(f"{source_stat.st_size}:{source_stat.st_mtime_ns}".encode("ascii"))
        digest.update(str(checkpoint_path).encode("utf-8"))
        digest.update(
            f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}".encode("ascii")
        )
        digest.update(
            f"{self.sample_rate}:{self.window_seconds}:{self.output_mode}".encode("ascii")
        )
        digest.update(timestamps.contiguous().numpy().tobytes())
        return cache_dir / digest.hexdigest()[:2] / f"{digest.hexdigest()}.pt"

    def _load_embedding_cache(
        self, cache_path: Path | None, timestamps: torch.Tensor
    ) -> torch.Tensor | None:
        if cache_path is None or not cache_path.is_file():
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            cached_timestamps = payload["end_timestamps"]
            embeddings = payload["embeddings"]
            if not torch.equal(cached_timestamps, timestamps):
                raise ValueError("timestamp mismatch")
            if embeddings.shape[0] != timestamps.numel():
                raise ValueError("embedding count mismatch")
            self._cache_hits = getattr(self, "_cache_hits", 0) + 1
            if self._cache_hits == 1 or self._cache_hits % 100 == 0:
                LOGGER.info(
                    "BEATs embedding cache hit=%d miss=%d path=%s",
                    self._cache_hits,
                    getattr(self, "_cache_misses", 0),
                    cache_path,
                )
            return embeddings.float()
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Ignoring invalid BEATs cache entry %s: %s", cache_path, exc)
            return None

    def _save_embedding_cache(
        self,
        cache_path: Path | None,
        timestamps: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "end_timestamps": timestamps.cpu(),
                "embeddings": embeddings.float().cpu(),
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)

    @torch.inference_mode()
    def encode_windows(self, windows: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.beats is None:
            return self._fallback_features(windows)

        self.beats.to(device)
        outputs = []
        for start in range(0, windows.shape[0], self.batch_size):
            batch = windows[start : start + self.batch_size].to(device)
            padding_mask = torch.zeros_like(batch, dtype=torch.bool)
            # Fine-tuned BEATs checkpoints attach a 527-way AudioSet
            # predictor.  A_t must be the continuous 768-D encoder state, not
            # the classifier probabilities, so temporarily bypass that head.
            predictor = self.beats.predictor
            self.beats.predictor = None
            try:
                representation, representation_padding = self.beats.extract_features(
                    batch, padding_mask=padding_mask
                )
            finally:
                self.beats.predictor = predictor
            if self.output_mode == "mean":
                if representation_padding is not None:
                    valid = (~representation_padding).unsqueeze(-1)
                    representation = (
                        (representation * valid).sum(1)
                        / valid.sum(1).clamp_min(1)
                    )
                else:
                    representation = representation.mean(dim=1)
            elif representation_padding is not None:
                # Fixed-size causal windows normally have no BEATs padding.
                # Zero any padded encoder positions defensively while keeping
                # the temporal [B,L,D] contract intact.
                representation = representation.masked_fill(
                    representation_padding.unsqueeze(-1), 0
                )
            outputs.append(representation.float().cpu())
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def encode_video(
        self,
        video_path: str,
        end_timestamps: Iterable[float],
        device: torch.device,
        ablation: str = "real",
        crossvideo_path: str = "",
    ) -> Dict[str, object]:
        timestamps = torch.as_tensor(list(end_timestamps), dtype=torch.float32)
        audio_source_path = str(video_path)
        if ablation == "crossvideo":
            if not crossvideo_path:
                raise ValueError(
                    "crossvideo ablation requires an explicit crossvideo_path"
                )
            audio_source_path = str(crossvideo_path)
        cache_path = self._embedding_cache_path(audio_source_path, timestamps)
        embeddings = self._load_embedding_cache(cache_path, timestamps)
        if embeddings is None:
            self._cache_misses = getattr(self, "_cache_misses", 0) + 1
            waveform = self.decode_audio(audio_source_path, self.sample_rate)
            windows = self._causal_windows(waveform, timestamps)
            embeddings = self.encode_windows(windows, device=device)
            self._save_embedding_cache(cache_path, timestamps, embeddings)

        if ablation == "zero":
            embeddings.zero_()
        elif ablation == "shuffled" and embeddings.shape[0] > 1:
            generator = torch.Generator().manual_seed(0)
            embeddings = embeddings[torch.randperm(embeddings.shape[0], generator=generator)]
        elif ablation == "stale" and embeddings.shape[0] > 1:
            embeddings = torch.cat([torch.zeros_like(embeddings[:1]), embeddings[:-1]], dim=0)
        elif ablation not in ("real", "crossvideo"):
            raise ValueError(f"Unsupported audio ablation: {ablation}")

        return {
            "embeddings": embeddings.cpu(),
            "end_timestamps": timestamps.cpu(),
            "source": self.source,
            "window_seconds": self.window_seconds,
            "sample_rate": self.sample_rate,
            "ablation": ablation,
            "audio_source_path": audio_source_path,
        }

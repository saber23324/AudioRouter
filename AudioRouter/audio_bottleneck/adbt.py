"""Q-Former-style audio-conditioned visual bottleneck.

The implementation follows the learnable-query/cross-attention pattern used by
LAVIS BLIP-2 Q-Former, while keeping the central ADBT attention explicit:

    K_v = V W_K
    P   = softmax(cosine(Q, K_v) / tau)

Phase 4 uses temporal BEATs tokens, slot-specific audio/time conditioning,
bounded Fourier time features, and fixed-temperature cosine attention. Its
output is constrained to:

    Z = P V_native

The legacy projected path remains only for controlled old-checkpoint
reproduction. Audio changes Q (and therefore P), but is never concatenated
into Z or the LLM input in either architecture.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def batched_effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return entropy effective rank for each matrix in a batch.

    The distribution is formed from singular values (not squared singular
    values), so ``exp(H(s / sum(s)))`` is bounded by the slot dimension.  A
    Gram eigendecomposition avoids materializing a costly full SVD for the
    wide native visual latents ``[slots, 3584]``.
    """
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.dim() != 3:
        raise ValueError(
            f"matrix must be [B,M,N] or [M,N], got {tuple(matrix.shape)}"
        )
    values = matrix.detach().float()
    gram = torch.matmul(values, values.transpose(-1, -2))
    singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
    totals = singular_values.sum(dim=-1, keepdim=True)
    probabilities = singular_values / totals.clamp_min(eps)
    entropy = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=-1)
    rank = entropy.exp()
    return torch.where(totals.squeeze(-1) > eps, rank, torch.zeros_like(rank))


class QueryTransformerBlock(nn.Module):
    """Small pre-norm Q-Former-style self-attention/FFN block."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.self_attention(normed, normed, normed, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class LegacyAudioQueryGenerator(nn.Module):
    """Phase-3 query generator retained for old-checkpoint reproduction."""

    def __init__(
        self,
        num_queries: int,
        hidden_size: int,
        num_heads: int,
        audio_dim: int = 768,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.audio_projection = nn.Linear(audio_dim, hidden_size)
        self.time_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.audio_cross_norm = nn.LayerNorm(hidden_size)
        self.audio_cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.query_block = QueryTransformerBlock(hidden_size, num_heads)

    def static_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        end_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        if audio_embeddings.dim() == 2:
            audio_embeddings = audio_embeddings.unsqueeze(1)
        if audio_embeddings.dim() != 3:
            raise ValueError(
                f"audio_embeddings must be [B,D] or [B,L,D], got {tuple(audio_embeddings.shape)}"
            )

        batch_size = audio_embeddings.shape[0]
        q = self.static_queries(batch_size)
        audio_tokens = self.audio_projection(audio_embeddings)

        # T_t is an explicit causal timestamp state: absolute time and the
        # current inter-sample delta.  Neither component contains future data.
        end_timestamps = end_timestamps.reshape(batch_size).to(q.dtype)
        delta = torch.cat(
            [torch.zeros_like(end_timestamps[:1]), end_timestamps[1:] - end_timestamps[:-1]]
        )
        time_state = torch.stack(
            [torch.log1p(end_timestamps.clamp_min(0)), torch.log1p(delta.clamp_min(0))],
            dim=-1,
        )
        q = q + self.time_projection(time_state).unsqueeze(1)

        normed_q = self.audio_cross_norm(q)
        audio_delta, _ = self.audio_cross_attention(
            normed_q, audio_tokens, audio_tokens, need_weights=False
        )
        return self.query_block(q + audio_delta)


class AudioQueryGenerator(nn.Module):
    """Phase-4 slot-specific queries over temporal BEATs representations.

    Each learnable slot independently cross-attends the complete causal BEATs
    token sequence. Bounded Fourier time features and the slot identity are
    then fused by a shared per-slot MLP. Audio remains query-side conditioning
    and is never returned to, or concatenated with, the LLM input.
    """

    def __init__(
        self,
        num_queries: int,
        hidden_size: int,
        num_heads: int,
        audio_dim: int = 768,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.audio_projection = nn.Linear(audio_dim, hidden_size)
        self.audio_cross_norm = nn.LayerNorm(hidden_size)
        self.audio_cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

        # Periods span the 2-second window scale through multi-hour videos.
        # sin/cos keeps absolute time and delta-time bounded in [-1, 1].
        self.register_buffer(
            "time_periods",
            torch.tensor([2.0, 10.0, 60.0, 300.0, 1800.0, 7200.0]),
            persistent=False,
        )
        time_feature_dim = int(self.time_periods.numel()) * 4
        self.time_projection = nn.Sequential(
            nn.Linear(time_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.slot_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def static_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_tokens.expand(batch_size, -1, -1)

    def bounded_time_features(self, end_timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = end_timestamps.reshape(-1).float().clamp_min(0)
        delta = torch.cat(
            [torch.zeros_like(timestamps[:1]), timestamps[1:] - timestamps[:-1]]
        ).clamp_min(0)
        periods = self.time_periods.to(device=timestamps.device)
        absolute_phase = 2 * math.pi * timestamps.unsqueeze(-1) / periods
        delta_phase = 2 * math.pi * delta.unsqueeze(-1) / periods
        return torch.cat(
            [
                absolute_phase.sin(),
                absolute_phase.cos(),
                delta_phase.sin(),
                delta_phase.cos(),
            ],
            dim=-1,
        )

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        end_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        if audio_embeddings.dim() == 2:
            audio_embeddings = audio_embeddings.unsqueeze(1)
        if audio_embeddings.dim() != 3:
            raise ValueError(
                f"audio_embeddings must be [B,L,D], got {tuple(audio_embeddings.shape)}"
            )

        batch_size = audio_embeddings.shape[0]
        q0 = self.static_queries(batch_size)
        audio_tokens = self.audio_projection(audio_embeddings)
        audio_context, _ = self.audio_cross_attention(
            self.audio_cross_norm(q0), audio_tokens, audio_tokens, need_weights=False
        )
        time_features = self.bounded_time_features(end_timestamps).to(q0.dtype)
        time_context = self.time_projection(time_features).unsqueeze(1).expand(
            -1, self.num_queries, -1
        )
        slot_delta = self.slot_mlp(
            torch.cat([q0, audio_context, time_context], dim=-1)
        )
        return q0 + slot_delta


class AudioConditionedBottleneck(nn.Module):
    """Stages 1--3 of the visual bottleneck roadmap.

    Stage 1 uses static learnable queries.  Stage 2 constructs and exposes
    audio-conditioned queries while retaining the static visual bottleneck.
    Stage 3 uses the audio-conditioned queries for visual cross-attention.
    """

    def __init__(
        self,
        visual_dim: int,
        num_queries: int = 32,
        hidden_size: int = 256,
        num_heads: int = 8,
        audio_dim: int = 768,
        stage: int = 3,
        latent_norm: str = "rmsnorm",
        value_mode: str = "projected",
        architecture: str = "phase4",
        attention_temperature: float = 0.2,
    ):
        super().__init__()
        if stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
        self.stage = stage
        self.visual_dim = visual_dim
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.latent_norm_type = latent_norm
        self.value_mode = value_mode
        self.architecture = architecture
        self.attention_temperature = float(attention_temperature)
        if architecture not in ("legacy", "phase4"):
            raise ValueError(
                f"architecture must be legacy or phase4, got {architecture}"
            )
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be > 0")
        if value_mode not in ("projected", "native"):
            raise ValueError(
                f"value_mode must be projected or native, got {value_mode}"
            )
        if value_mode == "native" and latent_norm != "none":
            raise ValueError(
                "native value mode implements M=P@V exactly and therefore "
                "requires latent_norm='none'"
            )

        query_generator_class = (
            AudioQueryGenerator if architecture == "phase4" else LegacyAudioQueryGenerator
        )
        self.query_generator = query_generator_class(
            num_queries=num_queries,
            hidden_size=hidden_size,
            num_heads=num_heads,
            audio_dim=audio_dim,
        )
        self.visual_key = nn.Linear(visual_dim, hidden_size, bias=False)
        if value_mode == "projected":
            self.visual_value = nn.Linear(visual_dim, hidden_size, bias=False)
            self.visual_output = nn.Linear(hidden_size, visual_dim, bias=False)
            self.visual_norm = nn.LayerNorm(hidden_size)
            self.visual_block = QueryTransformerBlock(hidden_size, num_heads)
            if latent_norm == "rmsnorm":
                self.latent_norm = nn.RMSNorm(visual_dim, eps=1e-6)
            elif latent_norm == "layernorm":
                self.latent_norm = nn.LayerNorm(visual_dim, eps=1e-6)
            elif latent_norm == "none":
                self.latent_norm = nn.Identity()
            else:
                raise ValueError(
                    f"latent_norm must be rmsnorm, layernorm, or none; got {latent_norm}"
                )
        else:
            # Do not instantiate unused learned transforms in the native path.
            # Its downstream representation is exactly M = P @ V_original.
            self.visual_value = None
            self.visual_output = None
            self.visual_norm = None
            self.visual_block = None
            self.latent_norm = None
        # Last-call diagnostics are detached and intentionally contain no raw
        # audio tensor.  They are useful for smoke tests and profiling.
        self.last_diagnostics: Dict[str, object] = {}

    def compute_attention_scores(
        self, queries: torch.Tensor, visual_keys: torch.Tensor
    ) -> torch.Tensor:
        if self.architecture == "phase4":
            # Normalize and score in FP32 even when frozen inference runs the
            # surrounding modules in FP16. This prevents norm amplification
            # and avoids prematurely rounding a usable distribution to one-hot.
            normalized_queries = F.normalize(queries.float(), dim=-1)
            normalized_keys = F.normalize(visual_keys.float(), dim=-1)
            return torch.matmul(
                normalized_queries, normalized_keys.transpose(-1, -2)
            ) / self.attention_temperature
        return torch.matmul(
            queries, visual_keys.transpose(-1, -2)
        ) / math.sqrt(self.hidden_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        end_timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if visual_tokens.dim() != 3:
            raise ValueError(
                f"visual_tokens must be [B,N,D], got {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual hidden size changed: expected {self.visual_dim}, got {visual_tokens.shape[-1]}"
            )

        batch_size = visual_tokens.shape[0]
        if end_timestamps is None:
            end_timestamps = torch.arange(
                1, batch_size + 1, device=visual_tokens.device, dtype=torch.float32
            )

        static_q = self.query_generator.static_queries(batch_size)
        audio_q = None
        if self.stage >= 2:
            if audio_embeddings is None:
                raise ValueError("Stage 2/3 requires aligned audio embeddings")
            audio_q = self.query_generator(audio_embeddings, end_timestamps)

        # Stage 2 validates audio extraction/query generation independently.
        # Stage 3 is the first stage where audio changes visual compression.
        q = audio_q if self.stage == 3 else static_q

        # Audio controls only the attention distribution. Values in both modes
        # originate exclusively from visual_tokens.
        k_v = self.visual_key(visual_tokens)
        scores = self.compute_attention_scores(q, k_v)
        attention_float = torch.softmax(scores.float(), dim=-1)
        attention = attention_float.to(visual_tokens.dtype)

        if self.value_mode == "native":
            # Phase-3 native-value constraint: M = P @ V_original. There is no
            # value projection, query/audio residual, FFN, output projection,
            # or normalization after this operation.
            output = torch.matmul(attention, visual_tokens)
        else:
            v_v = self.visual_value(visual_tokens)
            z_visual = torch.matmul(attention.to(v_v.dtype), v_v)

            # Legacy controlled baseline. Post-attention processing starts
            # from Z only; q/audio still has no downstream residual path.
            z_visual = self.visual_block(self.visual_norm(z_visual))
            output = self.visual_output(z_visual)
            output = self.latent_norm(output)

        query_norms = q.detach().float().norm(dim=-1)
        key_norms = k_v.detach().float().norm(dim=-1)
        top_scores = scores.detach().float().topk(k=2, dim=-1).values
        score_gap = top_scores[..., 0] - top_scores[..., 1]
        normalized_queries = F.normalize(q.detach().float(), dim=-1)
        query_cosine = torch.matmul(
            normalized_queries, normalized_queries.transpose(-1, -2)
        )
        off_diagonal = ~torch.eye(
            self.num_queries, dtype=torch.bool, device=query_cosine.device
        )
        query_cosine = query_cosine[:, off_diagonal]
        attention_entropy = -(
            attention_float * attention_float.clamp_min(1e-30).log()
        ).sum(dim=-1)
        unique_argmax = torch.tensor(
            [
                len(torch.unique(frame_attention.argmax(dim=-1)))
                for frame_attention in attention_float.detach()
            ],
            dtype=torch.float32,
            device=attention_float.device,
        )
        self.last_diagnostics = {
            "stage": self.stage,
            "architecture": self.architecture,
            "value_mode": self.value_mode,
            "attention_temperature": self.attention_temperature,
            "input_shape": tuple(visual_tokens.shape),
            "output_shape": tuple(output.shape),
            "audio_conditioned": self.stage == 3,
            "attention_row_sum_min": float(attention.sum(-1).min().detach().cpu()),
            "attention_row_sum_max": float(attention.sum(-1).max().detach().cpu()),
            "q_norm_mean": float(query_norms.mean().cpu()),
            "q_norm_max": float(query_norms.max().cpu()),
            "k_norm_mean": float(key_norms.mean().cpu()),
            "k_norm_max": float(key_norms.max().cpu()),
            "score_min": float(scores.detach().float().min().cpu()),
            "score_max": float(scores.detach().float().max().cpu()),
            "score_std": float(scores.detach().float().std(unbiased=False).cpu()),
            "top1_score_mean": float(top_scores[..., 0].mean().cpu()),
            "top2_score_mean": float(top_scores[..., 1].mean().cpu()),
            "top1_minus_top2_mean": float(score_gap.mean().cpu()),
            "top1_minus_top2_max": float(score_gap.max().cpu()),
            "attention_entropy_mean": float(attention_entropy.mean().detach().cpu()),
            "effective_support_mean": float(
                attention_entropy.exp().mean().detach().cpu()
            ),
            "query_cosine_mean": float(query_cosine.mean().cpu()),
            "query_cosine_min": float(query_cosine.min().cpu()),
            "query_cosine_max": float(query_cosine.max().cpu()),
            "unique_argmax_mean": float(unique_argmax.mean().cpu()),
            "unique_argmax_min": float(unique_argmax.min().cpu()),
            "latent_max_abs": float(output.detach().abs().max().cpu()),
            "latent_std": float(output.detach().float().std(unbiased=False).cpu()),
        }
        return output, attention

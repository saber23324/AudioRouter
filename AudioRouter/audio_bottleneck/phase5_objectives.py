"""Phase-5 supervision for learning useful audio-conditioned routing.

These losses never alter the ADBT value path.  Audio/visual alignment is
computed from pooled frozen BEATs tokens and pooled native visual tokens using
the projection layers already shared by ADBT.  Counterfactual supervision is
computed downstream by the trainer from real/wrong-audio QA option margins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AVAlignmentResult:
    loss: torch.Tensor
    audio_embeddings: torch.Tensor
    visual_embeddings: torch.Tensor
    metrics: Dict[str, float]


class AVFeatureQueue:
    """Small FIFO of raw cross-video features used only as negatives."""

    def __init__(self, capacity: int = 256):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = int(capacity)
        self.audio: Optional[torch.Tensor] = None
        self.visual: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self.audio is None else int(self.audio.shape[0])

    def enqueue(self, audio: torch.Tensor, visual: torch.Tensor) -> None:
        if self.capacity == 0:
            return
        audio = audio.detach().float().cpu()
        visual = visual.detach().float().cpu()
        if audio.shape[0] != visual.shape[0]:
            raise ValueError("audio/visual queue entries must have equal length")
        if self.audio is not None:
            audio = torch.cat([self.audio, audio], dim=0)
            visual = torch.cat([self.visual, visual], dim=0)
        self.audio = audio[-self.capacity :].contiguous()
        self.visual = visual[-self.capacity :].contiguous()

    def tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.audio is None:
            return None, None
        return (
            self.audio.to(device=device, dtype=dtype),
            self.visual.to(device=device, dtype=dtype),
        )


def gradient_activation_importance(
    visual_tokens: torch.Tensor,
    visual_gradients: torch.Tensor,
    topk: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a per-frame teacher routing distribution with Grad x Activation.

    Both inputs use the native visual-token layout ``[frames,tokens,dim]``.
    The returned tensor is detached and normalized over visual tokens.  Top-k
    filtering is applied independently per frame; a uniform distribution is
    used only for an all-zero teacher gradient.
    """
    if visual_tokens.shape != visual_gradients.shape or visual_tokens.dim() != 3:
        raise ValueError(
            "visual tokens/gradients must have equal [F,N,D] shapes, got "
            f"{tuple(visual_tokens.shape)} and {tuple(visual_gradients.shape)}"
        )
    scores = (visual_tokens.detach().float() * visual_gradients.detach().float()).norm(
        dim=-1
    )
    if topk < 0:
        raise ValueError("topk must be non-negative")
    if 0 < topk < scores.shape[-1]:
        indices = scores.topk(topk, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)
        scores = scores.masked_fill(~mask, 0.0)
    totals = scores.sum(dim=-1, keepdim=True)
    normalized = scores / totals.clamp_min(eps)
    uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
    return torch.where(totals > eps, normalized, uniform).detach()


def teacher_visual_feature_targets(
    visual_tokens: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    """Deterministically resample native visual tokens to a fixed slot count."""
    if visual_tokens.dim() != 3:
        raise ValueError(
            f"visual_tokens must be [F,N,D], got {tuple(visual_tokens.shape)}"
        )
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    # Adaptive pooling preserves chronological frame boundaries and ordered
    # spatial-token neighborhoods.  No learned visual Value path is added.
    return F.adaptive_avg_pool1d(
        visual_tokens.detach().float().transpose(1, 2), num_slots
    ).transpose(1, 2)


def teacher_routing_alignment_loss(
    teacher_importance: torch.Tensor,
    student_attention: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match mean student slot routing to teacher token importance."""
    if student_attention.dim() != 3:
        raise ValueError(
            f"student_attention must be [F,Q,N], got {tuple(student_attention.shape)}"
        )
    if teacher_importance.shape != (
        student_attention.shape[0],
        student_attention.shape[2],
    ):
        raise ValueError(
            "teacher importance must be [F,N] matching student attention, got "
            f"{tuple(teacher_importance.shape)} and {tuple(student_attention.shape)}"
        )
    teacher = teacher_importance.detach().float()
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)
    student = student_attention.float().mean(dim=1)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(eps)
    per_frame = (
        teacher
        * (teacher.clamp_min(eps).log() - student.clamp_min(eps).log())
    ).sum(dim=-1)
    teacher_peak = teacher.max(dim=-1).values.mean()
    student_on_teacher_top1 = student.gather(
        1, teacher.argmax(dim=-1, keepdim=True)
    ).mean()
    return per_frame.mean(), {
        "teacher_importance_peak": float(teacher_peak.detach().cpu()),
        "student_mass_on_teacher_top1": float(
            student_on_teacher_top1.detach().cpu()
        ),
    }


def visual_feature_distillation_loss(
    student_latents: torch.Tensor,
    teacher_targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cosine-distill native visual latents without a learned Value encoder."""
    if student_latents.shape != teacher_targets.shape or student_latents.dim() != 3:
        raise ValueError(
            "student/teacher features must have equal [F,Q,D] shapes, got "
            f"{tuple(student_latents.shape)} and {tuple(teacher_targets.shape)}"
        )
    cosine = F.cosine_similarity(
        student_latents.float(), teacher_targets.detach().float(), dim=-1
    )
    return (1.0 - cosine).mean(), {
        "feature_cosine": float(cosine.detach().mean().cpu())
    }


def audio_visual_routing_contrastive_loss(
    adbt,
    audio_tokens: torch.Tensor,
    student_latents: torch.Tensor,
    timestamps: torch.Tensor,
    temperature: float = 0.07,
    min_temporal_offset: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align audio with stop-gradient routed visual summaries.

    Gradients flow only through the audio embedding branch.  In particular,
    the visual summary and its shared projection are detached, preventing the
    A->P->Z path from moving its own contrastive target.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if audio_tokens.dim() != 3 or student_latents.dim() != 3:
        raise ValueError("audio tokens and student latents must both be rank 3")
    if audio_tokens.shape[0] != student_latents.shape[0]:
        raise ValueError("audio and routed visual frame counts must match")
    audio_embeddings = F.normalize(
        adbt.query_generator.audio_projection(audio_tokens.float()).mean(dim=1),
        dim=-1,
    )
    with torch.no_grad():
        visual_summaries = student_latents.detach().float().mean(dim=1)
        routed_visual_embeddings = F.normalize(
            adbt.visual_key(visual_summaries), dim=-1
        ).detach()
    similarities = audio_embeddings @ routed_visual_embeddings.transpose(0, 1)
    timestamps = timestamps.reshape(-1).to(similarities.device).float()
    near = (
        (timestamps[:, None] - timestamps[None, :]).abs()
        < float(min_temporal_offset)
    )
    near.fill_diagonal_(False)
    masked = similarities.masked_fill(near, float("-inf"))
    targets = torch.arange(masked.shape[0], device=masked.device)
    loss = F.cross_entropy(masked / temperature, targets)
    metrics = _retrieval_metrics(
        similarities, timestamps, min_temporal_offset
    )
    metrics["positive_similarity"] = float(
        similarities.diagonal().detach().mean().cpu()
    )
    return loss, metrics


def pool_native_av(
    audio_tokens: torch.Tensor, visual_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool temporal BEATs and native visual tokens without using ADBT Z."""
    if audio_tokens.dim() != 3:
        raise ValueError(
            f"audio_tokens must be [B,L,Da], got {tuple(audio_tokens.shape)}"
        )
    if visual_tokens.dim() != 3:
        raise ValueError(
            f"visual_tokens must be [B,N,Dv], got {tuple(visual_tokens.shape)}"
        )
    if audio_tokens.shape[0] != visual_tokens.shape[0]:
        raise ValueError("audio and visual frame batches must be aligned")
    return audio_tokens.mean(dim=1), visual_tokens.detach().mean(dim=1)


def _retrieval_metrics(
    similarities: torch.Tensor,
    timestamps: torch.Tensor,
    min_temporal_offset: float,
) -> Dict[str, float]:
    batch = similarities.shape[0]
    targets = torch.arange(batch, device=similarities.device)
    ranking = similarities.argsort(dim=-1, descending=True)
    recall1 = (ranking[:, 0] == targets).float().mean()
    k = min(5, similarities.shape[1])
    recall5 = (ranking[:, :k] == targets[:, None]).any(dim=-1).float().mean()

    timestamps = timestamps.reshape(-1).to(similarities.device).float()
    hard_mask = (
        (timestamps[:, None] - timestamps[None, :]).abs()
        >= float(min_temporal_offset)
    )
    hard_mask.fill_diagonal_(False)
    valid = hard_mask.any(dim=-1)
    if valid.any():
        hard_max = similarities.masked_fill(~hard_mask, float("-inf")).max(dim=-1).values
        temporal_accuracy = (
            similarities.diagonal()[valid] > hard_max[valid]
        ).float().mean()
    else:
        temporal_accuracy = similarities.new_tensor(float("nan"))
    return {
        "recall_at_1": float(recall1.detach().cpu()),
        "recall_at_5": float(recall5.detach().cpu()),
        "temporal_hard_accuracy": float(temporal_accuracy.detach().cpu()),
    }


def temporal_av_contrastive_loss(
    adbt,
    audio_tokens: torch.Tensor,
    visual_tokens: torch.Tensor,
    timestamps: torch.Tensor,
    temperature: float = 0.07,
    min_temporal_offset: float = 5.0,
    queue: Optional[AVFeatureQueue] = None,
) -> AVAlignmentResult:
    """Symmetric InfoNCE with temporal and cross-video negatives.

    The positive for row ``i`` is the aligned ``(A_i, V_i)`` pair. Same-video
    candidates closer than ``min_temporal_offset`` are masked from the
    denominator to avoid treating near-synchronous frames as false negatives.
    Raw entries in ``queue`` come only from earlier videos and are projected
    with the current weights, so they act as cross-video negatives.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw_audio, raw_visual = pool_native_av(audio_tokens, visual_tokens)
    audio_embeddings = F.normalize(
        adbt.query_generator.audio_projection(raw_audio.float()), dim=-1
    )
    visual_embeddings = F.normalize(
        adbt.visual_key(raw_visual.float()), dim=-1
    )
    batch = audio_embeddings.shape[0]
    queued_audio, queued_visual = (
        queue.tensors(audio_embeddings.device, raw_audio.dtype)
        if queue is not None
        else (None, None)
    )
    if queued_audio is not None:
        queued_audio_embeddings = F.normalize(
            adbt.query_generator.audio_projection(queued_audio.float()), dim=-1
        )
        queued_visual_embeddings = F.normalize(
            adbt.visual_key(queued_visual.float()), dim=-1
        )
        audio_candidates = torch.cat(
            [audio_embeddings, queued_audio_embeddings], dim=0
        )
        visual_candidates = torch.cat(
            [visual_embeddings, queued_visual_embeddings], dim=0
        )
    else:
        audio_candidates = audio_embeddings
        visual_candidates = visual_embeddings

    logits_a_to_v = audio_embeddings @ visual_candidates.transpose(0, 1)
    logits_v_to_a = visual_embeddings @ audio_candidates.transpose(0, 1)
    timestamps = timestamps.reshape(-1).to(logits_a_to_v.device).float()
    near = (
        (timestamps[:, None] - timestamps[None, :]).abs()
        < float(min_temporal_offset)
    )
    near.fill_diagonal_(False)
    logits_a_to_v[:, :batch] = logits_a_to_v[:, :batch].masked_fill(
        near, float("-inf")
    )
    logits_v_to_a[:, :batch] = logits_v_to_a[:, :batch].masked_fill(
        near, float("-inf")
    )
    targets = torch.arange(batch, device=logits_a_to_v.device)
    loss_a_to_v = F.cross_entropy(logits_a_to_v / temperature, targets)
    loss_v_to_a = F.cross_entropy(logits_v_to_a / temperature, targets)
    loss = 0.5 * (loss_a_to_v + loss_v_to_a)
    metrics = _retrieval_metrics(
        audio_embeddings @ visual_embeddings.transpose(0, 1),
        timestamps,
        min_temporal_offset,
    )
    metrics.update(
        {
            "loss_a_to_v": float(loss_a_to_v.detach().cpu()),
            "loss_v_to_a": float(loss_v_to_a.detach().cpu()),
            "queue_size": float(0 if queued_audio is None else queued_audio.shape[0]),
        }
    )
    return AVAlignmentResult(
        loss=loss,
        audio_embeddings=audio_embeddings,
        visual_embeddings=visual_embeddings,
        metrics=metrics,
    )


def counterfactual_qa_margin_loss(
    real_option_logits: torch.Tensor,
    wrong_option_logits: torch.Tensor,
    answer_indices: torch.Tensor,
    margin: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Require aligned audio to yield a larger correct-option QA margin."""
    if real_option_logits.shape != wrong_option_logits.shape:
        raise ValueError("real and wrong option logits must have the same shape")
    if real_option_logits.dim() != 2:
        raise ValueError("option logits must be [B,num_options]")
    answer_indices = answer_indices.reshape(-1).to(real_option_logits.device)
    if answer_indices.shape[0] != real_option_logits.shape[0]:
        raise ValueError("one answer index is required per batch row")
    correct_real = real_option_logits.gather(1, answer_indices[:, None]).squeeze(1)
    correct_wrong = wrong_option_logits.gather(1, answer_indices[:, None]).squeeze(1)
    one_hot = F.one_hot(
        answer_indices, num_classes=real_option_logits.shape[1]
    ).bool()
    best_other_real = real_option_logits.masked_fill(one_hot, float("-inf")).max(1).values
    best_other_wrong = wrong_option_logits.masked_fill(one_hot, float("-inf")).max(1).values
    real_margin = correct_real - best_other_real
    wrong_margin = correct_wrong - best_other_wrong
    violations = F.relu(float(margin) - real_margin + wrong_margin)
    return violations.mean(), {
        "real_margin": float(real_margin.detach().mean().cpu()),
        "wrong_margin": float(wrong_margin.detach().mean().cpu()),
        "violation_rate": float((violations > 0).float().mean().detach().cpu()),
    }


def counterfactual_teacher_kd_loss(
    teacher_option_logits: torch.Tensor,
    real_option_logits: torch.Tensor,
    wrong_option_logits: torch.Tensor,
    margin: float = 0.2,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Rank real/wrong audio by KL distance to a full-visual teacher."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if teacher_option_logits.shape != real_option_logits.shape:
        raise ValueError("teacher and real option logits must have equal shape")
    if real_option_logits.shape != wrong_option_logits.shape:
        raise ValueError("real and wrong option logits must have equal shape")
    teacher_log_probs = F.log_softmax(
        teacher_option_logits.detach().float() / temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    real_log_probs = F.log_softmax(real_option_logits.float() / temperature, dim=-1)
    wrong_log_probs = F.log_softmax(wrong_option_logits.float() / temperature, dim=-1)
    d_real = (teacher_probs * (teacher_log_probs - real_log_probs)).sum(dim=-1)
    d_wrong = (teacher_probs * (teacher_log_probs - wrong_log_probs)).sum(dim=-1)
    violations = F.relu(float(margin) + d_real - d_wrong)
    return violations.mean(), {
        "teacher_kl_real": float(d_real.detach().mean().cpu()),
        "teacher_kl_wrong": float(d_wrong.detach().mean().cpu()),
        "teacher_kl_advantage": float((d_wrong - d_real).detach().mean().cpu()),
        "violation_rate": float((violations > 0).float().mean().detach().cpu()),
    }

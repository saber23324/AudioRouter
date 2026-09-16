"""Audio-conditioned visual bottleneck modules.

Audio is used only to construct visual attention queries.  No audio tensor from
this package is passed to the language model.
"""

from .adbt import AudioConditionedBottleneck, AudioQueryGenerator, batched_effective_rank
from .beats_encoder import BEATsAudioEncoder
from .phase5_objectives import (
    AVFeatureQueue,
    audio_visual_routing_contrastive_loss,
    counterfactual_qa_margin_loss,
    counterfactual_teacher_kd_loss,
    gradient_activation_importance,
    teacher_routing_alignment_loss,
    teacher_visual_feature_targets,
    temporal_av_contrastive_loss,
    visual_feature_distillation_loss,
)

__all__ = [
    "AudioConditionedBottleneck",
    "AudioQueryGenerator",
    "BEATsAudioEncoder",
    "batched_effective_rank",
    "AVFeatureQueue",
    "audio_visual_routing_contrastive_loss",
    "counterfactual_qa_margin_loss",
    "counterfactual_teacher_kd_loss",
    "gradient_activation_importance",
    "teacher_routing_alignment_loss",
    "teacher_visual_feature_targets",
    "temporal_av_contrastive_loss",
    "visual_feature_distillation_loss",
]

import unittest

import torch
import torch.nn as nn

from AudioRouter.audio_bottleneck.phase5_objectives import (
    AVFeatureQueue,
    audio_visual_routing_contrastive_loss,
    counterfactual_qa_margin_loss,
    counterfactual_teacher_kd_loss,
    gradient_activation_importance,
    pool_native_av,
    teacher_routing_alignment_loss,
    teacher_visual_feature_targets,
    temporal_av_contrastive_loss,
    visual_feature_distillation_loss,
)


class _Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_projection = nn.Linear(3, 2, bias=False)


class _ADBT(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = _Generator()
        self.visual_key = nn.Linear(4, 2, bias=False)


class Phase5ObjectivesTest(unittest.TestCase):
    def test_gradient_activation_importance_is_topk_distribution(self):
        visual = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        gradients = torch.ones_like(visual)
        importance = gradient_activation_importance(visual, gradients, topk=2)
        self.assertEqual(tuple(importance.shape), (1, 3))
        self.assertEqual(int((importance > 0).sum()), 2)
        torch.testing.assert_close(importance.sum(-1), torch.ones(1))
        self.assertEqual(float(importance[0, 0]), 0.0)

    def test_routing_alignment_prefers_teacher_matched_attention(self):
        teacher = torch.tensor([[0.8, 0.1, 0.1]])
        matched = torch.tensor([[[0.7, 0.2, 0.1]]]).repeat(1, 4, 1)
        matched.requires_grad_()
        wrong = torch.tensor([[[0.1, 0.1, 0.8]]]).repeat(1, 4, 1)
        matched_loss, _ = teacher_routing_alignment_loss(teacher, matched)
        wrong_loss, _ = teacher_routing_alignment_loss(teacher, wrong)
        self.assertLess(float(matched_loss), float(wrong_loss))
        matched_loss.backward()
        self.assertGreater(float(matched.grad.abs().sum()), 0.0)

    def test_visual_feature_target_and_distillation_shapes(self):
        visual = torch.arange(2 * 6 * 4.0).reshape(2, 6, 4)
        target = teacher_visual_feature_targets(visual, num_slots=3)
        self.assertEqual(tuple(target.shape), (2, 3, 4))
        loss, metrics = visual_feature_distillation_loss(target.clone(), target)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(metrics["feature_cosine"], 1.0, places=6)

    def test_av_routing_stops_visual_branch_and_trains_audio_projection(self):
        torch.manual_seed(12)
        adbt = _ADBT()
        audio = torch.randn(4, 3, 3)
        latents = torch.randn(4, 5, 4, requires_grad=True)
        timestamps = torch.tensor([0.0, 6.0, 12.0, 18.0])
        loss, metrics = audio_visual_routing_contrastive_loss(
            adbt, audio, latents, timestamps
        )
        loss.backward()
        self.assertGreater(
            float(adbt.query_generator.audio_projection.weight.grad.norm()), 0.0
        )
        self.assertIsNone(adbt.visual_key.weight.grad)
        self.assertIsNone(latents.grad)
        self.assertIn("recall_at_1", metrics)

    def test_pool_uses_audio_tokens_and_native_visual_tokens(self):
        audio = torch.arange(24.0).reshape(2, 4, 3)
        visual = torch.arange(40.0).reshape(2, 5, 4).requires_grad_()
        pooled_audio, pooled_visual = pool_native_av(audio, visual)
        self.assertTrue(torch.equal(pooled_audio, audio.mean(1)))
        self.assertTrue(torch.equal(pooled_visual, visual.detach().mean(1)))
        self.assertFalse(pooled_visual.requires_grad)

    def test_av_loss_has_gradients_for_both_shared_projections(self):
        torch.manual_seed(0)
        adbt = _ADBT()
        audio = torch.randn(4, 3, 3)
        visual = torch.randn(4, 5, 4)
        timestamps = torch.tensor([0.0, 6.0, 12.0, 18.0])
        result = temporal_av_contrastive_loss(
            adbt, audio, visual, timestamps, queue=AVFeatureQueue(8)
        )
        result.loss.backward()
        self.assertGreater(float(adbt.query_generator.audio_projection.weight.grad.norm()), 0)
        self.assertGreater(float(adbt.visual_key.weight.grad.norm()), 0)

    def test_counterfactual_margin_prefers_aligned_audio(self):
        real = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
        wrong = torch.tensor([[2.0, 1.5, 0.0, -1.0]])
        loss, metrics = counterfactual_qa_margin_loss(
            real, wrong, torch.tensor([0]), margin=0.2
        )
        self.assertEqual(float(loss), 0.0)
        self.assertGreater(metrics["real_margin"], metrics["wrong_margin"])

    def test_counterfactual_margin_penalizes_wrong_audio_advantage(self):
        real = torch.tensor([[2.0, 1.5, 0.0, -1.0]], requires_grad=True)
        wrong = torch.tensor([[4.0, 1.0, 0.0, -1.0]], requires_grad=True)
        loss, metrics = counterfactual_qa_margin_loss(
            real, wrong, torch.tensor([0]), margin=0.2
        )
        self.assertGreater(float(loss), 0)
        self.assertEqual(metrics["violation_rate"], 1.0)
        loss.backward()
        self.assertIsNotNone(real.grad)
        self.assertIsNotNone(wrong.grad)

    def test_teacher_kd_prefers_real_branch_near_teacher(self):
        teacher = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
        real = torch.tensor([[3.9, 1.1, 0.0, -1.0]], requires_grad=True)
        wrong = torch.tensor([[0.0, 3.0, 1.0, -1.0]], requires_grad=True)
        loss, metrics = counterfactual_teacher_kd_loss(
            teacher, real, wrong, margin=0.2
        )
        self.assertEqual(float(loss), 0.0)
        self.assertGreater(metrics["teacher_kl_advantage"], 0.2)

    def test_teacher_kd_penalizes_wrong_branch_near_teacher(self):
        teacher = torch.tensor([[4.0, 1.0, 0.0, -1.0]])
        real = torch.tensor([[0.0, 3.0, 1.0, -1.0]], requires_grad=True)
        wrong = torch.tensor([[3.9, 1.1, 0.0, -1.0]], requires_grad=True)
        loss, metrics = counterfactual_teacher_kd_loss(
            teacher, real, wrong, margin=0.2
        )
        self.assertGreater(float(loss), 0.2)
        self.assertLess(metrics["teacher_kl_advantage"], 0.0)
        loss.backward()
        self.assertGreater(float(real.grad.abs().sum()), 0)
        self.assertGreater(float(wrong.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

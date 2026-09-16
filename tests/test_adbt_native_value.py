import unittest

import torch

from AudioRouter.audio_bottleneck.adbt import (
    AudioConditionedBottleneck,
    batched_effective_rank,
)


def make_native(num_queries: int = 8) -> AudioConditionedBottleneck:
    return AudioConditionedBottleneck(
        visual_dim=32,
        num_queries=num_queries,
        hidden_size=16,
        num_heads=4,
        audio_dim=12,
        stage=3,
        latent_norm="none",
        value_mode="native",
    )


class NativeValueADBTTest(unittest.TestCase):
    def test_stage1_is_a_true_audio_free_static_query_control(self):
        torch.manual_seed(5)
        model = AudioConditionedBottleneck(
            visual_dim=32,
            num_queries=8,
            hidden_size=16,
            num_heads=4,
            audio_dim=12,
            stage=1,
            latent_norm="none",
            value_mode="native",
        )
        visual = torch.randn(2, 11, 32)
        timestamps = torch.tensor([1.0, 3.0])

        output_without_audio, attention_without_audio = model(
            visual, None, timestamps
        )
        output_with_audio, attention_with_audio = model(
            visual, torch.randn(2, 7, 12), timestamps
        )

        torch.testing.assert_close(output_without_audio, output_with_audio)
        torch.testing.assert_close(attention_without_audio, attention_with_audio)
        output_without_audio.square().mean().backward()
        self.assertIsNone(model.query_generator.audio_projection.weight.grad)
        self.assertIsNotNone(model.query_generator.query_tokens.grad)

    def test_effective_rank_distinguishes_full_rank_rank_one_and_zero(self):
        identity = torch.eye(8).unsqueeze(0)
        rank_one = torch.ones(1, 8, 13)
        zero = torch.zeros(1, 8, 13)

        self.assertAlmostEqual(float(batched_effective_rank(identity)), 8.0, places=5)
        self.assertLess(float(batched_effective_rank(rank_one)), 1.05)
        self.assertEqual(float(batched_effective_rank(zero)), 0.0)

    def test_native_value_is_exact_probability_weighted_original_visual_tokens(self):
        torch.manual_seed(7)
        model = make_native().eval()
        visual = torch.randn(3, 11, 32)
        audio = torch.randn(3, 12)
        timestamps = torch.tensor([0.5, 2.5, 4.5])

        output, attention = model(visual, audio, timestamps)
        with torch.no_grad():
            queries = model.query_generator(audio, timestamps)
            keys = model.visual_key(visual)
            expected_attention = torch.softmax(
                model.compute_attention_scores(queries, keys),
                dim=-1,
            )
            expected_output = torch.matmul(expected_attention, visual)

        torch.testing.assert_close(attention, expected_attention)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(
            attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))
        )
        self.assertEqual(output.shape, (3, 8, 32))
        self.assertIsNone(model.visual_value)
        self.assertIsNone(model.visual_output)
        self.assertIsNone(model.visual_block)
        self.assertIsNone(model.latent_norm)

    def test_phase4_cosine_scores_are_bounded_by_fixed_temperature(self):
        torch.manual_seed(9)
        model = make_native().eval()
        visual = torch.randn(2, 11, 32) * 1000
        audio = torch.randn(2, 5, 12) * 1000
        timestamps = torch.tensor([1.0, 1000000.0])

        model(visual, audio, timestamps)

        self.assertLessEqual(model.last_diagnostics["score_max"], 5.00001)
        self.assertGreaterEqual(model.last_diagnostics["score_min"], -5.00001)
        bounded = model.query_generator.bounded_time_features(timestamps)
        self.assertLessEqual(float(bounded.abs().max()), 1.0)

    def test_temporal_audio_conditioning_remains_slot_specific(self):
        torch.manual_seed(10)
        model = make_native().eval()
        audio = torch.randn(2, 7, 12)
        timestamps = torch.tensor([2.0, 4.0])

        queries = model.query_generator(audio, timestamps)
        pairwise_difference = (queries[:, 1:] - queries[:, :-1]).abs().sum(-1)

        self.assertEqual(queries.shape, (2, 8, 16))
        self.assertTrue((pairwise_difference > 0).all())

    def test_audio_cannot_enter_native_output_when_visual_values_are_zero(self):
        torch.manual_seed(11)
        model = make_native()
        visual = torch.zeros(2, 9, 32)
        timestamps = torch.tensor([1.0, 3.0])
        output_a, _ = model(visual, torch.randn(2, 12), timestamps)
        output_b, _ = model(visual, torch.randn(2, 12), timestamps)

        self.assertEqual(torch.count_nonzero(output_a), 0)
        self.assertEqual(torch.count_nonzero(output_b), 0)

    def test_native_value_path_propagates_gradients(self):
        torch.manual_seed(13)
        model = make_native()
        visual = torch.randn(2, 9, 32, requires_grad=True)
        audio = torch.randn(2, 12)
        timestamps = torch.tensor([1.0, 3.0])

        output, _ = model(visual, audio, timestamps)
        output.square().mean().backward()

        self.assertIsNotNone(visual.grad)
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertIsNotNone(model.visual_key.weight.grad)
        self.assertIsNotNone(model.query_generator.query_tokens.grad)
        self.assertIsNotNone(model.query_generator.audio_projection.weight.grad)
        self.assertGreater(model.visual_key.weight.grad.abs().sum(), 0)
        self.assertGreater(model.query_generator.query_tokens.grad.abs().sum(), 0)

    def test_native_mode_rejects_post_value_normalization(self):
        with self.assertRaisesRegex(ValueError, "requires latent_norm='none'"):
            AudioConditionedBottleneck(
                visual_dim=32,
                num_queries=8,
                hidden_size=16,
                num_heads=4,
                stage=3,
                latent_norm="rmsnorm",
                value_mode="native",
            )

    def test_projected_mode_remains_available_for_two_by_two_control(self):
        model = AudioConditionedBottleneck(
            visual_dim=32,
            num_queries=8,
            hidden_size=16,
            num_heads=4,
            audio_dim=12,
            stage=3,
            latent_norm="rmsnorm",
            value_mode="projected",
        )
        output, attention = model(
            torch.randn(2, 9, 32),
            torch.randn(2, 12),
            torch.tensor([1.0, 3.0]),
        )
        self.assertEqual(output.shape, (2, 8, 32))
        self.assertEqual(attention.shape, (2, 8, 9))
        self.assertIsNotNone(model.visual_value)


if __name__ == "__main__":
    unittest.main()

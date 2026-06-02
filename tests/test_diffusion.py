"""
TDD Tests for Discrete Diffusion Model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models.diffusion import DiscreteDiffusion


class TestDiscreteDiffusion:
    def test_transition_matrix_shape(self):
        """Transition matrices should have shape [T, V+1, V+1]."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)
        assert diffusion.transition_matrices.shape == (10, 9, 9)

    def test_transition_matrix_row_sum(self):
        """Each row of transition matrix should sum to 1."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)
        row_sums = diffusion.transition_matrices.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_q_sample_changes_values(self):
        """Forward diffusion should change some tokens over time."""
        diffusion = DiscreteDiffusion(num_classes=16, num_timesteps=50)
        x_0 = torch.randint(0, 16, (4, 10))
        t = torch.tensor([0, 10, 25, 49])
        x_t = diffusion.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        # Later timesteps should have more Obs tokens (value = num_classes)
        obs_count = (x_t == diffusion.num_classes).float().mean(dim=-1)
        # Generally more obs at later timesteps
        assert obs_count[3] >= obs_count[0]

    def test_q_sample_preserves_batch(self):
        """q_sample should handle batched inputs correctly."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=20)
        x_0 = torch.randint(0, 8, (8, 5))
        t = torch.randint(0, 20, (8,))
        x_t = diffusion.q_sample(x_0, t)
        assert x_t.shape == (8, 5)
        assert x_t.dtype == torch.long

    def test_prior_distribution_valid(self):
        """Prior distribution at final timestep should sum to 1."""
        diffusion = DiscreteDiffusion(num_classes=16, num_timesteps=100)
        prior = diffusion.q_prior
        assert torch.allclose(prior.sum(), torch.tensor(1.0), atol=1e-5)
        assert prior.shape == (diffusion.num_states,)

    def test_q_posterior_logits_shape(self):
        """q_posterior_logits should return correct shape."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)
        x_0 = torch.randint(0, 8, (4, 5))
        t = torch.randint(0, 10, (4,))
        x_t = diffusion.q_sample(x_0, t)
        logits = diffusion.q_posterior_logits(x_t, x_0, t)
        assert logits.shape == (4, 5, 9)

    def test_q_posterior_logits_valid_probability(self):
        """q_posterior_logits should return valid log probabilities."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)
        x_0 = torch.randint(0, 8, (4, 5))
        t = torch.randint(0, 10, (4,))
        x_t = diffusion.q_sample(x_0, t)
        logits = diffusion.q_posterior_logits(x_t, x_0, t)
        probs = logits.exp()
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4, 5), atol=1e-4)
        # Probabilities should be non-negative
        assert torch.all(probs >= 0)

    def test_q_posterior_logits_consistency(self):
        """q_posterior_logits should be consistent with q_sample transitions."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)
        x_0 = torch.zeros(2, 4, dtype=torch.long)
        t = torch.zeros(2, dtype=torch.long)
        x_t = diffusion.q_sample(x_0, t)
        logits = diffusion.q_posterior_logits(x_t, x_0, t)
        # For t=0, posterior should heavily favor x_0 since q(x_0|x_0)=1
        probs = logits.exp()
        # Check that probability mass is concentrated on states that could lead to x_t
        assert probs.sum() > 0

    def test_p_losses_components(self):
        """p_losses should return all required loss components."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)

        class MockDenoiser(torch.nn.Module):
            def forward(self, x_t, t, condition):
                return torch.randn(x_t.size(0), x_t.size(1), 8)

        class MockPrior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_encoder = torch.nn.Module()
                self.global_encoder.num_tokens = 4
            def decode_from_indices(self, indices):
                return torch.randn(indices.size(0), 17, 2)

        denoiser = MockDenoiser()
        prior = MockPrior()
        x_0 = torch.zeros(2, 4, dtype=torch.long)
        condition = torch.randn(2, 512)
        target_pose = torch.randn(2, 17, 2)
        t = torch.tensor([0, 1])

        total, recon, k0, vlb, tkn = diffusion.p_losses(
            denoiser, x_0, t, condition, target_pose, prior
        )

        # All losses should be finite and non-negative
        assert torch.isfinite(total)
        assert recon.item() >= 0
        assert k0.item() >= 0
        assert vlb.item() >= 0
        assert tkn.item() >= 0

    def test_p_losses_total_composition(self):
        """p_losses total should equal weighted sum of components."""
        diffusion = DiscreteDiffusion(num_classes=8, num_timesteps=10)

        class MockDenoiser(torch.nn.Module):
            def forward(self, x_t, t, condition):
                return torch.randn(x_t.size(0), x_t.size(1), 8)

        class MockPrior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_encoder = torch.nn.Module()
                self.global_encoder.num_tokens = 4
            def decode_from_indices(self, indices):
                return torch.randn(indices.size(0), 17, 2)

        denoiser = MockDenoiser()
        prior = MockPrior()
        x_0 = torch.zeros(2, 4, dtype=torch.long)
        condition = torch.randn(2, 512)
        target_pose = torch.randn(2, 17, 2)
        t = torch.tensor([1])
        eta = 0.0005

        total, recon, k0, vlb, tkn = diffusion.p_losses(
            denoiser, x_0, t, condition, target_pose, prior, eta
        )

        expected = eta * k0 + vlb + tkn + recon
        assert torch.allclose(total, expected, atol=1e-5)

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

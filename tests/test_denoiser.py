"""
TDD Tests for Pose Denoiser (Transformer with AdaLN).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models.denoiser import SinusoidalPosEmb, AdaLN, PoseDenoiser


class TestSinusoidalPosEmb:
    def test_output_shape(self):
        emb = SinusoidalPosEmb(dim=128)
        t = torch.tensor([0, 5, 10])
        out = emb(t)
        assert out.shape == (3, 128)

    def test_different_timesteps_produce_different_embeddings(self):
        emb = SinusoidalPosEmb(dim=64)
        t1 = torch.tensor([0])
        t2 = torch.tensor([50])
        assert not torch.allclose(emb(t1), emb(t2))


class TestAdaLN:
    def test_output_shape(self):
        adaln = AdaLN(hidden_dim=256, cond_dim=128)
        x = torch.randn(2, 10, 256)
        t_emb = torch.randn(2, 128)
        out = adaln(x, t_emb)
        assert out.shape == (2, 10, 256)

    def test_scale_shift_effect(self):
        """AdaLN should apply different scale/shift for different conditions."""
        adaln = AdaLN(hidden_dim=64, cond_dim=32)
        x = torch.ones(2, 5, 64)
        t1 = torch.randn(2, 32)
        t2 = torch.randn(2, 32)
        out1 = adaln(x, t1)
        out2 = adaln(x, t2)
        assert not torch.allclose(out1, out2)


class TestPoseDenoiser:
    def test_forward_shape(self):
        denoiser = PoseDenoiser(
            num_tokens=34, hidden_dim=128, num_classes=64,
            num_blocks=2, num_heads=4, cond_dim=64
        )
        x_t = torch.randint(0, 65, (4, 34))  # [B, N]
        t = torch.randint(0, 100, (4,))
        condition = torch.randn(4, 64)
        logits = denoiser(x_t, t, condition)
        assert logits.shape == (4, 34, 64)

    def test_gradient_flow(self):
        denoiser = PoseDenoiser(
            num_tokens=10, hidden_dim=64, num_classes=8,
            num_blocks=2, num_heads=2, cond_dim=32
        )
        x_t = torch.randint(0, 9, (2, 10))
        t = torch.randint(0, 100, (2,))
        condition = torch.randn(2, 32, requires_grad=True)
        logits = denoiser(x_t, t, condition)
        loss = logits.sum()
        loss.backward()
        assert condition.grad is not None
        assert condition.grad.abs().sum() > 0

    def test_inference_mode(self):
        denoiser = PoseDenoiser(
            num_tokens=34, hidden_dim=128, num_classes=64,
            num_blocks=2, num_heads=4, cond_dim=64
        )
        denoiser.eval()
        x_t = torch.randint(0, 65, (1, 34))
        t = torch.tensor([50])
        condition = torch.randn(1, 64)
        with torch.no_grad():
            logits = denoiser(x_t, t, condition)
        assert logits.shape == (1, 34, 64)

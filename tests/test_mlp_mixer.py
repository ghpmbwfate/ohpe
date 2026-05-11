"""
TDD Tests for MLP-Mixer Block
RED phase: tests should fail before implementation exists.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models.mlp_mixer import MixerBlock


class TestMixerBlock:
    def test_output_shape_same_as_input(self):
        """MixerBlock should preserve [B, N, D] shape."""
        batch, num_patches, dim = 4, 17, 512
        x = torch.randn(batch, num_patches, dim)
        block = MixerBlock(dim=dim, num_patches=num_patches)
        out = block(x)
        assert out.shape == (batch, num_patches, dim)

    def test_parameters_exist(self):
        """MixerBlock should have trainable parameters."""
        block = MixerBlock(dim=256, num_patches=10)
        params = list(block.parameters())
        assert len(params) > 0
        assert any(p.requires_grad for p in params)

    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        block = MixerBlock(dim=128, num_patches=8)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_different_patches(self):
        """Should work with varying num_patches."""
        for num_patches in [1, 10, 34, 100]:
            block = MixerBlock(dim=64, num_patches=num_patches)
            x = torch.randn(2, num_patches, 64)
            out = block(x)
            assert out.shape == (2, num_patches, 64)

    def test_inference_mode(self):
        """Should work in eval mode without error."""
        block = MixerBlock(dim=256, num_patches=17)
        block.eval()
        x = torch.randn(1, 17, 256)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (1, 17, 256)

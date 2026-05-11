"""
TDD Tests for Multimodal Condition Encoder.
Note: These tests download pretrained models on first run.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest


class TestMultimodalConditionEncoder:
    @pytest.fixture(scope='class')
    def encoder(self):
        from models.multimodal_encoder import MultimodalConditionEncoder
        # Use small cond_dim for faster testing
        enc = MultimodalConditionEncoder(cond_dim=64, cache_dir='./models_pretrained')
        return enc

    def test_output_shape(self, encoder):
        """Should output [B, cond_dim]."""
        images = torch.randn(2, 3, 224, 224)
        texts = ["a person standing", "a person running"]
        out = encoder(images, texts)
        assert out.shape == (2, 64)

    def test_backbones_frozen(self, encoder):
        """Swin and CLIP backbone parameters should be frozen."""
        for name, param in encoder.named_parameters():
            # Only actual backbone params, not projection layers
            if name.startswith('swin.') or name.startswith('clip_model.'):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_projection_trainable(self, encoder):
        """Projection layers should be trainable."""
        trainable = []
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        assert len(trainable) > 0
        # Should contain proj layers
        assert any('proj' in n for n in trainable)

    def test_gradient_flow_to_projections(self, encoder):
        """Gradients should flow through projection layers."""
        images = torch.randn(1, 3, 224, 224)
        texts = ["a person standing"]
        out = encoder(images, texts)
        loss = out.sum()
        loss.backward()
        # Check that some proj param has grad
        has_grad = False
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_different_inputs_produce_different_outputs(self, encoder):
        """Different images should produce different conditions."""
        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)
        texts = ["a person standing"]
        out1 = encoder(img1, texts)
        out2 = encoder(img2, texts)
        assert not torch.allclose(out1, out2)

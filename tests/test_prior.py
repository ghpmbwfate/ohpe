"""
TDD Tests for Hierarchical Pose Prior (VQ-VAE).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models.prior import PartEncoder, PartDecoder, HierarchicalPosePrior


class TestPartEncoder:
    def test_output_shape(self):
        """PartEncoder should map [B, K, 2] to [B, num_tokens, D]."""
        enc = PartEncoder(num_keypoints=5, token_dim=512, num_tokens=10)
        x = torch.randn(4, 5, 2)
        out = enc(x)
        assert out.shape == (4, 10, 512)

    def test_gradient_flow(self):
        enc = PartEncoder(num_keypoints=6, token_dim=128, num_tokens=12)
        x = torch.randn(2, 6, 2, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestPartDecoder:
    def test_output_shape(self):
        """PartDecoder should map [B, num_tokens, D] back to [B, K, 2]."""
        dec = PartDecoder(num_tokens=10, token_dim=512, num_keypoints=5)
        x = torch.randn(4, 10, 512)
        out = dec(x)
        assert out.shape == (4, 5, 2)


class TestHierarchicalPosePrior:
    def test_forward_shape(self):
        """Full prior should reconstruct [B, 17, 2]."""
        prior = HierarchicalPosePrior(
            codebook_size=128, embed_dim=64,
            head_keypoints=[0,1,2,3,4],
            arm_keypoints=[5,6,7,8,9,10],
            leg_keypoints=[11,12,13,14,15,16],
        )
        pose = torch.randn(2, 17, 2)
        recon, indices, loss = prior(pose)
        assert recon.shape == (2, 17, 2)
        assert isinstance(indices, dict)
        assert indices['global'].shape[0] == 2
        assert loss.ndim == 0

    def test_reconstruction_loss_decreases(self):
        """After a few optimization steps, reconstruction loss should decrease."""
        prior = HierarchicalPosePrior(
            codebook_size=64, embed_dim=32,
            head_keypoints=[0,1,2,3,4],
            arm_keypoints=[5,6,7,8,9,10],
            leg_keypoints=[11,12,13,14,15,16],
        )
        optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)
        pose = torch.randn(8, 17, 2)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            recon, _, vq_loss = prior(pose)
            recon_loss = torch.nn.functional.smooth_l1_loss(recon, pose)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            losses.append(recon_loss.item())

        assert losses[-1] < losses[0]

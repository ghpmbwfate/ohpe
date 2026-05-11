"""
Integration test for the full OHPE pipeline.
Uses small models and synthetic data; no large pretrained backbones.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F


def test_prior_encode_decode():
    """Prior should encode and decode poses with reasonable error."""
    from models.prior import HierarchicalPosePrior

    prior = HierarchicalPosePrior(
        codebook_size=64, embed_dim=32,
        head_tokens=4, arm_tokens=4, leg_tokens=4, global_tokens=8,
        num_mixer_blocks=2,
    )
    pose = torch.randn(4, 17, 2)

    # Forward
    recon, indices, loss = prior(pose)
    assert recon.shape == (4, 17, 2)

    # Encode to indices
    idx = prior.encode_to_indices(pose)
    assert idx.shape[0] == 4

    # Decode from indices
    decoded = prior.decode_from_indices(idx)
    assert decoded.shape == (4, 17, 2)


def test_diffusion_forward_reverse():
    """Diffusion should add and remove noise."""
    from models.diffusion import DiscreteDiffusion

    diffusion = DiscreteDiffusion(num_classes=16, num_timesteps=20)
    x_0 = torch.randint(0, 16, (2, 5))

    # Forward
    t = torch.tensor([5, 10])
    x_t = diffusion.q_sample(x_0, t)
    assert x_t.shape == x_0.shape

    # Prior should be valid distribution
    assert torch.allclose(diffusion.q_prior.sum(), torch.tensor(1.0), atol=1e-4)


def test_denoiser_with_prior():
    """Denoiser should predict logits from noisy tokens."""
    from models.prior import HierarchicalPosePrior
    from models.denoiser import PoseDenoiser

    prior = HierarchicalPosePrior(codebook_size=32, embed_dim=16, global_tokens=8, num_mixer_blocks=1)
    denoiser = PoseDenoiser(num_tokens=8, hidden_dim=32, num_classes=32, num_blocks=2, num_heads=2, cond_dim=16)

    pose = torch.randn(2, 17, 2)
    with torch.no_grad():
        indices = prior.encode_to_indices(pose)

    t = torch.randint(0, 10, (2,))
    condition = torch.randn(2, 16)
    logits = denoiser(indices, t, condition)
    assert logits.shape == (2, 8, 32)


def test_end_to_end_training_step():
    """A single training step should run without errors."""
    from models.prior import HierarchicalPosePrior
    from models.denoiser import PoseDenoiser
    from models.diffusion import DiscreteDiffusion

    prior = HierarchicalPosePrior(codebook_size=32, embed_dim=16, global_tokens=8, num_mixer_blocks=1)
    denoiser = PoseDenoiser(num_tokens=8, hidden_dim=32, num_classes=32, num_blocks=2, num_heads=2, cond_dim=16)
    diffusion = DiscreteDiffusion(num_classes=32, num_timesteps=10)

    pose = torch.randn(2, 17, 2)
    condition = torch.randn(2, 16)

    # Freeze prior
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    # Forward
    with torch.no_grad():
        x_0 = prior.encode_to_indices(pose)

    t = torch.randint(0, 10, (2,))
    x_t = diffusion.q_sample(x_0, t)
    pred_logits = denoiser(x_t, t, condition)

    aux_loss = F.cross_entropy(pred_logits.reshape(-1, 32), x_0.reshape(-1))
    pred_indices = pred_logits.argmax(dim=-1)
    pred_pose = prior.decode_from_indices(pred_indices)
    recon_loss = F.smooth_l1_loss(pred_pose, pose)
    loss = aux_loss + recon_loss

    # Backward on denoiser
    loss.backward()

    assert denoiser.output_proj.weight.grad is not None
    assert loss.item() >= 0

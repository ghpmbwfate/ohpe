"""
Quick training smoke test.
Runs a few epochs on tiny synthetic data to verify the training pipeline works.
Use this before uploading to server to avoid wasted effort.
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Set cache dirs before importing models
os.environ.setdefault('TORCH_HOME', './models_pretrained')
os.environ.setdefault('HF_HOME', './models_pretrained')

from models.prior import HierarchicalPosePrior
from models.denoiser import PoseDenoiser
from models.diffusion import DiscreteDiffusion
from utils.data_utils import PoseDataset, collate_fn

# Try to import MultimodalConditionEncoder, fallback to mock if timm/open_clip missing
try:
    from models.multimodal_encoder import MultimodalConditionEncoder
except ImportError:
    print("[WARN] timm/open_clip not available, using mock condition encoder for testing")

    class MultimodalConditionEncoder(torch.nn.Module):
        def __init__(self, cond_dim=64):
            super().__init__()
            self.cond_dim = cond_dim
            self.proj = torch.nn.Linear(3 * 256 * 256, cond_dim)

        def forward(self, image, text_input):
            b = image.size(0)
            x = image.view(b, -1)
            return self.proj(x)


def test_prior_training(num_samples=100, num_epochs=3, device='cpu'):
    """Test Stage 1: VQ-VAE prior training."""
    print("=" * 60)
    print(f"TEST: Prior Training (samples={num_samples}, epochs={num_epochs}, device={device})")
    print("=" * 60)

    dataset = PoseDataset(num_samples=num_samples, num_keypoints=17, image_size=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = HierarchicalPosePrior(
        codebook_size=128,  # smaller for speed
        embed_dim=64,
        head_tokens=4,
        arm_tokens=4,
        leg_tokens=4,
        global_tokens=8,
        num_mixer_blocks=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            _, poses, _, _ = batch
            poses = poses.to(device)

            optimizer.zero_grad()
            recon, indices, vq_loss = model(poses)
            recon_loss = F.smooth_l1_loss(recon, poses)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")

    # Check loss decreased
    if losses[-1] < losses[0]:
        print(f"  PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    else:
        print(f"  WARN: Loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    return losses[-1] < losses[0]


def test_diffusion_training(num_samples=50, num_epochs=3, device='cpu'):
    """Test Stage 2: Diffusion model training."""
    print("=" * 60)
    print(f"TEST: Diffusion Training (samples={num_samples}, epochs={num_epochs}, device={device})")
    print("=" * 60)

    dataset = PoseDataset(num_samples=num_samples, num_keypoints=17, image_size=256)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Small prior for speed
    prior = HierarchicalPosePrior(
        codebook_size=128, embed_dim=64,
        head_tokens=4, arm_tokens=4, leg_tokens=4, global_tokens=8,
        num_mixer_blocks=2,
    ).to(device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    # Small denoiser
    num_tokens = prior.global_encoder.num_tokens
    denoiser = PoseDenoiser(
        num_tokens=num_tokens,
        hidden_dim=128,
        num_classes=128,
        num_blocks=2,
        num_heads=4,
        cond_dim=64,
    ).to(device)

    cond_encoder = MultimodalConditionEncoder(cond_dim=64).to(device)

    diffusion = DiscreteDiffusion(
        num_classes=128,
        num_timesteps=10,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(denoiser.parameters()) + list(cond_encoder.parameters()),
        lr=0.001,
    )

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            images, poses, _, texts = batch
            images = images.to(device)
            poses = poses.to(device)

            with torch.no_grad():
                x_0 = prior.encode_to_indices(poses)

            condition = cond_encoder(images, texts)

            t = torch.randint(0, diffusion.num_timesteps, (images.size(0),), device=device)
            x_t = diffusion.q_sample(x_0, t)
            pred_logits = denoiser(x_t, t, condition)

            aux_loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                x_0.reshape(-1)
            )

            pred_probs = F.softmax(pred_logits, dim=-1)
            pred_indices = pred_probs.argmax(dim=-1)
            pred_pose = prior.decode_from_indices(pred_indices)
            recon_loss = F.smooth_l1_loss(pred_pose, poses)

            loss = 0.0005 * aux_loss + recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")

    if losses[-1] < losses[0]:
        print(f"  PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    else:
        print(f"  WARN: Loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    return losses[-1] < losses[0]


def test_transition_matrix():
    """Verify transition matrix probabilities are valid."""
    print("=" * 60)
    print("TEST: Transition Matrix Validity")
    print("=" * 60)

    diffusion = DiscreteDiffusion(num_classes=2048, num_timesteps=100, device='cpu')
    mats = diffusion.transition_matrices

    # Check row sums
    row_sums = mats.sum(dim=-1)
    min_sum = row_sums.min().item()
    max_sum = row_sums.max().item()
    print(f"  Row sum range: [{min_sum:.8f}, {max_sum:.8f}]")

    # Check no negatives
    min_val = mats.min().item()
    print(f"  Min value: {min_val:.8f}")

    # Check Obs token transitions
    obs_stay = mats[:, -1, -1]
    print(f"  Obs token stay prob range: [{obs_stay.min():.6f}, {obs_stay.max():.6f}]")

    valid = (min_sum >= 0.999 and max_sum <= 1.001 and min_val >= -1e-8
             and obs_stay.min() > 0.99)

    if valid:
        print("  PASS: Transition matrix is valid")
    else:
        print("  FAIL: Transition matrix has issues")

    return valid


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    all_pass = True

    # Test 1: Transition matrix
    all_pass &= test_transition_matrix()
    print()

    # Test 2: Prior training
    all_pass &= test_prior_training(num_samples=100, num_epochs=3, device=device)
    print()

    # Test 3: Diffusion training
    all_pass &= test_diffusion_training(num_samples=50, num_epochs=3, device=device)
    print()

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED - Ready to train on server!")
    else:
        print("SOME TESTS FAILED - Fix issues before uploading")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

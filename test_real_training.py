"""
Quick real-data training test on a small COCO subset.
Runs a few epochs to verify the full training pipeline with real images.
Usage:
    python test_real_training.py /path/to/coco/train2017 /path/to/coco/annotations/person_keypoints_train2017.json
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault('TORCH_HOME', './models_pretrained')
os.environ.setdefault('HF_HOME', './models_pretrained')

from models.prior import HierarchicalPosePrior
from models.denoiser import PoseDenoiser
from models.diffusion import DiscreteDiffusion
from utils.data_utils import COCOPoseDataset, PoseDataset, collate_fn

try:
    from models.multimodal_encoder import MultimodalConditionEncoder
except ImportError:
    print("[WARN] timm/open_clip not available, using mock condition encoder")

    class MultimodalConditionEncoder(torch.nn.Module):
        def __init__(self, cond_dim=64):
            super().__init__()
            self.cond_dim = cond_dim
            self.proj = torch.nn.Linear(3 * 256 * 256, cond_dim)

        def forward(self, image, text_input):
            b = image.size(0)
            x = image.view(b, -1)
            return self.proj(x)


def test_prior_on_real_data(root_dir, ann_file, num_epochs=2, max_batches=50, device='cuda'):
    """Test prior training with real COCO data."""
    print("=" * 60)
    print("TEST: Prior Training on Real COCO Data")
    print("=" * 60)

    if not os.path.exists(root_dir) or not os.path.exists(ann_file):
        print(f"[SKIP] Dataset not found, falling back to synthetic data")
        dataset = PoseDataset(num_samples=500, num_keypoints=17, image_size=256)
    else:
        dataset = COCOPoseDataset(root_dir, ann_file, image_size=256, split='train')

    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=2, collate_fn=collate_fn)

    model = HierarchicalPosePrior(
        codebook_size=128, embed_dim=64,
        head_tokens=4, arm_tokens=4, leg_tokens=4, global_tokens=8,
        num_mixer_blocks=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            _, poses, _, _ = batch
            poses = poses.to(device)

            optimizer.zero_grad()
            recon, indices, vq_loss = model(poses)
            recon_loss = F.smooth_l1_loss(recon, poses)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / min(len(loader), max_batches)
        losses.append(avg)
        print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg:.4f}")

    if losses[-1] < losses[0]:
        print(f"  PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        return True
    else:
        print(f"  WARN: Loss did not decrease")
        return False


def test_diffusion_on_real_data(root_dir, ann_file, num_epochs=2, max_batches=30, device='cuda'):
    """Test diffusion training with real COCO data."""
    print("=" * 60)
    print("TEST: Diffusion Training on Real COCO Data")
    print("=" * 60)

    if not os.path.exists(root_dir) or not os.path.exists(ann_file):
        print(f"[SKIP] Dataset not found, falling back to synthetic data")
        dataset = PoseDataset(num_samples=200, num_keypoints=17, image_size=256)
    else:
        dataset = COCOPoseDataset(root_dir, ann_file, image_size=256, split='train')

    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        num_workers=2, collate_fn=collate_fn)

    prior = HierarchicalPosePrior(
        codebook_size=128, embed_dim=64,
        head_tokens=4, arm_tokens=4, leg_tokens=4, global_tokens=8,
        num_mixer_blocks=2,
    ).to(device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    num_tokens = prior.global_encoder.num_tokens
    denoiser = PoseDenoiser(
        num_tokens=num_tokens, hidden_dim=128, num_classes=128,
        num_blocks=2, num_heads=4, cond_dim=64,
    ).to(device)
    cond_encoder = MultimodalConditionEncoder(cond_dim=64).to(device)
    diffusion = DiscreteDiffusion(num_classes=128, num_timesteps=10, device=device)

    optimizer = torch.optim.AdamW(
        list(denoiser.parameters()) + list(cond_encoder.parameters()),
        lr=0.001,
    )

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
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

        avg = total_loss / min(len(loader), max_batches)
        losses.append(avg)
        print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg:.4f}")

    if losses[-1] < losses[0]:
        print(f"  PASS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        return True
    else:
        print(f"  WARN: Loss did not decrease")
        return False


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Try to get paths from args or auto-detect ./data
    if len(sys.argv) >= 3:
        root_dir = sys.argv[1]
        ann_file = sys.argv[2]
    else:
        root_dir = os.environ.get('COCO_TRAIN_ROOT', './data/train2017')
        ann_file = os.environ.get('COCO_TRAIN_ANN', './data/annotations/person_keypoints_train2017.json')

    all_pass = True
    all_pass &= test_prior_on_real_data(root_dir, ann_file, device=device)
    print()
    all_pass &= test_diffusion_on_real_data(root_dir, ann_file, device=device)
    print()

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED - Ready for full training!")
    else:
        print("SOME TESTS WARNED - Check loss trends")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

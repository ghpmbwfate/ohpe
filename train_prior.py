"""
Training script for Hierarchical VQ-VAE Prior.
"""
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set cache dirs for pretrained models before importing model classes
os.environ.setdefault('TORCH_HOME', './models_pretrained')
os.environ.setdefault('HF_HOME', './models_pretrained')

from models.prior import HierarchicalPosePrior
from utils.data_utils import PoseDataset, collate_fn


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0

    for batch in dataloader:
        _, poses, _, _ = batch
        poses = poses.to(device)

        optimizer.zero_grad()
        recon, indices, vq_loss = model(poses)
        recon_loss = F.smooth_l1_loss(recon, poses)
        loss = recon_loss + vq_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_vq += vq_loss.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_vq / n


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_recon = 0
    total_vq = 0

    for batch in dataloader:
        _, poses, _, _ = batch
        poses = poses.to(device)
        recon, indices, vq_loss = model(poses)
        recon_loss = F.smooth_l1_loss(recon, poses)
        total_recon += recon_loss.item()
        total_vq += vq_loss.item()

    n = len(dataloader)
    return total_recon / n, total_vq / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/prior_vqvae.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Dataset
    train_ds = PoseDataset(num_samples=5000, num_keypoints=cfg['num_keypoints'])
    val_ds = PoseDataset(num_samples=500, num_keypoints=cfg['num_keypoints'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], collate_fn=collate_fn)

    # Model
    model = HierarchicalPosePrior(
        codebook_size=cfg['codebook_size'],
        embed_dim=cfg['embed_dim'],
        head_tokens=cfg['head_tokens'],
        arm_tokens=cfg['arm_tokens'],
        leg_tokens=cfg['leg_tokens'],
        global_tokens=cfg['global_tokens'],
        num_mixer_blocks=cfg['num_mixer_blocks'],
        expansion_factor=cfg['expansion_factor'],
        dropout=cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    best_val = float('inf')
    for epoch in range(cfg['epochs']):
        train_loss, train_recon, train_vq = train_epoch(model, train_loader, optimizer, device)
        val_recon, val_vq = validate(model, val_loader, device)
        val_loss = val_recon + val_vq

        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} (recon={train_recon:.4f}, vq={train_vq:.4f}) | "
              f"val_loss={val_loss:.4f} (recon={val_recon:.4f}, vq={val_vq:.4f})")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'best.pth'))

        if (epoch + 1) % cfg.get('save_interval', 10) == 0:
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()

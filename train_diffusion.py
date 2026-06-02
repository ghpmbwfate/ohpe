"""
Training script for Discrete Diffusion Model.
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
from models.denoiser import PoseDenoiser
from models.diffusion import DiscreteDiffusion
from models.multimodal_encoder import MultimodalConditionEncoder
from utils.data_utils import COCOPoseDataset, PoseDataset, collate_fn


def train_epoch(prior, denoiser, diffusion, cond_encoder, dataloader,
                optimizer, device, eta):
    prior.eval()
    denoiser.train()
    cond_encoder.train()

    total_loss = 0
    total_recon = 0
    total_k0 = 0
    total_vlb = 0
    total_tkn = 0

    for batch in tqdm(dataloader, desc='Train', leave=False):
        images, poses, visibilities, texts = batch
        images = images.to(device)
        poses = poses.to(device)

        # Encode poses to indices with frozen prior
        with torch.no_grad():
            x_0 = prior.encode_to_indices(poses)  # [B, N]

        # Generate conditions using text prompts
        condition = cond_encoder(images, texts)

        # Sample timesteps and compute full paper losses
        t = torch.randint(0, diffusion.num_timesteps, (images.size(0),), device=device)
        loss, recon_loss, l_k0, l_vlb, l_tkn = diffusion.p_losses(
            denoiser, x_0, t, condition, poses, prior, eta
        )

<<<<<<< HEAD
        # Reconstruction loss via hard decoding (stable, no gradient explosion)
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_indices = pred_probs.argmax(dim=-1)
        with torch.no_grad():
            pred_pose = prior.decode_from_indices(pred_indices)
        recon_loss = F.smooth_l1_loss(pred_pose, poses)

        loss = eta * aux_loss + recon_loss

=======
>>>>>>> 17b401ce15c74c24cc904d1c96649063388352bd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_k0 += l_k0.item()
        total_vlb += l_vlb.item()
        total_tkn += l_tkn.item()

        # NaN detection
        if torch.isnan(loss):
            print(f"NaN detected! aux_loss={aux_loss.item():.4f}, recon_loss={recon_loss.item():.4f}")
            break

    n = len(dataloader)
    return (total_loss / n, total_recon / n, total_k0 / n,
            total_vlb / n, total_tkn / n)


@torch.no_grad()
def validate(prior, denoiser, diffusion, cond_encoder, dataloader, device, eta):
    prior.eval()
    denoiser.eval()
    cond_encoder.eval()

    total_loss = 0
    total_recon = 0
    total_k0 = 0
    total_vlb = 0
    total_tkn = 0

    for batch in tqdm(dataloader, desc='Val', leave=False):
        images, poses, visibilities, texts = batch
        images = images.to(device)
        poses = poses.to(device)

        with torch.no_grad():
            x_0 = prior.encode_to_indices(poses)

        condition = cond_encoder(images, texts)

        t = torch.randint(0, diffusion.num_timesteps, (images.size(0),), device=device)
        loss, recon_loss, l_k0, l_vlb, l_tkn = diffusion.p_losses(
            denoiser, x_0, t, condition, poses, prior, eta
        )

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_k0 += l_k0.item()
        total_vlb += l_vlb.item()
        total_tkn += l_tkn.item()

    n = len(dataloader)
    return (total_loss / n, total_recon / n, total_k0 / n,
            total_vlb / n, total_tkn / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/diffusion.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_dir', default='./data',
                        help='COCO dataset root directory')
    parser.add_argument('--use_coco', action='store_true', default=True,
                        help='Use COCO dataset (default: True, use --no-use_coco for synthetic)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Dataset
    train_ann = os.path.join(args.data_dir, 'annotations', 'person_keypoints_train2017.json')
    val_ann = os.path.join(args.data_dir, 'annotations', 'person_keypoints_val2017.json')
    train_root = os.path.join(args.data_dir, 'train2017')
    val_root = os.path.join(args.data_dir, 'val2017')

    if args.use_coco and os.path.exists(train_ann) and os.path.exists(train_root):
        print(f"Using COCO dataset from {args.data_dir}")
        train_ds = COCOPoseDataset(train_root, train_ann, image_size=cfg.get('image_size', 224),
                                    split='train')
        val_ds = COCOPoseDataset(val_root, val_ann, image_size=cfg.get('image_size', 224),
                                  split='val')
    else:
        print("COCO not found, falling back to synthetic data")
        train_ds = PoseDataset(num_samples=5000, image_size=cfg.get('image_size', 224))
        val_ds = PoseDataset(num_samples=500, image_size=cfg.get('image_size', 224))

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], collate_fn=collate_fn)

    # Load frozen prior
    prior = HierarchicalPosePrior(
        codebook_size=cfg['num_classes'],
        embed_dim=512,  # Must match prior training config, NOT cfg['hidden_dim']
    ).to(device)
    prior.load_state_dict(torch.load(cfg['prior_checkpoint'], map_location=device))
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    # Denoiser + condition encoder
    # Infer num_tokens from prior
    num_tokens = prior.global_encoder.num_tokens
    denoiser = PoseDenoiser(
        num_tokens=num_tokens,
        hidden_dim=cfg['hidden_dim'],
        num_classes=cfg['num_classes'],
        num_blocks=cfg['num_blocks'],
        num_heads=cfg['num_heads'],
        cond_dim=cfg['cond_dim'],
        mlp_ratio=cfg['mlp_ratio'],
        dropout=cfg['dropout'],
    ).to(device)

    cond_encoder = MultimodalConditionEncoder(cond_dim=cfg['cond_dim']).to(device)

    diffusion = DiscreteDiffusion(
        num_classes=cfg['num_classes'],
        num_timesteps=cfg['num_timesteps'],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(denoiser.parameters()) + list(cond_encoder.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        betas=(0.9, 0.96),
    )

    best_val = float('inf')
    for epoch in range(cfg['epochs']):
        (train_loss, train_recon, train_k0,
         train_vlb, train_tkn) = train_epoch(
            prior, denoiser, diffusion, cond_encoder, train_loader,
            optimizer, device, cfg['eta']
        )
        (val_loss, val_recon, val_k0,
         val_vlb, val_tkn) = validate(
            prior, denoiser, diffusion, cond_encoder, val_loader,
            device, cfg['eta']
        )

        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} (recon={train_recon:.4f}, k0={train_k0:.4f}, "
              f"vlb={train_vlb:.4f}, tkn={train_tkn:.4f}) | "
              f"val_loss={val_loss:.4f} (recon={val_recon:.4f}, k0={val_k0:.4f}, "
              f"vlb={val_vlb:.4f}, tkn={val_tkn:.4f})")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'denoiser': denoiser.state_dict(),
                'cond_encoder': cond_encoder.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
            }, os.path.join(cfg['save_dir'], 'best.pth'))

        if (epoch + 1) % cfg.get('save_interval', 10) == 0:
            torch.save({
                'denoiser': denoiser.state_dict(),
                'cond_encoder': cond_encoder.state_dict(),
                'epoch': epoch + 1,
            }, os.path.join(cfg['save_dir'], f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()

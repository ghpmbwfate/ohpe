"""
Training script for Discrete Diffusion Model.
"""
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.prior import HierarchicalPosePrior
from models.denoiser import PoseDenoiser
from models.diffusion import DiscreteDiffusion
from models.multimodal_encoder import MultimodalConditionEncoder
from utils.data_utils import PoseDataset, collate_fn


def train_epoch(prior, denoiser, diffusion, cond_encoder, dataloader,
                optimizer, device, eta):
    prior.eval()
    denoiser.train()
    cond_encoder.train()

    total_loss = 0
    total_recon = 0
    total_aux = 0

    for batch in dataloader:
        images, poses, visibilities, texts = batch
        images = images.to(device)
        poses = poses.to(device)

        # Encode poses to indices with frozen prior
        with torch.no_grad():
            x_0 = prior.encode_to_indices(poses)  # [B, N]

        # Generate conditions
        # TODO: properly tokenize texts for CLIP
        # For now, use dummy text tokens
        text_tokens = torch.randint(0, 1000, (images.size(0), 10), device=device)
        condition = cond_encoder(images, text_tokens)

        # Sample timesteps
        t = torch.randint(0, diffusion.num_timesteps, (images.size(0),), device=device)

        # Forward diffusion + denoise
        x_t = diffusion.q_sample(x_0, t)
        pred_logits = denoiser(x_t, t, condition)

        # Losses
        aux_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            x_0.reshape(-1)
        )

        # Reconstruction loss via predicted indices
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_indices = pred_probs.argmax(dim=-1)
        pred_pose = prior.decode_from_indices(pred_indices)
        recon_loss = F.smooth_l1_loss(pred_pose, poses)

        loss = eta * aux_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_aux += aux_loss.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_aux / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/diffusion.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Dataset
    train_ds = PoseDataset(num_samples=5000)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], collate_fn=collate_fn)

    # Load frozen prior
    prior = HierarchicalPosePrior(
        codebook_size=cfg['num_classes'],
        embed_dim=cfg['hidden_dim'],
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

    for epoch in range(cfg['epochs']):
        train_loss, train_recon, train_aux = train_epoch(
            prior, denoiser, diffusion, cond_encoder, train_loader,
            optimizer, device, cfg['eta']
        )
        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"loss={train_loss:.4f} recon={train_recon:.4f} aux={train_aux:.4f}")

        if (epoch + 1) % cfg.get('save_interval', 10) == 0:
            torch.save({
                'denoiser': denoiser.state_dict(),
                'cond_encoder': cond_encoder.state_dict(),
            }, os.path.join(cfg['save_dir'], f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()

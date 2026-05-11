"""
Vector Quantization layer with EMA update.
Reference: VQ-VAE paper (van den Oord et al., NeurIPS 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """
    Vector Quantization layer with optional EMA codebook update.
    Input: z [B, N, D] (encoder output)
    Output: z_q [B, N, D], indices [B, N], loss (scalar)
    """
    def __init__(self, num_embed, embed_dim, beta=0.25, use_ema=True, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embed, 1.0 / num_embed)

        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embed))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, z):
        # z: [B, N, D]
        b, n, d = z.shape
        assert d == self.embed_dim

        z_flat = z.reshape(-1, d)  # [B*N, D]

        # Compute distances to all embeddings: [B*N, num_embed]
        # Using ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
        z_sq = z_flat.pow(2).sum(dim=1, keepdim=True)          # [B*N, 1]
        e_sq = self.embedding.weight.pow(2).sum(dim=1)         # [num_embed]
        ze = torch.matmul(z_flat, self.embedding.weight.t())   # [B*N, num_embed]
        distances = z_sq + e_sq - 2 * ze                       # [B*N, num_embed]

        indices = distances.argmin(dim=1)                       # [B*N]
        z_q = self.embedding(indices).view(b, n, d)            # [B, N, D]

        # Compute losses
        # codebook_loss: move encoder output toward embedding (only encoder grad)
        codebook_loss = F.mse_loss(z_q.detach(), z)
        # commitment_loss: move encoder output toward embedding (only encoder grad)
        commitment_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # EMA update (during training)
        if self.training and self.use_ema:
            self._ema_update(z_flat, indices)

        # Straight-Through Estimator: pass gradient from z_q to z
        z_q = z + (z_q - z).detach()
        return z_q, indices.view(b, n), loss

    @torch.no_grad()
    def _ema_update(self, z_flat, indices):
        """Exponential Moving Average update of codebook."""
        enc_one_hot = F.one_hot(indices, self.num_embed).float()  # [B*N, num_embed]

        # Update cluster sizes
        self.ema_cluster_size.mul_(self.decay).add_(
            enc_one_hot.sum(dim=0), alpha=1 - self.decay
        )

        # Update embeddings
        embed_sum = z_flat.t() @ enc_one_hot  # [D, num_embed]
        self.ema_w.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

        # Normalize
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.num_embed * self.eps) * n
        )
        self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

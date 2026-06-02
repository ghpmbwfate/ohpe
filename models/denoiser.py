"""
Pose Denoiser: Transformer with AdaLN for discrete diffusion.
"""
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (DDPM standard)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] or [B, 1]
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Formula: AdaLN(f, t) = (1 + a_t) * LayerNorm(f) + b_t
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Project condition to scale + shift
        self.scale_shift = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)

    def forward(self, x, t_emb):
        # x: [B, N, hidden_dim]
        # t_emb: [B, cond_dim]
        scale, shift = self.scale_shift(t_emb).chunk(2, dim=-1)  # [B, hidden_dim]
        scale = scale.unsqueeze(1)   # [B, 1, hidden_dim]
        shift = shift.unsqueeze(1)   # [B, 1, hidden_dim]
        return self.norm(x) * (1 + scale) + shift


class DenoiserBlock(nn.Module):
    """
    Single Transformer block with:
    - AdaLN-conditioned Self-Attention
    - AdaLN-conditioned Cross-Attention (condition as KV)
    - AdaLN-conditioned FFN
    """
    def __init__(self, hidden_dim=1024, num_heads=8, cond_dim=512,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        # Self-attention
        self.adaLN_sa = AdaLN(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.adaLN_ca = AdaLN(hidden_dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.adaLN_ffn = AdaLN(hidden_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, condition, t_emb):
        # x: [B, N, D]
        # condition: [B, M, D]
        # t_emb: [B, cond_dim]

        # Self-attention
        x_norm = self.adaLN_sa(x, t_emb)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention (x queries, condition provides KV)
        x_norm = self.adaLN_ca(x, t_emb)
        cross_out, _ = self.cross_attn(x_norm, condition, condition)
        x = x + cross_out

        # FFN
        x_norm = self.adaLN_ffn(x, t_emb)
        x = x + self.ffn(x_norm)
        return x


class PoseDenoiser(nn.Module):
    """
    Full denoiser: 19 Transformer blocks (paper spec).
    Input: noisy token indices [B, N], timestep [B], condition [B, cond_dim]
    Output: logits [B, N, num_classes]
    """
    def __init__(self, num_tokens=34, hidden_dim=1024, num_classes=2048,
                 num_blocks=19, num_heads=8, cond_dim=512, mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_states = num_classes + 1  # +1 for Obs token

        # Token embedding
        self.token_embed = nn.Embedding(self.num_states, hidden_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Condition projection to hidden_dim for cross-attention
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(hidden_dim, num_heads, cond_dim, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_t, t, condition):
        """
        x_t: [B, N]  noisy token indices (0..num_classes)
        t: [B]       timestep
        condition: [B, cond_dim]
        Returns: logits [B, N, num_classes]
        """
        # Token embedding
        x = self.token_embed(x_t)  # [B, N, hidden_dim]

        # Timestep embedding
        t_emb = self.time_embed(t)  # [B, cond_dim]

        # Project condition for cross-attention
        cond_seq = self.cond_proj(condition).unsqueeze(1)  # [B, 1, hidden_dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond_seq, t_emb)

        # Output
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [B, N, num_classes]
        return logits

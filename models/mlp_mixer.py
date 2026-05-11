"""
MLP-Mixer Block implementation.
Reference: timm.models.mlp_mixer.MixerBlock
"""
import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """
    MLP-Mixer block with token-mixing and channel-mixing MLPs.
    Input/output shape: [B, N, D]
    """
    def __init__(self, dim, num_patches, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp_tokens = nn.Sequential(
            nn.Linear(num_patches, num_patches * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * expansion_factor, num_patches),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_channels = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        # Token-mixing: transpose to [B, D, N], apply MLP, transpose back
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.mlp_tokens(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = x + residual

        # Channel-mixing: MLP across channels
        residual = x
        x = self.norm2(x)
        x = self.mlp_channels(x)  # [B, N, D]
        x = x + residual
        return x

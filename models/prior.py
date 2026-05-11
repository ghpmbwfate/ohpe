"""
Hierarchical Pose Prior based on VQ-VAE-2.
Learns part-aware discrete representations of 2D human poses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp_mixer import MixerBlock
from models.vector_quantize import VectorQuantize


class PartEncoder(nn.Module):
    """
    Encodes a body part (subset of keypoints) into a sequence of tokens.
    Architecture: flatten keypoints -> linear projection -> MixerBlocks
    """
    def __init__(self, num_keypoints, token_dim, num_tokens,
                 num_mixer_blocks=4, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Flatten keypoints [K, 2] -> [K*2] and project to token_dim
        # Then expand to num_tokens via linear projection
        self.proj = nn.Sequential(
            nn.Linear(num_keypoints * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.to_tokens = nn.Linear(token_dim, num_tokens * token_dim)

        self.mixers = nn.Sequential(*[
            MixerBlock(token_dim, num_tokens, expansion_factor, dropout)
            for _ in range(num_mixer_blocks)
        ])

    def forward(self, pose_part):
        # pose_part: [B, K, 2]
        b = pose_part.shape[0]
        x = pose_part.view(b, -1)              # [B, K*2]
        x = self.proj(x)                        # [B, token_dim]
        x = self.to_tokens(x)                   # [B, num_tokens * token_dim]
        x = x.view(b, self.num_tokens, self.token_dim)  # [B, num_tokens, token_dim]
        x = self.mixers(x)                      # [B, num_tokens, token_dim]
        return x


class PartDecoder(nn.Module):
    """
    Decodes a sequence of tokens back to keypoint coordinates.
    Architecture: MixerBlocks -> linear projection -> reshape to [K, 2]
    """
    def __init__(self, num_tokens, token_dim, num_keypoints,
                 num_mixer_blocks=1, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.mixers = nn.Sequential(*[
            MixerBlock(token_dim, num_tokens, expansion_factor, dropout)
            for _ in range(num_mixer_blocks)
        ])
        # Aggregate tokens and project to keypoints
        self.to_pose = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_keypoints * 2),
        )

    def forward(self, tokens):
        # tokens: [B, num_tokens, token_dim]
        x = self.mixers(tokens)                 # [B, num_tokens, token_dim]
        # Mean pool across tokens then project
        x = x.mean(dim=1)                       # [B, token_dim]
        x = self.to_pose(x)                     # [B, num_keypoints * 2]
        b = x.shape[0]
        num_keypoints = x.shape[-1] // 2
        x = x.view(b, num_keypoints, 2)         # [B, K, 2]
        return x


class HierarchicalPosePrior(nn.Module):
    """
    Full hierarchical VQ-VAE prior with 4 levels:
    head, arms, legs (local) + global.
    """
    def __init__(self, codebook_size=2048, embed_dim=512,
                 head_keypoints=None, arm_keypoints=None, leg_keypoints=None,
                 head_tokens=10, arm_tokens=12, leg_tokens=12, global_tokens=34,
                 num_mixer_blocks=4, expansion_factor=4, dropout=0.0):
        super().__init__()
        # Default COCO keypoint grouping
        if head_keypoints is None:
            head_keypoints = [0, 1, 2, 3, 4]
        if arm_keypoints is None:
            arm_keypoints = [5, 6, 7, 8, 9, 10]
        if leg_keypoints is None:
            leg_keypoints = [11, 12, 13, 14, 15, 16]

        self.head_keypoints = head_keypoints
        self.arm_keypoints = arm_keypoints
        self.leg_keypoints = leg_keypoints
        self.num_keypoints = 17

        # Local encoders
        self.head_encoder = PartEncoder(
            len(head_keypoints), embed_dim, head_tokens, num_mixer_blocks, expansion_factor, dropout)
        self.arm_encoder = PartEncoder(
            len(arm_keypoints), embed_dim, arm_tokens, num_mixer_blocks, expansion_factor, dropout)
        self.leg_encoder = PartEncoder(
            len(leg_keypoints), embed_dim, leg_tokens, num_mixer_blocks, expansion_factor, dropout)
        self.global_encoder = PartEncoder(
            self.num_keypoints, embed_dim, global_tokens, num_mixer_blocks, expansion_factor, dropout)

        # Local codebooks
        self.head_vq = VectorQuantize(codebook_size, embed_dim)
        self.arm_vq = VectorQuantize(codebook_size, embed_dim)
        self.leg_vq = VectorQuantize(codebook_size, embed_dim)

        # Global fusion: concatenate local quantized tokens + global encoded tokens
        total_local_tokens = head_tokens + arm_tokens + leg_tokens
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.global_vq = VectorQuantize(codebook_size, embed_dim)

        # Global decoder
        self.decoder = PartDecoder(
            global_tokens, embed_dim, self.num_keypoints, num_mixer_blocks=1)

    def forward(self, pose_2d):
        # pose_2d: [B, 17, 2]
        b = pose_2d.shape[0]

        # Extract parts
        head_part = pose_2d[:, self.head_keypoints, :]    # [B, 5, 2]
        arm_part = pose_2d[:, self.arm_keypoints, :]      # [B, 6, 2]
        leg_part = pose_2d[:, self.leg_keypoints, :]      # [B, 6, 2]

        # Local encoding
        head_feat = self.head_encoder(head_part)          # [B, head_tokens, D]
        arm_feat = self.arm_encoder(arm_part)             # [B, arm_tokens, D]
        leg_feat = self.leg_encoder(leg_part)             # [B, leg_tokens, D]
        global_feat = self.global_encoder(pose_2d)        # [B, global_tokens, D]

        # Local quantization
        head_q, head_idx, head_loss = self.head_vq(head_feat)
        arm_q, arm_idx, arm_loss = self.arm_vq(arm_feat)
        leg_q, leg_idx, leg_loss = self.leg_vq(leg_feat)

        # Fuse local + global
        local_q = torch.cat([head_q, arm_q, leg_q], dim=1)  # [B, local_tokens, D]
        # Combine with global features (simple sum on first few tokens)
        min_len = min(local_q.shape[1], global_feat.shape[1])
        fused = global_feat.clone()
        fused[:, :min_len, :] = fused[:, :min_len, :] + self.global_fusion(local_q[:, :min_len, :])

        # Global quantization
        global_q, global_idx, global_loss = self.global_vq(fused)

        # Decode
        recon = self.decoder(global_q)  # [B, 17, 2]

        # Aggregate indices and losses
        indices = {
            'head': head_idx, 'arm': arm_idx,
            'leg': leg_idx, 'global': global_idx,
        }
        vq_loss = head_loss + arm_loss + leg_loss + global_loss
        return recon, indices, vq_loss

    @torch.no_grad()
    def encode_to_indices(self, pose_2d):
        """
        Encode pose to discrete token indices.
        Returns: global_indices [B, global_tokens]
        """
        self.eval()
        b = pose_2d.shape[0]
        head_part = pose_2d[:, self.head_keypoints, :]
        arm_part = pose_2d[:, self.arm_keypoints, :]
        leg_part = pose_2d[:, self.leg_keypoints, :]

        head_feat = self.head_encoder(head_part)
        arm_feat = self.arm_encoder(arm_part)
        leg_feat = self.leg_encoder(leg_part)
        global_feat = self.global_encoder(pose_2d)

        head_q, _, _ = self.head_vq(head_feat)
        arm_q, _, _ = self.arm_vq(arm_feat)
        leg_q, _, _ = self.leg_vq(leg_feat)

        local_q = torch.cat([head_q, arm_q, leg_q], dim=1)
        min_len = min(local_q.shape[1], global_feat.shape[1])
        fused = global_feat.clone()
        fused[:, :min_len, :] = fused[:, :min_len, :] + self.global_fusion(local_q[:, :min_len, :])

        _, global_idx, _ = self.global_vq(fused)
        return global_idx

    def decode_from_indices(self, indices):
        """
        Decode discrete token indices to pose.
        indices: [B, global_tokens]
        Returns: pose [B, 17, 2]
        """
        # Lookup embeddings from global codebook
        tokens = self.global_vq.embedding(indices)  # [B, global_tokens, embed_dim]
        pose = self.decoder(tokens)
        return pose

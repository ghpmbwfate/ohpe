"""
Multimodal Condition Encoder: Swin-Base + CLIP Image + CLIP Text.
All backbone parameters frozen; only projection layers are trainable.
"""
import torch
import torch.nn as nn
import timm
import open_clip


class MultimodalConditionEncoder(nn.Module):
    """
    Extracts and fuses three types of conditional features:
    1. Swin-Base: fine-grained local image features
    2. CLIP Image: high-level semantic visual features
    3. CLIP Text: semantic text features (occlusion descriptions)
    """
    def __init__(self, cond_dim=512, swin_model='swin_base_patch4_window7_224',
                 clip_model='ViT-B-32', clip_pretrained='openai',
                 cache_dir='./models_pretrained'):
        super().__init__()
        self.cond_dim = cond_dim
        self.cache_dir = cache_dir

        # Swin-Base backbone (frozen)
        # Set cache dir via env if provided
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TORCH_HOME'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
        self.swin = timm.create_model(
            swin_model,
            pretrained=True,
            num_classes=0,
        )
        for param in self.swin.parameters():
            param.requires_grad = False
        swin_dim = self.swin.num_features  # 1024 for swin_base

        # CLIP model (frozen)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        clip_dim = self.clip_model.visual.output_dim  # 512 for ViT-B-32
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)

        # Projection layers (trainable)
        self.swin_proj = nn.Sequential(
            nn.Linear(swin_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )
        self.clip_img_proj = nn.Sequential(
            nn.Linear(clip_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )
        self.clip_txt_proj = nn.Sequential(
            nn.Linear(clip_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim * 3, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )

    def forward(self, image, text_input):
        """
        image: [B, 3, H, W]  (expected 224x224)
        text_input: str or list of str or [B, L] tokenized text
        Returns: condition [B, cond_dim]
        """
        # Tokenize if needed
        if isinstance(text_input, str):
            text_input = [text_input]
        if isinstance(text_input, list):
            text_tokens = self.clip_tokenizer(text_input).to(image.device)
        else:
            text_tokens = text_input.to(image.device)

        # Swin features
        swin_feat = self.swin(image)              # [B, 1024]
        swin_cond = self.swin_proj(swin_feat)     # [B, cond_dim]

        # CLIP image features
        clip_img_feat = self.clip_model.encode_image(image)  # [B, 512]
        clip_img_cond = self.clip_img_proj(clip_img_feat)    # [B, cond_dim]

        # CLIP text features
        clip_txt_feat = self.clip_model.encode_text(text_tokens)  # [B, 512]
        clip_txt_cond = self.clip_txt_proj(clip_txt_feat)         # [B, cond_dim]

        # Fusion
        concat = torch.cat([swin_cond, clip_img_cond, clip_txt_cond], dim=-1)
        condition = self.fusion(concat)  # [B, cond_dim]
        return condition

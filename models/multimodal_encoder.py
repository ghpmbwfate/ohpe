"""
Multimodal Condition Encoder: Swin-Base + CLIP Image + CLIP Text.
All backbone parameters frozen; only projection layers are trainable.
Handles offline environments by loading weights from local cache.
"""
import os
import torch
import torch.nn as nn
import timm
import open_clip
import safetensors.torch as st


def _load_swin_local(model, cache_dir):
    """Load Swin weights from local HuggingFace cache or safetensors file."""
    # Try HuggingFace cache structure first
    hf_path = os.path.join(
        cache_dir,
        "hub/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k"
        "/snapshots/a6a1eb2321b4f556fa0fa243fb777d47679f13c9/model.safetensors"
    )
    if os.path.exists(hf_path):
        state_dict = st.load_file(hf_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"[MultimodalEncoder] Loaded Swin weights from {hf_path}")
        return True

    # Try direct .pth file
    pth_path = os.path.join(cache_dir, "swin_base_patch4_window7_224.pth")
    if os.path.exists(pth_path):
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"[MultimodalEncoder] Loaded Swin weights from {pth_path}")
        return True

    print("[MultimodalEncoder] Warning: No local Swin weights found, using random init")
    return False


def _load_clip_local(clip_model_obj, cache_dir):
    """Try to load CLIP weights from local cache."""
    # Try common HuggingFace cache locations for openai CLIP ViT-B-32
    search_dirs = [
        os.path.join(cache_dir, "hub/models--timm--vit_base_patch32_clip_224.openai"),
        os.path.join(cache_dir, "hub/models--laion--ViT-B-32-quickopenai"),
    ]
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        # Find snapshot dirs
        snap_dir = os.path.join(base_dir, "snapshots")
        if not os.path.isdir(snap_dir):
            continue
        for snapshot in os.listdir(snap_dir):
            for fname in ["open_clip_pytorch_model.bin", "model.safetensors", "pytorch_model.bin"]:
                fpath = os.path.join(snap_dir, snapshot, fname)
                if os.path.exists(fpath):
                    if fname.endswith(".safetensors"):
                        state_dict = st.load_file(fpath)
                    else:
                        state_dict = torch.load(fpath, map_location="cpu", weights_only=True)
                    clip_model_obj.load_state_dict(state_dict, strict=False)
                    print(f"[MultimodalEncoder] Loaded CLIP weights from {fpath}")
                    return True

    # Try direct file
    direct_path = os.path.join(cache_dir, "open_clip_pytorch_model.bin")
    if os.path.exists(direct_path):
        state_dict = torch.load(direct_path, map_location="cpu", weights_only=True)
        clip_model_obj.load_state_dict(state_dict, strict=False)
        print(f"[MultimodalEncoder] Loaded CLIP weights from {direct_path}")
        return True

    print("[MultimodalEncoder] Warning: No local CLIP weights found, using random init")
    return False


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

        # Ensure cache directory exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Swin-Base backbone (frozen, load from local)
        self.swin = timm.create_model(
            swin_model,
            pretrained=False,
            num_classes=0,
        )
        _load_swin_local(self.swin, cache_dir)

        for param in self.swin.parameters():
            param.requires_grad = False
        swin_dim = self.swin.num_features  # 1024 for swin_base

        # CLIP model (frozen, try local first then download)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=False,
        )
        _load_clip_local(self.clip_model, cache_dir)

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

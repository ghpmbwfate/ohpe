"""
推理脚本：从单张图像预测人体姿态（带遮挡修复）。

使用方式：
    python inference.py --image path/to/image.jpg --output result.jpg
    python inference.py --image path/to/images/ --output output_dir/

流程：图像 → 多模态条件编码 → 离散扩散采样 → 解码姿态 → 画骨架。
"""
import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from models.diffusion import DiscreteDiffusion
from models.denoiser import PoseDenoiser
from models.multimodal_encoder import MultimodalConditionEncoder
from models.prior import HierarchicalPosePrior
from utils.visualize import denormalize_keypoints, draw_skeleton


# ImageNet 均值和标准差，用于预处理
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(image_path, image_size=224):
    """
    读取图像并进行预处理（Resize + ImageNet 归一化）。

    Args:
        image_path: str, 图像文件路径。
        image_size: int, 模型输入尺寸（默认 224）。

    Returns:
        tensor: torch.Tensor [1, 3, H, W]，已归一化，在 device 上。
        orig_image: np.ndarray [H, W, 3] (BGR)，原图，用于可视化。
        (orig_h, orig_w): tuple, 原始图像尺寸。
    """
    # PIL 读取并转换为 RGB
    pil_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_img.size

    # Resize
    pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)

    # 转为 numpy [H, W, 3]，范围 [0, 255]
    img_np = np.array(pil_img, dtype=np.float32) / 255.0

    # 归一化
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

    # 读取原图用于可视化（OpenCV 格式）
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        orig_image = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)

    return img_tensor, orig_image, (orig_h, orig_w)


def load_models(config_path, prior_ckpt, diffusion_ckpt, device):
    """
    加载先验模型、去噪器、条件编码器和扩散模型。

    Args:
        config_path: str, diffusion 配置文件路径（YAML）。
        prior_ckpt: str, 先验模型检查点路径（.pth）。
        diffusion_ckpt: str, 扩散模型检查点路径（.pth）。
        device: torch.device。

    Returns:
        dict: {'prior': prior, 'denoiser': denoiser,
               'cond_encoder': cond_encoder, 'diffusion': diffusion}
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 先验模型（冻结）— embed_dim=512 matches prior training, NOT cfg['hidden_dim']=1024
    prior = HierarchicalPosePrior(
        codebook_size=cfg['num_classes'],
        embed_dim=512,
    ).to(device)
    prior.load_state_dict(torch.load(prior_ckpt, map_location=device))
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    # 去噪器
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

    # 条件编码器
    cond_encoder = MultimodalConditionEncoder(cond_dim=cfg['cond_dim']).to(device)

    # 加载扩散模型检查点
    ckpt = torch.load(diffusion_ckpt, map_location=device)
    denoiser.load_state_dict(ckpt['denoiser'])
    cond_encoder.load_state_dict(ckpt['cond_encoder'])
    denoiser.eval()
    cond_encoder.eval()

    # 扩散模型
    diffusion = DiscreteDiffusion(
        num_classes=cfg['num_classes'],
        num_timesteps=cfg['num_timesteps'],
        device=device,
    )

    return {
        'prior': prior,
        'denoiser': denoiser,
        'cond_encoder': cond_encoder,
        'diffusion': diffusion,
    }


def run_inference(image_tensor, models, diffusion, device, text_prompt=None, num_leapfrog=10):
    """
    对单张图像执行完整推理。

    Args:
        image_tensor: torch.Tensor [1, 3, H, W]。
        models: dict，包含 prior、denoiser、cond_encoder。
        diffusion: DiscreteDiffusion 实例。
        device: torch.device。
        text_prompt: str，可选，文本提示。默认 "The body of the human is unoccluded"。
        num_leapfrog: int，跳跃采样步长。

    Returns:
        np.ndarray [17, 2]，归一化到 [-1, 1] 的姿态关键点。
    """
    prior = models['prior']
    denoiser = models['denoiser']
    cond_encoder = models['cond_encoder']

    image_tensor = image_tensor.to(device)

    if text_prompt is None:
        text_prompt = "The body of the human is unoccluded"

    with torch.no_grad():
        # 多模态条件编码
        condition = cond_encoder(image_tensor, text_prompt)  # [1, cond_dim]

        # 离散扩散采样：从噪声逐步去噪到干净 token
        indices = diffusion.sample(
            denoiser=denoiser,
            condition=condition,
            prior_model=prior,
            num_leapfrog=num_leapfrog,
        )  # [1, N]

        # 解码 token 为姿态
        pose_norm = prior.decode_from_indices(indices)  # [1, 17, 2]，范围 [-1, 1]

    return pose_norm[0].cpu().numpy()  # [17, 2]


def infer_single_image(image_path, models, diffusion, device, output_path,
                        text_prompt=None, num_leapfrog=10):
    """
    对单张图像执行推理并保存结果。

    Returns:
        output_path: str，保存路径。
    """
    image_tensor, orig_image, (orig_h, orig_w) = preprocess_image(image_path)
    pose_norm = run_inference(image_tensor, models, diffusion, device,
                               text_prompt, num_leapfrog)

    # 反归一化到原始图像像素坐标
    pose_pixels = denormalize_keypoints(pose_norm, orig_w, orig_h)

    # 在原图上画骨架
    result_image = draw_skeleton(orig_image.copy(), pose_pixels)

    # 保存
    cv2.imwrite(output_path, result_image)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='OHPE 推理：从图像预测人体姿态')
    parser.add_argument('--image', required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', default='output.jpg',
                        help='输出图像路径或目录（默认 output.jpg）')
    parser.add_argument('--config', default='configs/diffusion.yaml',
                        help='Diffusion 配置文件')
    parser.add_argument('--prior_ckpt', default='./checkpoints/prior/best.pth',
                        help='先验模型检查点')
    parser.add_argument('--diffusion_ckpt', default='./checkpoints/diffusion/epoch_200.pth',
                        help='扩散模型检查点')
    parser.add_argument('--device', default='cuda',
                        help='计算设备（cuda 或 cpu）')
    parser.add_argument('--text_prompt', default=None,
                        help='文本提示（默认："The body of the human is unoccluded"）')
    parser.add_argument('--num_leapfrog', type=int, default=10,
                        help='跳跃采样步长（默认 10）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("加载模型...")
    models = load_models(args.config, args.prior_ckpt, args.diffusion_ckpt, device)
    print("模型加载完成。")

    # 判断输入是单张图还是目录
    if os.path.isdir(args.image):
        os.makedirs(args.output, exist_ok=True)
        image_files = [f for f in os.listdir(args.image)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"发现 {len(image_files)} 张图像，开始批量推理...")
        for fname in image_files:
            in_path = os.path.join(args.image, fname)
            out_path = os.path.join(args.output, fname)
            infer_single_image(in_path, models, models['diffusion'], device,
                                out_path, args.text_prompt, args.num_leapfrog)
            print(f"  已保存: {out_path}")
        print("批量推理完成。")
    else:
        out_path = infer_single_image(args.image, models, models['diffusion'], device,
                                       args.output, args.text_prompt, args.num_leapfrog)
        print(f"推理完成，结果已保存到: {out_path}")


if __name__ == '__main__':
    main()

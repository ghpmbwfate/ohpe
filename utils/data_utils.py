"""
Data loading and preprocessing utilities.
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class PoseDataset(Dataset):
    """
    Simple in-memory pose dataset for testing.
    For real training, replace with COCO/OcMotion loader using pycocotools.
    """
    def __init__(self, num_samples=1000, num_keypoints=17, image_size=256):
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.image_size = image_size

        # Generate synthetic data
        self.poses = torch.randn(num_samples, num_keypoints, 2) * 0.5
        # Normalize to [-1, 1]
        self.poses = torch.tanh(self.poses)

        # Random visibility
        self.visibility = torch.randint(0, 3, (num_samples, num_keypoints))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        from utils.text_prompt import generate_text_prompt
        pose = self.poses[idx]
        vis = self.visibility[idx].numpy()
        text = generate_text_prompt(vis)
        # Dummy image (zeros)
        image = torch.zeros(3, self.image_size, self.image_size)
        return image, pose, self.visibility[idx], text


def collate_fn(batch):
    """Collate function for DataLoader."""
    images, poses, visibilities, texts = zip(*batch)
    images = torch.stack(images)
    poses = torch.stack(poses)
    visibilities = torch.stack(visibilities)
    return images, poses, visibilities, list(texts)

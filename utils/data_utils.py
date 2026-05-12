"""
Data loading and preprocessing utilities.
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from utils.text_prompt import generate_text_prompt


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
        pose = self.poses[idx]
        vis = self.visibility[idx].numpy()
        text = generate_text_prompt(vis)
        # Dummy image (zeros)
        image = torch.zeros(3, self.image_size, self.image_size)
        return image, pose, self.visibility[idx], text


class COCOPoseDataset(Dataset):
    """
    COCO Keypoints dataset for single-person pose estimation.
    Loads images and keypoint annotations from COCO format.
    """
    def __init__(self, root_dir, ann_file, image_size=256, bbox_scale=1.25,
                 min_keypoints=5, split='train'):
        """
        Args:
            root_dir: path to COCO images folder (e.g., '/data/coco/train2017')
            ann_file: path to keypoints annotation JSON
            image_size: size to resize cropped images to
            bbox_scale: expand bbox by this factor to include more context
            min_keypoints: minimum number of visible keypoints to keep an instance
            split: 'train' or 'val'
        """
        if COCO is None:
            raise ImportError("pycocotools is required. Install with: pip install pycocotools")

        self.root_dir = root_dir
        self.image_size = image_size
        self.bbox_scale = bbox_scale
        self.min_keypoints = min_keypoints
        self.split = split

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)

        # Collect valid instances
        self.instances = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                # Filter: need keypoints and enough visible points
                if 'keypoints' not in ann:
                    continue
                keypoints = ann['keypoints']
                num_visible = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0)
                if num_visible >= self.min_keypoints and ann.get('num_keypoints', 0) >= self.min_keypoints:
                    self.instances.append({
                        'img_id': img_id,
                        'ann_id': ann['id'],
                        'bbox': ann['bbox'],  # [x, y, w, h]
                        'keypoints': keypoints,  # [x1, y1, v1, x2, y2, v2, ...]
                    })

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[COCOPoseDataset {split}] Loaded {len(self.instances)} valid person instances "
              f"from {len(self.img_ids)} images")

    def __len__(self):
        return len(self.instances)

    def _load_image(self, img_id):
        """Load image from disk."""
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        return image

    def _expand_bbox(self, bbox, img_w, img_h):
        """Expand bbox to include more context."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        new_w = w * self.bbox_scale
        new_h = h * self.bbox_scale
        new_x = max(0, cx - new_w / 2)
        new_y = max(0, cy - new_h / 2)
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        return [new_x, new_y, new_w, new_h]

    def __getitem__(self, idx):
        instance = self.instances[idx]
        img_id = instance['img_id']

        # Load image
        image = self._load_image(img_id)
        img_w, img_h = image.size

        # Expand and crop bbox
        bbox = self._expand_bbox(instance['bbox'], img_w, img_h)
        x, y, w, h = bbox
        image = image.crop((int(x), int(y), int(x + w), int(y + h)))

        # Process keypoints
        keypoints = instance['keypoints']
        pose = np.zeros((17, 2), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.int64)

        for i in range(17):
            kx = keypoints[i * 3]
            ky = keypoints[i * 3 + 1]
            kv = keypoints[i * 3 + 2]

            # Convert to crop-relative coordinates
            rel_x = (kx - x) / w
            rel_y = (ky - y) / h

            # Normalize to [-1, 1]
            pose[i, 0] = (rel_x * 2) - 1
            pose[i, 1] = (rel_y * 2) - 1

            # Map COCO visibility to our format
            # COCO: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
            # Our format: 0=not in image, 1=occluded, 2=visible
            if kv == 0:
                visibility[i] = 0
            elif kv == 1:
                visibility[i] = 1
            else:
                visibility[i] = 2

        # Generate text prompt
        text = generate_text_prompt(visibility)

        # Transform image
        image = self.transform(image)

        pose = torch.from_numpy(pose)
        visibility = torch.from_numpy(visibility)

        return image, pose, visibility, text


def collate_fn(batch):
    """Collate function for DataLoader."""
    images, poses, visibilities, texts = zip(*batch)
    images = torch.stack(images)
    poses = torch.stack(poses)
    visibilities = torch.stack(visibilities)
    return images, poses, visibilities, list(texts)

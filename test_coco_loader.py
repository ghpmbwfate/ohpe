"""
Test COCO keypoints data loader.
Auto-detects common COCO directory layouts under ./data.
Usage:
    python test_coco_loader.py                  # auto-detect ./data
    python test_coco_loader.py /path/to/coco    # specify root dir
"""
import os
import sys
from utils.data_utils import COCOPoseDataset, collate_fn
from torch.utils.data import DataLoader


def find_coco_paths(root='./data'):
    """Auto-detect COCO dataset paths from common layouts."""
    root = os.path.abspath(root)
    if not os.path.exists(root):
        return None, None

    # Common image folder names
    img_candidates = ['train2017', 'train', 'Train', 'images/train2017', 'images/train']
    # Common annotation paths
    ann_candidates = [
        'annotations/person_keypoints_train2017.json',
        'annotations/person_keypoints_train.json',
        'annotations_trainval2017/person_keypoints_train2017.json',
        'person_keypoints_train2017.json',
        'person_keypoints_train.json',
    ]

    train_root = None
    for cand in img_candidates:
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            train_root = p
            break

    train_ann = None
    for cand in ann_candidates:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            train_ann = p
            break

    return train_root, train_ann


def main():
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        root_dir = './data'

    train_root, train_ann = find_coco_paths(root_dir)

    if train_root is None or train_ann is None:
        print("[SKIP] COCO dataset not found.")
        print()
        print(f"  Searched under: {os.path.abspath(root_dir)}")
        print(f"  Found train images: {train_root}")
        print(f"  Found annotations:  {train_ann}")
        print()
        print("  Expected layout (one of):")
        print("    ./data/train2017/")
        print("    ./data/annotations/person_keypoints_train2017.json")
        print()
        print("  Or specify path:")
        print("    python test_coco_loader.py /path/to/coco")
        print()
        print("  Or pass image dir + ann file:")
        print("    python test_coco_loader.py /path/to/images /path/to/ann.json")
        return 0

    if len(sys.argv) >= 3:
        train_root = sys.argv[1]
        train_ann = sys.argv[2]

    print("=" * 60)
    print("TEST: COCO Keypoints Data Loader")
    print("=" * 60)
    print(f"Train images: {train_root}")
    print(f"Train annotations: {train_ann}")
    print()

    # Test single sample
    print("Loading dataset...")
    try:
        train_ds = COCOPoseDataset(train_root, train_ann, image_size=256, split='train')
    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"Total instances: {len(train_ds)}")
    print()

    # Inspect first sample
    print("Inspecting first sample...")
    image, pose, visibility, text = train_ds[0]
    print(f"  Image tensor shape: {image.shape}")
    print(f"  Image value range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"  Pose shape: {pose.shape}")
    print(f"  Pose value range: [{pose.min():.4f}, {pose.max():.4f}]")
    print(f"  Visibility: {visibility.tolist()}")
    print(f"  Text prompt: '{text}'")
    print()

    # Test DataLoader
    print("Testing DataLoader (batch_size=8)...")
    try:
        loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                           num_workers=2, collate_fn=collate_fn)
        images, poses, visibilities, texts = next(iter(loader))
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch poses shape: {poses.shape}")
        print(f"  Batch visibilities shape: {visibilities.shape}")
        print(f"  Batch texts count: {len(texts)}")
        print(f"  Example text: '{texts[0]}'")
    except Exception as e:
        print(f"[FAIL] DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("[PASS] COCO loader works correctly!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())

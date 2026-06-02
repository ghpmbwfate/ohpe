"""
Download CLIP model files for offline deployment using open_clip.
Run this on your local machine (with internet), then upload the output
directory to your server at: ./models_pretrained/

Usage:
    python download_model_files.py

Output:
    ./models_pretrained/clip/ViT-B-32.pt       # open_clip state_dict
    ./models_pretrained/clip/bpe_simple_vocab_16e6.txt.gz  # Tokenizer vocab
"""
import os
import urllib.request
import torch

OUTPUT_DIR = "./models_pretrained/clip"
BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


def download_bpe(output_dir):
    """Download BPE tokenizer vocabulary file."""
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "bpe_simple_vocab_16e6.txt.gz")

    if os.path.exists(dst):
        print(f"[SKIP] BPE vocab already exists: {dst}")
        return

    print(f"Downloading BPE vocab from {BPE_URL} ...")
    urllib.request.urlretrieve(BPE_URL, dst)
    print(f"[OK] Saved to {dst}")


def download_openclip_model(output_dir):
    """Download ViT-B-32 via open_clip and save as state_dict."""
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "ViT-B-32.pt")

    if os.path.exists(dst):
        print(f"[SKIP] open_clip model already exists: {dst}")
        return

    print("Downloading ViT-B-32 via open_clip (this may take a minute)...")
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    torch.save(model.state_dict(), dst)
    print(f"[OK] Saved open_clip state_dict to {dst}")


def main():
    print("=" * 60)
    print("Downloading CLIP files for offline deployment (open_clip)")
    print("=" * 60)

    download_bpe(OUTPUT_DIR)
    download_openclip_model(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("Done! Upload the following to your server:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ViT-B-32.pt")
    print(f"    bpe_simple_vocab_16e6.txt.gz")
    print()
    print("Target path on server: ./models_pretrained/clip/")
    print("=" * 60)


if __name__ == "__main__":
    main()

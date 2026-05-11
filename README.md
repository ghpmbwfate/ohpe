# OHPE Reproduce: Occluded Human Pose Estimation with Part-aware Discrete Diffusion

Reproduction of "Occluded human pose estimation based on part-aware discrete diffusion priors" (Knowledge-Based Systems, 2025).

## Project Structure

```
ohpe_reproduce/
├── configs/                    # YAML training configs
│   ├── prior_vqvae.yaml
│   └── diffusion.yaml
├── models/                     # Core model implementations
│   ├── mlp_mixer.py           # MLP-Mixer block
│   ├── vector_quantize.py     # VQ layer with EMA
│   ├── prior.py               # Hierarchical VQ-VAE prior
│   ├── diffusion.py           # Discrete diffusion (D3PM + Obs token)
│   ├── denoiser.py            # Transformer denoiser with AdaLN
│   └── multimodal_encoder.py  # Swin + CLIP fusion
├── utils/                      # Utilities
│   ├── data_utils.py          # Dataset loader
│   └── text_prompt.py         # Occlusion text generation
├── tests/                      # TDD test suite (37 tests)
├── train_prior.py             # Prior training script
├── train_diffusion.py         # Diffusion training script
└── requirements.txt
```

## Environment Setup

```bash
conda env create -f environment.yml  # or use existing ohpe env
conda activate ohpe
```

Or manually:
```bash
conda create -n ohpe python=3.10
conda activate ohpe
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_prior.py -v
pytest tests/test_diffusion.py -v
pytest tests/test_denoiser.py -v
```

## Training

### Stage 1: Hierarchical VQ-VAE Prior

```bash
python train_prior.py --config configs/prior_vqvae.yaml
```

### Stage 2: Discrete Diffusion Model

```bash
# Requires trained prior checkpoint
python train_diffusion.py --config configs/diffusion.yaml
```

## Key Implementation Details

- **MLP-Mixer**: Token-mixing + channel-mixing MLPs, drop-in replacement for attention
- **Vector Quantization**: EMA codebook update with straight-through estimator
- **Hierarchical Prior**: 4 levels (head, arms, legs, global) with separate encoders/codebooks
- **Discrete Diffusion**: Obscured-and-Replace transition matrix with Obs token
- **Denoiser**: 19-layer Transformer with AdaLN, self-attention + cross-attention
- **Multimodal Encoder**: Frozen Swin-Base + CLIP (image/text) with trainable projections

## Test Coverage

37 tests covering:
- Input/output shapes for all modules
- Gradient flow verification
- EMA codebook update correctness
- Diffusion transition matrix properties
- End-to-end training step
- Multimodal encoder backbone freezing

## Citation

```bibtex
@article{xiao2025occluded,
  title={Occluded human pose estimation based on part-aware discrete diffusion priors},
  author={Xiao, Hongyu and He, Hui and Xie, Yifan and Zheng, Yi},
  journal={Knowledge-Based Systems},
  volume={315},
  pages={113272},
  year={2025}
}
```

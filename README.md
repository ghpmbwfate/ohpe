# OHPE Reproduce: Occluded Human Pose Estimation with Part-aware Discrete Diffusion

复现论文"基于部位感知离散扩散先验的遮挡人体姿态估计"（Knowledge-Based Systems, 2025）。

## 项目结构

```
ohpe_reproduce/
├── configs/                    # YAML 训练配置
│   ├── prior_vqvae.yaml
│   └── diffusion.yaml
├── models/                     # 核心模型实现
│   ├── mlp_mixer.py           # MLP-Mixer 模块
│   ├── vector_quantize.py     # 带 EMA 的 VQ 层
│   ├── prior.py               # 层次化 VQ-VAE 先验
│   ├── diffusion.py           # 离散扩散（D3PM + Obs token）
│   ├── denoiser.py            # 带 AdaLN 的 Transformer 去噪器
│   └── multimodal_encoder.py  # Swin + CLIP 融合
├── utils/                      # 工具函数
│   ├── data_utils.py          # 数据集加载器
│   └── text_prompt.py         # 遮挡文本生成
├── tests/                      # TDD 测试套件（37 个测试）
├── train_prior.py             # 先验训练脚本
├── train_diffusion.py         # 扩散训练脚本
└── requirements.txt
```

## 环境配置

```bash
conda env create -f environment.yml  # 或使用已有的 ohpe 环境
conda activate ohpe
```

或手动配置：
```bash
conda create -n ohpe python=3.10
conda activate ohpe
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个模块测试
pytest tests/test_prior.py -v
pytest tests/test_diffusion.py -v
pytest tests/test_denoiser.py -v
```

## 训练

### 阶段一：层次化 VQ-VAE 先验

```bash
python train_prior.py --config configs/prior_vqvae.yaml
```

### 阶段二：离散扩散模型

```bash
# 需要先训练好先验的检查点
python train_diffusion.py --config configs/diffusion.yaml
```

## 关键实现细节

- **MLP-Mixer**：Token 混合 + 通道混合 MLP，可替代注意力机制
- **向量量化**：EMA 码本更新 + 直通估计器
- **层次化先验**：4 个层级（头部、手臂、腿部、全局），各有独立的编码器/码本
- **离散扩散**：遮挡替换转移矩阵 + Obs token
- **去噪器**：19 层 Transformer + AdaLN，自注意力 + 交叉注意力
- **多模态编码器**：冻结的 Swin-Base + CLIP（图像/文本）+ 可训练投影层

## 测试覆盖

37 个测试，覆盖：
- 所有模块的输入/输出形状
- 梯度流验证
- EMA 码本更新正确性
- 扩散转移矩阵性质
- 端到端训练步骤
- 多模态编码器骨干网络冻结

## 引用

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

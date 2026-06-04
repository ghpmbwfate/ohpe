# OHPE Diffusion Training Fix Log

> 记录本 session 中针对训练不稳定、OOM、NaN 等问题的所有代码修改与参数调整，便于回溯追查。

---

## 1. 环境/部署问题

### 1.1 CLIP 离线加载
- **问题**: 服务器无外网，`open_clip.create_model_and_transforms` 与 `get_tokenizer` 卡住，导致 GPU 1 完全不工作
- **修改文件**:
  - `models/multimodal_encoder.py`: 新增 `_get_tokenizer_local()`，从 `./models_pretrained/clip/bpe_simple_vocab_16e6.txt.gz` 加载 tokenizer；顶部设置 `HF_HUB_OFFLINE=1`
  - 新增 `download_model_files.py`: 在本机下载 open_clip 格式的 `ViT-B-32.pt` + BPE tokenizer
- **状态**: 已解决 ✅

### 1.2 PyTorch CUDA 驱动不兼容
- **问题**: 服务器 PyTorch 2.12.0+cu130 要求 CUDA 13.0，但驱动 570 仅支持到 CUDA 12.8，导致 `torch.cuda.is_available() == False`
- **修复命令**:
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **状态**: 已解决 ✅

---

## 2. 显存 OOM

### 2.1 `posterior_all` 张量过大
- **问题**: `models/diffusion.py:244` 的 `posterior_all = numer_all / denom_all.unsqueeze(1)` 产生 `[B=16, N≈68, V=2048, S=2049]` 张量（约 18GB），24GB 显存也不够
- **临时修复**: 完全跳过 VLB 计算，`l_vlb = 0`
- **状态**: ⚠️ 临时绕过，未根治

### 2.2 batch_size 调整
- **修改**: `configs/diffusion.yaml` batch_size 从 `16` 降到 `8`
- **命令**:
  ```bash
  sed -i 's/batch_size: 16/batch_size: 8/' configs/diffusion.yaml
  ```
- **状态**: 已应用 ✅

---

## 3. 训练 NaN / 发散

### 3.1 NaN 来源定位
- **根因 1**: VLB 计算中的 `kl.mean() * self.num_timesteps` 数值不稳定 → 已临时禁用
- **根因 2**: lr=0.0008 对 batch_size=8 的 19 层 Transformer 偏高，导致 k0 (cross entropy) 从 4.79 涨到 5.85
- **根因 3**: Denoiser 缺少保守初始化，`output_proj` 默认 Kaiming 初始化导致早期 logits 过大

### 3.2 已应用的修复

#### A. Denoiser 保守初始化 (`models/denoiser.py`)
```python
nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
nn.init.zeros_(self.output_proj.bias)
```

#### B. NaN batch 跳过 (`train_diffusion.py`)
- 将 NaN 检测从 `backward()/step()` **之后**移到 **之前**，避免 bad batch 污染 Adam 动量缓冲区
```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"[WARN] Skip bad batch: ...")
    continue
```

#### C. 降低学习率
- `configs/diffusion.yaml` lr 从 `0.0008` 降到 `0.0002`

---

## 4. 当前训练状态（待观察）

| Epoch | train_loss | recon  | k0     | val_loss | 备注 |
|-------|-----------|--------|--------|----------|------|
| 1     | 5.9351    | 1.1431 | 4.7896 | 5.8059   | 正常 |
| 2     | 6.0954    | 1.1535 | 4.9393 | 6.6477   | 开始涨 |
| 4     | 7.4198    | 1.5683 | 5.8485 | 7.4374   | 发散 |
| 6     | 7.3375    | 1.5706 | 5.7640 | 7.3383   | plateau |

- **问题**: loss 前 4 epoch 持续上升，epoch 4-6 开始 plateau 但未下降
- **可能原因**: 前几轮高 lr 已造成参数漂移；缺少 warmup；VLB 完全禁用导致约束不足

---

## 5. 待办/待验证改进

- [ ] **恢复 VLB 但分块计算**: 对 V 维度 chunk（如 256/512）循环计算 `posterior_all`，避免 OOM，恢复核心约束
- [ ] **添加 lr warmup + cosine decay**: 前 1000 steps linear warmup，之后 cosine decay 到 0
- [ ] **修正 AdamW betas**: 当前 `betas=(0.9, 0.96)`，建议改回默认 `(0.9, 0.999)`
- [ ] **recon loss 梯度**: `pred_indices = argmax` 不可导，且 decode 在 `no_grad()` 中，`l_recon` 完全不给 denoiser 梯度 — 需确认是否为设计意图
- [ ] **验证 CLIP 权重加载**: 当前 `WARNING:root:No pretrained weights loaded` 虽然已手动覆盖，但理想情况应消除 warning

---

## 6. 第二轮修复 (2026-06-03 续)

针对训练若干 epoch 后 loss 缓慢上升至 NaN (lr=0.0002, bs=4 仍发散) 的诊断与修复：

### 6.1 已应用修改

| Step | 文件 | 修改 |
|------|------|------|
| 1 | `models/diffusion.py` p_losses | `total_loss` 移除 `l_recon`（argmax+no_grad 不可导，只虚增 loss 不驱动学习），保留作监控 |
| 2 | `models/diffusion.py` | 新增 `_compute_vlb_chunked()`，chunk_size=128 分块计算 VLB；`total_loss` 加入 `0.01 * l_vlb` |
| 3 | `models/diffusion.py` p_losses | `pred_logits.clamp(-30, 30)` 防止极端值经过 softmax/CE 放大 |
| 4 | `train_diffusion.py` | 新增 `get_cosine_schedule_with_warmup()`，1000 步 warmup + cosine 到 0；每 step 后 `scheduler.step()` |
| 5 | `train_diffusion.py` | `AdamW betas (0.9, 0.96) → (0.9, 0.999)` |
| 6 | `train_diffusion.py` | 每 epoch 打印 `lr`, `grad_norm_max`（denoiser & cond_encoder 最大梯度范数） |
| - | `configs/diffusion.yaml` | 新增 `warmup_steps: 1000` |

### 6.2 设计要点

- **VLB chunked**: 论文 Eq.18 重参数化 `g_θ(k_{s-1}|k_s,y) = Σ_{k̄_0} q(k_{s-1}|k_s,k̄_0)·pred_probs[k̄_0]`。原实现一次性构建 `[B,N,V,S]` ≈ 18GB 张量。新实现对 V 维度分块累加 `model_post [B,N,S]`，峰值显存 ~`B*N*chunk*S*4B = 16*34*128*2049*4 ≈ 0.57GB`，可控。
- **vlb_weight=0.01**: 论文写权重 1.0，但实测初期 KL ≈ 4.0（远大于 CE ≈ 5），直接 1.0 会主导训练。0.01 给软约束、不抢主导，后续可调高。
- **l_recon 移出 total_loss**: 这是本轮最关键的修复。原先 `l_recon ≈ 1.5` 加进 `total_loss=7` 占 20%，但反向时是 0 梯度。导致 loss 数值波动大、误以为发散。

### 6.3 验证步骤

1. 单 batch 烟雾测试：已通过 (total_loss=7.7, vlb=4.07, gradient norm=3.25, 无 NaN)
2. 跑 5 epoch 观察 `grad_norm_max` 是否稳定 (< 50)
3. 跑 10 epoch 观察 `train_tkn` 是否单调下降
4. 若稳定，再考虑 vlb_weight 提高到 0.1 或论文的 1.0


---

## 6. 快速恢复命令备忘

```bash
# 修改 lr
sed -i 's/lr: 0.000[28]/lr: 0.0001/' configs/diffusion.yaml

# 修改 batch_size
sed -i 's/batch_size: 16/batch_size: 8/' configs/diffusion.yaml

# 重启训练
tmux attach -t diffusion_train
# Ctrl+C
CUDA_VISIBLE_DEVICES=1 python -u train_diffusion.py --config configs/diffusion.yaml --device cuda
```

---

*记录时间: 2026-06-03*

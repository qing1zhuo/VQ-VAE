# VQ-VAE + Autoregressive Priors：图像 · 音频 · 视频

基于 PyTorch 复现 **VQ-VAE**（Vector Quantized Variational Autoencoder，van den Oord et al., 2017）及其离散码本上的自回归先验模型，涵盖 **图像**、**音频** 与 **视频** 三个模态。

| 模态 | 编码器 | 量化器 | 解码器 / 先验 | 数据集 |
|------|--------|--------|---------------|--------|
| 🖼️ 图像 | CNN（下采样 4×） | `VectorQuantizerEMA` | Decoder (ConvTranspose2d) / **Gated PixelCNN** | MNIST, CIFAR-10, Mini-ImageNet |
| 🎵 音频 | Conv1D（下采样 64×） | `VectorQuantizerEMA`（复用） | **WaveNet Decoder** / **WaveNet1DPrior** (causal dilated conv) | VCTK, LibriSpeech, 任意多说话人音频 |
| 🎮 视频 | 帧级 CNN VQ-VAE（64×64 → 16×16） | `VectorQuantizerEMA` | **Transformer Prior**（action-conditioned temporal autoregressive） | Atari Breakout (DQN replay) |

---

## 1. 项目目标

### 图像分支
1. **训练 VQ-VAE**：学一个高质量的编码器 + 码本 + 解码器，使得 $\hat{x} = D(E(x))$ 与原图尽可能接近。
2. **训练先验**：固定 VQ-VAE，把训练集所有图像编码为索引图 $z$，再用 **Gated PixelCNN** 拟合类条件分布 $p(z \mid y)$。
3. **采样生成**：先用 PixelCNN 自回归采样得到一张索引图 $\hat{z}$，再用 VQ-VAE 解码器得到生成图像 $\hat{x} = D(\hat{z})$。

### 音频分支
1. **训练音频 VQ-VAE**：将原始波形 $x \in [-1,1]^T$ 下采样 64 倍后量化为离散 token 序列 $z \in \{0,\dots,K-1\}^{T/64}$，再由 WaveNet 解码器自回归重建。
2. **训练音频先验**：固定 VQ-VAE，在 token 序列上用 **WaveNet1DPrior**（causal dilated conv）拟合 $p(z \mid \text{speaker})$。
3. **语音转换（Voice Conversion）**：对源说话人波形 `encode` 得到内容 token，再用目标说话人的 speaker embedding `decode` 合成新语音。
4. **码本分析**：统计码字使用率（usage ratio）和 perplexity，检测码本坍缩。

### 视频分支（待实现）
1. **帧级 VQ-VAE**：将 Atari 单帧 (3, 64, 64) 压缩为离散索引图 (16, 16)，冻结权重。
2. **Transformer 先验**：在 rollout 帧序列上训练 **action-conditioned temporal Transformer**，拟合 $p(z_t \mid z_{<t}, a_{\le t})$。
3. **视频生成**：给定前 6 帧 context + 动作序列 → 自回归生成后 10 帧 → VQ-VAE decode 为完整 rollout。

---

## 2. 目录结构

```
VQ-VAE/
├── .vscode/
│   └── settings.json                     VS Code 配置（Conda 环境 "dl"）
│
├── Image_Ex/                             图像分支代码
│   ├── GetData.py                        图像数据加载（MNIST, CIFAR-10, MiniImageNetDataset, IndicesDataset）
│   ├── Model.py                          VQ-VAE / VectorQuantizerEMA / Encoder-Decoder / GatedPixelCNN
│   ├── Train.py                          图像 VQ-VAE / PixelCNN 训练、验证、可视化
│   ├── MNIST_recon.ipynb                 MNIST VQ-VAE 训练与重建 demo
│   ├── MNIST_gene.ipynb                  MNIST PixelCNN 先验训练与类条件生成 demo
│   ├── CIAFAR_recon.ipynb                CIFAR-10 VQ-VAE 训练与重建 demo          [注 1]
│   ├── CIFAR_gene.ipynb                  CIFAR-10 PixelCNN 先验训练与类条件生成 demo
│   ├── Mini_ImageNet_recon.ipynb         Mini-ImageNet VQ-VAE 训练与重建 demo
│   └── Mini_ImageNet_gene.ipynb          Mini-ImageNet PixelCNN 先验训练与类条件生成 demo
│
├── Audio_Ex/                             音频分支代码
│   ├── GetData.py                        音频数据管线（mu-law, AudioFolderDataset, 多说话人 DataLoader）
│   ├── Model.py                          音频模型（Encoder1D, WaveNetDecoder, AudioVQVAE, WaveNet1DPrior）
│   ├── Train.py                          音频 VQ-VAE / 先验训练、重建、语音转换、码本分析
│   └── VCTK_recon&gene.ipynb             VCTK 音频 VQ-VAE 训练与语音转换 demo
│
├── Video_Ex/                             视频分支代码（待实现）
│   ├── GetData.py                        Atari rollout 帧数据采集（AtariDQNExperienceReplay）
│   ├── Atari_recon.ipynb                 帧级 VQ-VAE 重建实验
│   └── Atari_gene.ipynb                  Transformer 先验 + 视频生成实验
│
├── checkpoints/                          已训练好的权重 / 缓存数据
│   ├── best_vqvae_mnist.pt               MNIST 上训练的 VQ-VAE（K=15, 32×32）
│   ├── best_vqvae_cifar.pt               CIFAR-10 上训练的 VQ-VAE（K=512, 32×32）
│   ├── best_vqvae_MiniImageNet.pt        Mini-ImageNet 上训练的 VQ-VAE（K=512, 128×128）
│   ├── best_vqvae_audio.pt               VCTK 上训练的音频 VQ-VAE
│   ├── best_MNIST_pixelcnn_prior.pt      MNIST 索引图上训练的 PixelCNN 先验
│   ├── best_CIFAR_pixelcnn_prior.pt      CIFAR-10 索引图上训练的 PixelCNN 先验
│   ├── best_pixelcnn_prior.pt            Mini-ImageNet 索引图上训练的 PixelCNN 先验
│   ├── MNIST_indices_dataset.pt          MNIST 预编码训练集索引
│   ├── MNIST_labels_dataset.pt           MNIST 类别标签
│   ├── CIFAR_indices_dataset.pt          CIFAR-10 预编码训练集索引
│   ├── CIFAR_labels_dataset.pt           CIFAR-10 类别标签
│   ├── vqvae_indices_dataset.pt          Mini-ImageNet 预编码训练集索引
│   ├── vqvae_labels_dataset.pt           Mini-ImageNet 类别标签
│   ├── vqvae_audio_token_dataset.pt      预编码音频 token 序列（gitignored）
│   └── vqvae_labels_dataset.pt           音频说话人标签（与 Mini-ImageNet 标签文件同名，注意区分）
│
├── data/
│   ├── cifar-10-batches-py/              torchvision 自动下载的 CIFAR-10
│   ├── miniImageNet--ravi/               Mini-ImageNet（Ravi split），含 images/ 与 train/val/test.csv
│   ├── VCTK/                             VCTK 语音语料库（按说话人分目录）
│   ├── MNIST/                            torchvision 自动下载的 MNIST
│   └── Atari/                            Atari DQN replay 数据（由 Video_Ex/GetData.py 自动下载）
│
├── picture/                               训练曲线 / 重建 / 生成可视化
│   ├── CIFAR/                            CIFAR-10 各阶段结果（VQ-VAE loss, perplexity, PixelCNN loss, 重建/生成图）
│   ├── MNIST/                            MNIST 各阶段结果
│   └── MIN/                              Mini-ImageNet 各阶段结果（VQ-VAE loss, perplexity, PixelCNN loss, 重建/生成图）
│
├── Note.md                                实验日记（各数据集完整参数配置、损失记录、结果分析与图片引用）
├── test.py                                打印 torch / CUDA / torchaudio 环境检查
├── CLAUDE.md                              Claude Code 项目指南
└── README.md
```

> **[注 1]**：文件名 "CIAFAR" 为历史笔误，实为 CIFAR-10 数据集内容。

---

## 3. 各数据集关键配置

| 数据集 | 分辨率 | IN_CH | NUM_EMBED | EMB_DIM | RES_LAYERS | mean/std 归一化 |
|--------|--------|-------|-----------|---------|------------|----------------|
| MNIST | 32×32 | 1 | **15** | 64 | 2 | mean=0, std=1 (灰度) |
| CIFAR-10 | 32×32 | 3 | **512** | **128** | **4** | ImageNet CIFAR 统计量 |
| Mini-ImageNet | 128×128 | 3 | **512** | 64 | 2 | ImageNet 通用统计量 |
| Atari Breakout | 64×64 | 3 | **512** | 64 | — (Transformer) | mean=0, std=1 (像素值 0~1) |

> MNIST 仅用 15 个码字即可实现近乎无损重建（perplexity 最高约 7.5/15），远小于 CIFAR-10 的 512 个码字。
>
> Atari 帧使用 **patch embedding + Transformer encoder** 替代纯 CNN，将 64×64 帧下采样到 16×16 潜空间。
>
> 详细实验记录与训练日志见 [Note.md](Note.md)。

---

## 4. 图像分支核心实现（`Image_Ex/Model.py`）

### 4.1 `VectorQuantizerEMA` —— EMA 更新的向量量化层

- 维护一个码本 `self._embedding`：形状 `(K, D)`，`K = num_embeddings`，`D = embedding_dim`；码本权重**不参与梯度更新**（`requires_grad_(False)`）。
- 除了码本嵌入，还维护两份 EMA 缓冲：
  - `_ema_cluster_size` (K,)：每个码字被选中的次数滑动平均
  - `_ema_w` (K, D)：被分配到每个码字的 latent 向量之和的滑动平均
  - 每步: `ema_cluster = decay * ema_cluster + (1-decay) * actual_count`
  - **Laplace 平滑**：`ema_cluster = (ema_cluster + eps) / (sum + K*eps) * sum`，防止久未使用的码字被除以 0
  - 码本权重更新：`embedding = ema_w / ema_cluster.unsqueeze(1)`
- 前向接收编码器输出 `z_e`（形状 `(B, C, H, W)`，其中 `C = D`），对每个空间位置：
  1. 计算它与码本中所有 $K$ 个码字的欧氏距离，取 $\arg\min$ 得到索引；
  2. 由独热矩阵选出对应码字得到 $z_q$；
  3. **直通梯度**：`quantized = z_e + (quantized - z_e).detach()`，让反传梯度直接从解码器穿到编码器；
  4. 计算 **承诺损失（commitment loss）** $\beta \cdot \|z_e - \text{sg}(z_q)\|^2$；
  5. 计算 **perplexity** $\exp(-\sum p_k \log p_k)$ 衡量码本利用率。

### 4.2 `Encoder` / `Decoder` / `ResidualStack`

- `Encoder`：两个 stride=2 卷积（kernel=4）把 $32\times 32$ 下采样 4 倍到 $8\times 8$，再接一个 `Conv2d(3,3)` + `ResidualStack`。
- `Decoder`：`Conv2d(1,1)` → `ResidualStack` → `ConvTranspose2d(4,2)` → `ConvTranspose2d(4,2)` 上采样回原分辨率。
- `Residual`：`ReLU → 3×3 conv → ReLU → 1×1 conv` + 跳连（恒等映射）。
- `ResidualStack`：由 `_residual_layers` 个 `Residual` 堆叠（默认 2），最后接一个 ReLU。

### 4.3 `Model` —— 完整 VQ-VAE

```
x ─► Encoder ─► 1×1 conv (hidden→D) ─► VectorQuantizerEMA ─► Decoder ─► x̂
                                          │
                                          └── commit_loss, perplexity
```

forward 返回 `(commit_loss, perplexity, x_recon)`。提供两个推断接口：

- `encode(image) -> indices (B, H, W)`：把图片压成离散索引图；
- `decode(indices) -> image (B, C, H, W)`：查码本嵌入 → permute → Decoder。

### 4.4 `GatedPixelCNN` —— 码本上的自回归先验

把 $z$ 当作一张"伪图像"做自回归建模。三层构成：

- `GatedActivation`：`tanh(x₁) ⊙ σ(x₂)` 门控激活。
- `GatedMaskConv`：**双流（vertical / horizontal）** 掩码卷积，消除原始 PixelCNN 的"盲点"：
  - 垂直流：kernel=(kernel//2+1, kernel)，关注当前行及上方所有行
  - 水平流：kernel=(1, kernel//2+1)，关注当前行左侧
  - 第一层用 `Mask-A`（屏蔽自身像素），其余层用 `Mask-B`（不屏蔽自身）
  - 每层通过 `h_embed` 注入类别条件嵌入，实现 **类条件生成**
  - 垂直流经 1×1 conv 传入水平流（`v2h = self._vertic_horiz(h_vertic)`）
  - 非第一层有残差连接：`out_h = self._horiz_resid(out_h) + x_h`
- `GatedPixelCNN` 堆叠 15 层 GatedMaskConv（首层 kernel=7、非残差，其余 kernel=3、残差），最后用两个 1×1 conv 输出 K 类 logits。
- `generate(label, shape, batch_size)`：按 raster-scan 顺序逐位置 `multinomial` 采样一张索引图。

---

## 5. 训练 / 推断流程（`Image_Ex/Train.py`、`Video_Ex/`、`Audio_Ex/Train.py`）

### 5.1 图像分支

| 函数 | 所在文件 | 作用 |
|---|---|---|
| `train(...)` | `Image_Ex/Train.py` | VQ-VAE 单 epoch：`loss = MSE(x̂, x) + commit_loss`，跟踪 perplexity。 |
| `train_pipeline(...)` | `Image_Ex/Train.py` | 完整训练循环，按总损失保留 `best_model`（deepcopy）。 |
| `valid(...)` | `Image_Ex/Train.py` | 在验证集做一次重建可视化（取前 `idx` 个 batch）。 |
| `draw(...)` | `Image_Ex/Train.py` | 上排原图、下排重建图的对比绘制（反归一化 + clamp 到 [0,1]）。 |
| `train_prior(...)` | `Image_Ex/Train.py` | PixelCNN 训练：输入 `(z, y)`，对每个空间位置做 $K$ 类 CE，按验证集 loss 保留最优权重。 |

### 5.2 视频分支（待实现）

| 函数 | 所在文件 | 作用 |
|---|---|---|
| `train_pipeline(...)` | `Video_Ex/`（对齐 Image_Ex 接口） | 帧级 VQ-VAE 训练循环。 |
| `valid(...)` / `draw(...)` | `Video_Ex/` | 帧重建对比可视化。 |
| `train_video_prior(...)` | `Video_Ex/` | Transformer 先验训练：输入 `(z_seq, action_seq)`，teacher forcing CE |
| `draw_video_strip(...)` | `Video_Ex/` | 多帧 rollout 可视化条带。 |

### 5.3 音频分支

| 函数 | 所在文件 | 作用 |
|---|---|---|
| `train_audio(...)` | `Audio_Ex/Train.py` | Audio VQ-VAE 单 epoch：`loss = CE(logits, mulaw) + commit_loss`。 |
| `train_audio_pipeline(...)` | `Audio_Ex/Train.py` | 完整训练循环，按 val_recon 保留 best，支持 `init_best_loss` 接续。 |
| `valid_audio(...)` | `Audio_Ex/Train.py` | 验证集上平均 recon loss（CE）。 |
| `train_audio_prior(...)` | `Audio_Ex/Train.py` | WaveNet1DPrior 训练：输入 `(tokens, speaker_id)`，CE 按时间步分类。 |
| `reconstruct_audio(...)` | `Audio_Ex/Train.py` | 重建：`teacher_forced=True` → 单次 forward + argmax（快）；`False` → 真自回归 encode→decode（慢）。 |
| `draw_audio(...)` | `Audio_Ex/Train.py` | 上下两行波形对比图（原波形 vs 重建波形），横轴为时间（秒）。 |
| `codebook_usage_audio(...)` | `Audio_Ex/Train.py` | 码本统计：每个码字出现次数、usage_ratio、perplexity。 |
| `voice_convert_audio(...)` | `Audio_Ex/Train.py` | 语音转换：`encode(源说话人波形)` → `decode(token, 目标说话人 id)`。 |

---

## 6. Notebook 快速上手

项目提供了 **9 个 Jupyter notebook**，覆盖三个模态完整的训练/重建/生成工作流：

| Notebook | 路径 | 内容 |
|---|---|---|
| MNIST 重建 | `Image_Ex/MNIST_recon.ipynb` | 加载/训练 MNIST VQ-VAE，可视化重建效果 |
| MNIST 生成 | `Image_Ex/MNIST_gene.ipynb` | 训练 PixelCNN 先验 → 类条件采样生成手写数字 |
| CIFAR-10 重建 | `Image_Ex/CIAFAR_recon.ipynb` | 加载/训练 CIFAR-10 VQ-VAE，可视化重建效果 |
| CIFAR-10 生成 | `Image_Ex/CIFAR_gene.ipynb` | 训练 PixelCNN 先验 → 类条件采样生成 CIFAR 图像 |
| Mini-ImageNet 重建 | `Image_Ex/Mini_ImageNet_recon.ipynb` | 加载/训练 Mini-ImageNet VQ-VAE |
| Mini-ImageNet 生成 | `Image_Ex/Mini_ImageNet_gene.ipynb` | 训练 PixelCNN 先验 → 类条件采样生成 |
| VCTK 语音转换 | `Audio_Ex/VCTK_recon&gene.ipynb` | 音频 VQ-VAE 训练 + 语音转换演示 |
| Atari 帧重建 | `Video_Ex/Atari_recon.ipynb` | Atari 帧级 VQ-VAE 重建实验（待实现） |
| Atari 视频生成 | `Video_Ex/Atari_gene.ipynb` | Transformer 先验 + action-conditioned rollout 生成（待实现） |

---

## 7. 图像分支端到端使用

### 7.1 环境

仅需常见的科研栈：

- Python ≥ 3.9
- PyTorch ≥ 2.0（带 CUDA）
- torchvision、numpy、matplotlib、Pillow、tqdm

先运行根目录下的 `test.py` 检查 CUDA：

```bash
python test.py
```

### 7.2 图像数据准备

- **MNIST**：`Image_Ex.GetData.get_MNIST(bs)` 通过 torchvision 自动下载到 `data/`。Resize 到 32×32，Normalize：`mean=[0.0]`, `std=[1.0]`。
- **CIFAR-10**：`Image_Ex.GetData.get_CIFAR(bs)` 通过 torchvision 自动下载到 `data/`。Normalize 参数：`mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`。
- **Mini-ImageNet (Ravi split)**：CSV 每行 `filename,label`，`MiniImageNetDataset` 读取并自动构建 `class_to_idx`。Normalize 参数：`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`。
- **IndicesDataset**：包装预编码的索引张量 `(N, H, W)`，直接给 PixelCNN 训练使用。

### 7.3 训练图像 VQ-VAE

打开对应 notebook：

- `Image_Ex/MNIST_recon.ipynb` —— MNIST (32×32 → 索引图 8×8, K=15)
- `Image_Ex/CIAFAR_recon.ipynb` —— CIFAR-10 (32×32 → 索引图 8×8)
- `Image_Ex/Mini_ImageNet_recon.ipynb` —— Mini-ImageNet (128×128 → 索引图 32×32)

主要超参：

```python
HIDDEN_CHANNELS   = 128
RESIDUAL_CHANNELS = 32
RESIDUAL_LAYERS   = 2      # CIFAR-10 使用 4
NUM_EMBEDDINGS    = 512    # MNIST 仅用 15
EMBEDDING_DIM     = 64     # CIFAR-10 使用 128
DECAY             = 0.99   # EMA
COMMIT_COST       = 0.25   # β
BATCH_SIZE        = 128~256
lr                = 2e-4
```

### 7.4 训练 PixelCNN 先验 + 类条件生成

打开对应的 `_gene.ipynb`：

1. **加载 VQ-VAE**：从 checkpoint 恢复模型与 cfg。
2. **编码全部训练样本**为索引图并缓存到 `checkpoints/<dataset>_indices_dataset.pt`。
3. **训练 Gated PixelCNN**：调用 `Image_Ex.Train.train_prior(...)`。
4. **采样生成**：
   ```python
   latent = prior_Model.generate(labels, shape=(8, 8), batch_size=len(labels))
   images = vqvae_model.decode(latent)
   ```

---

## 8. 视频分支详解（待实现，`Video_Ex/`）

### 8.1 两阶段流水线

与图像/音频分支一致，视频分支同样采用两阶段解耦训练：

1. **帧级 VQ-VAE**：将 Atari rollout 每帧独立压缩为离散潜在表示
2. **时序 Transformer 先验**：在潜在序列上拟合 $p(z_t \mid z_{<t}, a_{\le t})$

### 8.2 数据管线（`Video_Ex/GetData.py`）

通过 `torchrl` 的 `AtariDQNExperienceReplay` 接口获取 Atari Breakout DQN agent 的游戏 rollout 帧：

- `_collect_frames()`：遍历 Breakout/1~5 号数据集，每个随机采样 20,000 帧，转为 `(3, 64, 64)` float 张量
- `AtariFrameDataset`：将帧打包为 `(image, label)` 格式，label 置 0（占位）
- `get_AtariFrame(bs)`：按 90/10 划分 train/valid，返回 `(train_loader, valid_loader)`

### 8.3 帧级 VQ-VAE（`Video_Ex/Atari_recon.ipynb`）

将 (3, 64, 64) Atari 帧通过 patch embedding 下采样到 16×16 潜在空间：

```
(B, 3, 64, 64) → PatchEmbed(4×4) → Transformer Encoder → VQ → Decoder → (B, 3, 64, 64)
```

- **Patch size**: 4 → 潜在分辨率 16×16
- **码本大小**: K=512, D=64
- 训练/验证/可视化接口对齐 `Image_Ex/Train.py`（`train_pipeline`, `valid`, `draw`）

### 8.4 Transformer 时率先验（`Video_Ex/Atari_gene.ipynb`）

冻结 VQ-VAE 后，逐帧 encode rollout 得到索引序列 `indices (T, 16, 16)`，训练 action-conditioned temporal Transformer：

| 方法 | 说明 |
|------|------|
| `forward(indices, actions)` | teacher forcing → logits `(B, T-1, H, W, K)` |
| `generate(context, actions, n_generate, temperature)` | 前 6 帧 context + 动作 → 自回归生成后 10 帧索引图 |

推理 pipeline：`context(6帧) + actions → Transformer generate → VQ-VAE decode → rollout(16帧)`。

---

## 9. 音频分支详解

### 9.1 网络结构（`Audio_Ex/Model.py`）

#### 9.1.1 `CausalConv1d`

因果 1D 卷积 —— 只在序列左侧 pad `(kernel-1)*dilation` 个 0，右侧不做 pad，确保第 $t$ 步只看 $x_{<t}$ 和当前步。

#### 9.1.2 `Encoder1D` —— 波形下采样编码器

6 层 stride=2 卷积（kernel=4, padding=1）将 $T$ 下采样 64 倍 → `$T \to T/2 \to T/4 \to T/8 \to T/16 \to T/32 \to T/64$`，通道数逐步 `1 → H/4 → H/4 → H/2 → H/2 → H → H`。后接 `ResidualStack1D` 精炼特征，再用 1×1 conv 投影到码本维度 D。

#### 9.1.3 `WaveNetBlock` —— WaveNet 解码器基本单元

每个 block：
1. `CausalConv1d` 将 $x$ 映射到 $2\times C_{\text{res}}$（gated activation 双路）
2. 加上**局部条件**（z_q 上采样后的特征，1×1 conv 投影）和**全局条件**（speaker embedding，全连接投影）
3. Gated activation：`tanh(a) ⊙ sigmoid(b)`
4. 残差支路：`x + conv(g)`；skip 支路：`conv(g)`，各层 skip 累加到输出头

#### 9.1.4 `WaveNetDecoder` —— 自回归解码器

- 训练时 **teacher forcing**：输入 µ-law 索引序列，做 **right-shift**（起始填 silence=K//2），经过 embedding + 多层 WaveNetBlock + 累加 skip + 输出头，得到每步 256 类的 logits。
- z_q 通过 `repeat_interleave(64, dim=-1)` 简单上采样到样本级分辨率。
- 支持多个 `dilation_cycles`（每周期 dilation=1,2,4,...,2^{L-1}）。
- `generate()`：自回归逐时间步采样，$O(T)$ 次 forward。

#### 9.1.5 `AudioVQVAE` —— 完整音频 VQ-VAE

```
waveform (B,T) ─► Encoder1D (64× down) ─► VQ (unsqueeze→2D VQ→squeeze) ─► WaveNetDecoder ─► logits (B,256,T)
                       │                                                          ▲
                       └── speaker_embed ──────────────────────────────────────────┘
```

- `encode(waveform_float)` → indices `(B, T/64)` long
- `decode(indices, speaker_id)` → µ-law 波形 `(B, T)` long
- VQ 层复用图像版的 `VectorQuantizerEMA`（通过 `unsqueeze(-1)` 把 1D 适配到 2D 接口）

#### 9.1.6 `WaveNet1DPrior` —— token 序列上的自回归先验

对 VQ-VAE 编码出的 token 序列 `(B, T_lat)` 做 causal 建模：
- right-shift 输入（防止看到自己）
- 多层 WaveNetBlock（周期性 dilation，每 8 层一个周期）
- 以 speaker embedding 作为全局条件
- `generate(speaker_id, length)` 自回归采样 token 序列

### 9.2 音频数据管线（`Audio_Ex/GetData.py`）

#### 9.2.1 µ-law 编解码

- `mu_law_encode(x, 256)`：将 `[-1, 1]` 浮点波形量化为 `[0, 255]` 整数（torchaudio AF.mu_law_encoding）
- `mu_law_decode(x, 256)`：逆操作

#### 9.2.2 `AudioFolderDataset`

目录约定：`<data_root>/<speaker_name>/**/*.wav|.flac|.mp3|.ogg`

每条样本返回 `(waveform_float, waveform_mulaw, speaker_id)`，流程：
1. `_load_audio()`：优先 **soundfile**（wav/flac/ogg），避免 torchaudio 2.11+ 的 FFmpeg 依赖；mp3 回退到 torchaudio
2. 多声道 → 平均为单声道
3. 重采样到目标采样率（torchaudio Resample 缓存）
4. 随机 crop（训练）或固定从 0 开始 crop（验证/测试）到固定长度
5. µ-law 编码得到分类标签

#### 9.2.3 `get_audio_loaders`

一键构建三个 DataLoader：
- 按说话人内随机切分（train/val/test），保证每个 speaker 在三个 split 都出现
- 返回 `(train_loader, valid_loader, test_loader, speaker_map)`

#### 9.2.4 `pick_speechy_samples`

从 DataLoader 中按 RMS 能量挑出非静音样本，避免验证/测试时 crop 到纯静音。

### 9.3 音频端到端使用

#### 环境依赖

```bash
pip install torch torchaudio soundfile numpy matplotlib pillow tqdm
```

#### 音频数据准备

下载 VCTK / LibriSpeech 或自建数据集，目录结构：

```
data/VCTK/
├── p225/
│   ├── p225_001.wav
│   └── ...
├── p226/
│   └── ...
```

#### 训练音频 VQ-VAE

```python
from Audio_Ex.GetData import get_audio_loaders
from Audio_Ex.Model import AudioVQVAE
from Audio_Ex.Train import train_audio_pipeline

train_loader, val_loader, _, spk_map = get_audio_loaders(
    data_root="data/VCTK", batch_size=8, segment_samples=16000)

model = AudioVQVAE(
    num_speakers=len(spk_map), speaker_dim=64,
    mu_law_channels=256, downsample_factor=64,
    hidden_channels=128, residual_channels=32,
    residual_layers=2, num_embeddings=512,
    embedding_dim=64, decay=0.99, commitment_cost=0.25,
    wn_residual_channels=128, wn_skip_channels=256,
    wn_kernel_size=2, wn_dilation_cycles=2,
    wn_layers_per_cycle=10,
)

best_model, best_loss, history = train_audio_pipeline(
    model, train_loader, val_loader,
    epochs=100, print_epoc=10, lr=2e-4, wd=1e-6, device="cuda")
```

#### 训练音频先验（WaveNet1DPrior）

```python
# 先用 VQ-VAE encode 所有训练样本得到 tokens
# 再训练 WaveNet1DPrior
from Audio_Ex.Train import train_audio_prior
from Audio_Ex.Model import WaveNet1DPrior

prior = WaveNet1DPrior(
    vocab_size=512, dim=64, n_layers=15,
    kernel_size=3, num_speakers=len(spk_map), speaker_dim=64)

best_prior, train_loss, val_loss = train_audio_prior(
    prior, train_loader, val_loader,
    epochs=100, print_epoc=10, lr=2e-4, wd=1e-6, device="cuda")
```

#### 语音转换

```python
from Audio_Ex.GetData import mu_law_decode
from Audio_Ex.Train import voice_convert_audio

wav_mu, indices = voice_convert_audio(
    model, source_waveform, target_speaker_id, device="cuda")
wav_float = mu_law_decode(wav_mu)  # (B, T) float ∈ [-1, 1]
```

#### 码本利用率分析

```python
from Audio_Ex.Train import codebook_usage_audio

counts, usage, perp = codebook_usage_audio(
    model, train_loader, num_embeddings=512, device="cuda")
print(f"Usage: {usage:.2%}  Perplexity: {perp:.1f}/{512}")
```

---

## 10. Checkpoints 一览

### 图像 VQ-VAE 权重

| 文件 | 数据集 | 内容 |
|---|---|---|
| `best_vqvae_mnist.pt` | MNIST | VQ-VAE（K=15, 灰度 32×32） |
| `best_vqvae_cifar.pt` | CIFAR-10 | VQ-VAE（K=512, RGB 32×32） |
| `best_vqvae_MiniImageNet.pt` | Mini-ImageNet | VQ-VAE（K=512, RGB 128×128） |

### 图像 PixelCNN 先验权重

| 文件 | 数据集 | 输入维度 | 索引图分辨率 |
|---|---|---|---|
| `best_MNIST_pixelcnn_prior.pt` | MNIST | K=15 | 8×8 |
| `best_CIFAR_pixelcnn_prior.pt` | CIFAR-10 | K=512 | 8×8 |
| `best_pixelcnn_prior.pt` | Mini-ImageNet | K=512 | 32×32 |

### 预编码索引缓存

| 文件 | 数据集 |
|---|---|
| `MNIST_indices_dataset.pt` + `MNIST_labels_dataset.pt` | MNIST |
| `CIFAR_indices_dataset.pt` + `CIFAR_labels_dataset.pt` | CIFAR-10 |
| `vqvae_indices_dataset.pt` + `vqvae_labels_dataset.pt` | Mini-ImageNet |

### 音频

| 文件 | 内容 |
|---|---|
| `best_vqvae_audio.pt` | VCTK 上训练好的音频 VQ-VAE（含 Encoder1D + VQ + WaveNetDecoder + speaker_map） |
| `vqvae_audio_token_dataset.pt` | 预编码的音频 token 序列，加速 WaveNet1DPrior 训练 |
| `vqvae_labels_dataset.pt` | 音频说话人标签（与 Mini-ImageNet 标签文件同名，但内容不同） |

> **注意**：Mini-ImageNet 和音频的标签文件同名 `vqvae_labels_dataset.pt`，分别位于不同上下文中使用时需留意。

---

## 11. 设计要点回顾

- **EMA 更新码本**（含 Laplace 平滑）比 SGD 更稳定、码字利用率更高。
- **直通梯度**（straight-through estimator）是 VQ-VAE 训练的关键。
- **Perplexity**：若远小于 K → **码本坍缩**（codebook collapse），需调 commitment_cost 或重启。MNIST 仅用 15 个码字即可。
- **PixelCNN 盲点消除**：vertical + horizontal 双流 masked convolution（Mask-A 屏蔽自身，Mask-B 不屏蔽）。
- **两阶段解耦**：VQ-VAE 与先验模型完全分开训练，第二阶段 VQ-VAE 冻结。
- **音频 64 倍下采样**：6 层 stride=2 Conv1d + ResidualStack1D。
- **因果卷积**：只在序列左侧 pad，确保未来信息不泄露。
- **WaveNet 多周期 dilation**：1, 2, 4, ..., 2^{L-1} 周期重复，指数级扩大感受野。
- **Voice Conversion**：内容由 token 序列承载（来自源说话人的 encode），音色由 speaker embedding 控制（decode 时指定目标说话人）。
- **MNIST 轻量化码本**：简单灰度手写数字仅需 K=15 个码字（vs 512），perplexity ~7.5，近乎无损重建。
- **视频帧级 VQ-VAE**：帧独立编码，冻结后再训练时序先验，减少计算开销（避免端到端 video autoencoder）。
- **Action-conditioned 时序生成**：Transformer 先验以动作序列为条件，实现可控的视频 rollout 预测。

---

## 12. 参考资料

- Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. *Neural Discrete Representation Learning*. NeurIPS 2017.
- Aaron van den Oord et al. *Conditional Image Generation with PixelCNN Decoders*. NeurIPS 2016.
- Aaron van den Oord et al. *WaveNet: A Generative Model for Raw Audio*. 2016.
- 详细的 PixelCNN 中文讲解见 `test.md`（待完善）。

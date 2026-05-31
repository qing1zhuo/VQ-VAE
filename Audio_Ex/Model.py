from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,     # 码本中“词条数”K
        embedding_dim,      # 码本维度D
        commitment_cost,    # 承诺损失参数
        decay,              # 指数滑动平均历史保留系数
        eps=1e-5            # eps防止除0
    ):
        super().__init__()
        # 保存参数
        self._commitment_cost=commitment_cost
        self._num_embeddings=num_embeddings
        self._embedding_dim=embedding_dim
        self._decay = decay
        self._eps = eps

        # 嵌入层
        self._embedding=nn.Embedding(self._num_embeddings,self._embedding_dim)
        nn.init.normal_(self._embedding.weight)
        self._embedding.weight.requires_grad_(False)

        # 使用次数的滑动平均，注册为标量，不参与优化更新 (K,)
        self.register_buffer('_ema_cluster_size',torch.zeros(num_embeddings))
        # 历史向量的滑动平均 (K,D)
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

    def forward(self,img):
        # TODO 1: 形状预处理
        # BCHW->BHWC
        img=img.permute(0,2,3,1).contiguous()
        img_shape=img.shape
        # 展平为(BHW,C)/(BHW,D)
        flat_img=img.view(-1,self._embedding_dim)

        # TODO 2: 计算距离
        distances=(
            torch.sum(flat_img**2,dim=1,keepdim=True)
            +torch.sum(self._embedding.weight**2,dim=1)
            -2*(flat_img@self._embedding.weight.T)
        )

        # TODO 3: 计算索引和对应的隐向量
        # 找索引
        indices=torch.argmin(distances,dim=1).unsqueeze(1) # (N,1)

        # 找对应的向量上，使用独热编码绕一圈，方便做使用率统计
        # 做一个独热矩阵(N,K)
        # 先做全零矩阵
        encodings=torch.zeros(indices.shape[0],self._num_embeddings,device=img.device)
        # 在列上按照index给出的索引写1
        encodings.scatter_(1,indices,1)

        # 得到对应的向量(N,D)
        quantized=encodings@self._embedding.weight
        # 形状变为(B,H,W,C)
        quantized=quantized.view(img_shape)

        # TODO 4: 更新码本
        if self.training:
            # 更新K个码本向量各自的移动平均使用次数
            self._ema_cluster_size=self._ema_cluster_size*self._decay+(1-self._decay)*torch.sum(encodings,dim=0)
            # 计算当前总的使用次数
            n=torch.sum(self._ema_cluster_size.data)
            # Laplace平滑，让那些没怎么使用的向量对应的使用次数不至于为0
            self._ema_cluster_size=(self._ema_cluster_size+self._eps)/(n+self._num_embeddings*self._eps)*n

            # (K,D) 每个码本向量被分配到的latent向量进行元素级求和得到的东西
            dw=encodings.T@flat_img   
            # 更新累加向量的移动平均 
            self._ema_w=nn.Parameter(self._ema_w*self._decay+(1-self._decay)*dw)

            # 更新码本 (K,D)/(K,1)
            self._embedding.weight=nn.Parameter(self._ema_w/self._ema_cluster_size.unsqueeze(1))

        # TODO 5: 计算承诺损失
        # 把quantized从计算图剥离，避免承诺损失梯度回传到码本
        e_latene_loss=F.mse_loss(quantized.detach(),img)
        commit_loss=self._commitment_cost*e_latene_loss

        # TODO 6: 直通梯度传递
        quantized=img+(quantized-img).detach()

        # TODO 7: 计算码本平均使用率
        # (K,)
        avg_probs=encodings.mean(dim=0)
        # 等效有几个 code 在被使用
        perplexity=torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return commit_loss,perplexity,quantized.permute(0,3,1,2).contiguous(),encodings

# ============================================================================
#                  ↓↓↓  音频 VQ-VAE (van den Oord 2017 §4.2)  ↓↓↓
#
# 设计原则:
#   - 复用现有 `VectorQuantizerEMA` (通过 unsqueeze 把 1D 适配到 2D 接口),
#     不修改原代码;
#   - 1D 编码器/残差块/WaveNet 解码器/WaveNet 先验全部新增类, 命名带 1D
#     或 Audio 前缀, 避免和图像版冲突;
#   - 所有 forward 注释里都写清楚 "输入->输出" 的张量形状变化.
# ============================================================================


# ---------------------------------------------------------------------------
# 1) 因果 1D 卷积 (causal Conv1d)
#   普通 Conv1d 同时看 "过去 + 未来", 因果版只能看 "过去 + 当前".
#   做法: 在序列左侧手动 pad (kernel-1)*dilation 个 0, 右侧不 pad,
#         conv 自身 padding=0; 输出长度仍为 T.
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # 需要 pad 的总长度
        self._pad = (kernel_size - 1) * dilation
        self._conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,        # 不让 conv 自动 pad, 手动只在左侧 pad
        )

    def forward(self, x):
        # x: (B, C_in, T) -> 左侧 pad self._pad 个 0 -> (B, C_in, T + pad)
        # conv 后长度 = (T+pad) - (kernel-1)*dilation = T
        x = F.pad(x, (self._pad, 0))
        return self._conv(x)


# ---------------------------------------------------------------------------
# 2) 1D 残差块 / 残差栈
#   仿照原 Residual / ResidualStack 的结构, 把 Conv2d 换成 Conv1d.
#   结构: ReLU -> 3 conv -> ReLU -> 1 conv, + skip connection.
# ---------------------------------------------------------------------------
class Residual1D(nn.Module):
    def __init__(self, in_channels, residual_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, residual_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # x: (B, C, T)  ->  same shape
        return x + self.block(x)


class ResidualStack1D(nn.Module):
    def __init__(self, in_channels, residual_channels, out_channels, n_layers):
        super().__init__()
        self._n_layers = n_layers
        self._layers = nn.ModuleList([
            Residual1D(in_channels, residual_channels, out_channels)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


# ---------------------------------------------------------------------------
# 3) 1D Encoder: 把原始波形 (B, 1, T) 下采样 64x 成 (B, D, T/64)
#   6 个 stride=2 卷积  ->  2^6 = 64 倍下采样
#   后接一个残差栈做特征精炼, 再用 1x1 conv 投影到 VQ 码本维度 D.
# ---------------------------------------------------------------------------
class Encoder1D(nn.Module):
    def __init__(self, hidden_channels, residual_channels, residual_layers, embedding_dim):
        super().__init__()
        H = hidden_channels
        # 6 层 stride=2 卷积; kernel=4 + padding=1 + stride=2 是经典 "整除 2" 设置
        self.downs = nn.Sequential(
            nn.Conv1d(1,   H//4, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T -> T/2
            nn.Conv1d(H//4,H//4, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T/2 -> T/4
            nn.Conv1d(H//4,H//2, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T/4 -> T/8
            nn.Conv1d(H//2,H//2, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T/8 -> T/16
            nn.Conv1d(H//2,H,    kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T/16 -> T/32
            nn.Conv1d(H,   H,    kernel_size=4, stride=2, padding=1), nn.ReLU(),  # T/32 -> T/64
        )
        # 在 stride=1 维度上接 residual stack 做特征精炼
        self.res_stack = ResidualStack1D(H, residual_channels, H, residual_layers)
        # 投影到 VQ 维度 D, 准备喂给 VectorQuantizerEMA
        self.to_vq = nn.Conv1d(H, embedding_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.downs(x)        # (B, H, T/64)
        x = self.res_stack(x)    # (B, H, T/64)
        x = self.to_vq(x)        # (B, D, T/64)
        return x


# ---------------------------------------------------------------------------
# 4) WaveNet 解码器 building block
#
#   每个 block:
#     y = causal_conv(x, dilation=d)                       # (B, 2D, T)
#     y += local_cond_conv (upsampled z)                   # local conditioning
#     y += global_cond_linear (speaker embedding)          # global conditioning
#     g = tanh(y[:, :D]) * sigmoid(y[:, D:])               # gated activation
#     res_out  = x + res_conv(g)                           # 残差到下一层
#     skip_out = skip_conv(g)                              # 累加到 output head
#
#   关键张量形状全程不变 (除了通道数):
#     x         : (B, C_res, T)
#     y         : (B, 2*C_res, T)
#     g         : (B, C_res, T)
#     res_out   : (B, C_res, T)
#     skip_out  : (B, C_skip, T)
# ---------------------------------------------------------------------------
class WaveNetBlock(nn.Module):
    def __init__(
        self,
        residual_channels,    # 残差路径通道数 C_res
        skip_channels,        # skip 路径通道数 C_skip
        kernel_size,          # 因果卷积核宽 (论文 2)
        dilation,             # 当前层的 dilation
        cond_channels,        # 局部条件 (z) 的通道数 = embedding_dim
        speaker_dim,          # 全局条件 (speaker emb) 维度
    ):
        super().__init__()
        # 主因果卷积: 输出 2*C_res, 用于 gated activation 的两路
        self.causal = CausalConv1d(residual_channels, 2 * residual_channels,
                                   kernel_size=kernel_size, dilation=dilation)
        # 局部条件: 把上采样到 T 的 z (B, D, T) 投影到 (B, 2*C_res, T)
        self.cond_local  = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1)
        # 全局条件: (B, speaker_dim) -> (B, 2*C_res) -> 之后 unsqueeze 广播到 T
        self.cond_global = nn.Linear(speaker_dim, 2 * residual_channels)
        # 残差/skip 输出投影
        self.res_proj  = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(residual_channels, skip_channels,    kernel_size=1)

    def forward(self, x, cond_local_feat, cond_global_feat):
        """
        x               : (B, C_res, T)     上一层输出
        cond_local_feat : (B, D,     T)     z_q 上采样到 T 后的局部条件
        cond_global_feat: (B, speaker_dim)  speaker 嵌入 (全局条件)
        """
        # ---- 主因果卷积 ----
        y = self.causal(x)                         # (B, 2*C_res, T)
        # ---- 局部条件 ----
        y = y + self.cond_local(cond_local_feat)    # (B, 2*C_res, T)
        # ---- 全局条件: (B, 2*C_res) -> (B, 2*C_res, 1) 广播到 T ----
        gcond = self.cond_global(cond_global_feat)  # (B, 2*C_res)
        y = y + gcond.unsqueeze(-1)                 # (B, 2*C_res, T)
        # ---- 门控激活 ----
        a, b = y.chunk(2, dim=1)                    # 各自 (B, C_res, T)
        g = torch.tanh(a) * torch.sigmoid(b)        # (B, C_res, T)
        # ---- 残差 & skip ----
        res_out  = x + self.res_proj(g)             # (B, C_res, T)
        skip_out = self.skip_proj(g)                # (B, C_skip, T)
        return res_out, skip_out


# ---------------------------------------------------------------------------
# 5) WaveNet 解码器整体
#
# 训练时 ("teacher forcing"):
#   输入 x_in: μ-law 索引序列 (B, T) long
#   做 right-shift: 把第 t 步的输入设为 x[t-1] (起始处填 silence=mu/2),
#   经过 embedding + 多层 WaveNetBlock + 输出头, 得到每步 K=256 类 logits.
#   target = 原始未 shift 的 x, 用 cross_entropy 对齐.
#
# 形状流转:
#   x_in (B, T) long
#     -> embed (B, T, C_res)  ->  permute (B, C_res, T)
#   z (B, D, T/64)  上采样 64x  ->  (B, D, T)
#   speaker_id (B,) -> embed (B, speaker_dim)
#   forward 后输出 logits: (B, K, T)
# ---------------------------------------------------------------------------
class WaveNetDecoder(nn.Module):
    def __init__(
        self,
        mu_law_channels,        # 输出类别数 K (=256)
        residual_channels,      # C_res (典型 128)
        skip_channels,          # C_skip (典型 256)
        kernel_size,            # 因果 conv kernel (典型 2)
        dilation_cycles,        # 几个 "1,2,4,...,2^(L-1)" 周期 (典型 2)
        layers_per_cycle,       # 每周期层数 L (典型 10)
        cond_channels,          # = embedding_dim (z 的通道数)
        speaker_dim,            # speaker emb 维度
        upsample_factor,        # = DOWNSAMPLE_FACTOR (64); z 要被上采样这么多倍对齐到样本级
    ):
        super().__init__()
        self._upsample_factor = upsample_factor

        # μ-law 索引 -> embedding 向量 (B, T, C_res)
        self.input_embed = nn.Embedding(mu_law_channels, residual_channels)
        # 第一层只是把 embedding 当成 (B, C_res, T) 喂进 WaveNetBlock, 不再单独投影

        # WaveNet 主体: 多周期 dilated causal blocks
        self.blocks = nn.ModuleList()
        for _ in range(dilation_cycles):
            for i in range(layers_per_cycle):
                self.blocks.append(WaveNetBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    cond_channels=cond_channels,
                    speaker_dim=speaker_dim,
                ))

        # 输出头: skip_sum -> ReLU -> 1x1 -> ReLU -> 1x1 -> K 类
        self.out_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, mu_law_channels, kernel_size=1),
        )

    @staticmethod
    def _right_shift(x, fill_value):
        """teacher-forcing 用: 把 (B, T) 整体右移一位, 开头填 fill_value (silence)."""
        # x: (B, T) long
        shifted = F.pad(x, (1, 0), value=fill_value)    # (B, T+1)
        return shifted[:, :-1]                          # (B, T)

    def _upsample_cond(self, z):
        """
        z: (B, D, T/64) -> (B, D, T)
        最简单稳的做法: nearest-neighbor 重复 (相当于 64x 倍数复制).
        论文也用过 transposed conv, 二者效果差不多, NN repeat 训练更稳.
        """
        return z.repeat_interleave(self._upsample_factor, dim=-1)

    def forward(self, x_mulaw, z, speaker_emb):
        """
        x_mulaw    : (B, T) long, μ-law 量化后的 GT 波形 (训练时用 teacher forcing)
        z          : (B, D, T/64) 量化后的 latent
        speaker_emb: (B, speaker_dim) speaker embedding (已嵌入, 不是 id)

        返回 logits: (B, K, T)
        """
        B, T = x_mulaw.shape
        K = self.input_embed.num_embeddings

        # ---- 1) right-shift 让第 t 步只依赖 x[<t] ----
        x_in = self._right_shift(x_mulaw, fill_value=K // 2)   # silence (中位数桶)

        # ---- 2) embedding ----
        h = self.input_embed(x_in)                  # (B, T, C_res)
        h = h.permute(0, 2, 1).contiguous()         # (B, C_res, T)

        # ---- 3) 上采样 z 到样本级 ----
        z_up = self._upsample_cond(z)               # (B, D, T)

        # ---- 4) 堆叠 WaveNetBlock, 累加 skip ----
        skip_sum = None
        for blk in self.blocks:
            h, skip = blk(h, z_up, speaker_emb)     # h: (B,C_res,T)  skip: (B,C_skip,T)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        # ---- 5) 输出头 -> K 类 logits ----
        logits = self.out_head(skip_sum)            # (B, K, T)
        return logits

    @torch.no_grad()
    def generate(self, z, speaker_emb, init_value=None):
        """
        自回归采样一段长度为 T = z.size(-1) * upsample_factor 的 μ-law 序列.
        慢: O(T) 次 forward; 仅用于 inference / demo.
        """
        device = z.device
        B = z.size(0)
        T = z.size(-1) * self._upsample_factor
        K = self.input_embed.num_embeddings

        # 初始化序列: 全部置成 silence (mu/2)
        x = torch.full((B, T), K // 2 if init_value is None else init_value,
                       dtype=torch.long, device=device)

        # 朴素实现: 每步把当前已生成的 x 整体 forward 一次, 取第 t 步的 logits 采样
        # (论文的快速实现需要缓存中间激活; 这里追求"对的", 不追求快)
        for t in range(T):
            logits = self.forward(x, z, speaker_emb)        # (B, K, T)
            probs = F.softmax(logits[:, :, t], dim=-1)      # (B, K)
            sampled = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            x[:, t] = sampled
        return x


# ---------------------------------------------------------------------------
# 6) 完整音频 VQ-VAE 模型
#
# forward 训练接口:
#     输入: waveform_float (B, T), waveform_mulaw (B, T), speaker_id (B,)
#     输出: commit_loss (标量), perplexity (标量), logits (B, K, T)
#
# 推理接口 (类比图像版的 encode / decode):
#     encode(waveform_float)        -> token 索引 (B, T/64) long
#     decode(tokens, speaker_id, ..) -> 自回归生成 μ-law 波形 (B, T) long
# ---------------------------------------------------------------------------
class AudioVQVAE(nn.Module):
    def __init__(
        self,
        # ----- 共享 -----
        num_speakers,
        speaker_dim,
        mu_law_channels,
        downsample_factor,
        # ----- Encoder + VQ -----
        hidden_channels,
        residual_channels,
        residual_layers,
        num_embeddings,
        embedding_dim,
        decay,
        commitment_cost,
        # ----- WaveNet decoder -----
        wn_residual_channels,
        wn_skip_channels,
        wn_kernel_size,
        wn_dilation_cycles,
        wn_layers_per_cycle,
    ):
        super().__init__()
        self._mu_law_channels = mu_law_channels
        self._downsample_factor = downsample_factor

        # ---- Encoder: 波形 -> z_e ----
        self.encoder = Encoder1D(
            hidden_channels=hidden_channels,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            embedding_dim=embedding_dim,
        )

        # ---- 向量量化层: 直接复用图像版 VectorQuantizerEMA, 通过 unsqueeze 适配 ----
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        # ---- Speaker embedding (全局条件) ----
        self.speaker_embed = nn.Embedding(num_speakers, speaker_dim)

        # ---- WaveNet decoder ----
        self.decoder = WaveNetDecoder(
            mu_law_channels=mu_law_channels,
            residual_channels=wn_residual_channels,
            skip_channels=wn_skip_channels,
            kernel_size=wn_kernel_size,
            dilation_cycles=wn_dilation_cycles,
            layers_per_cycle=wn_layers_per_cycle,
            cond_channels=embedding_dim,
            speaker_dim=speaker_dim,
            upsample_factor=downsample_factor,
        )

    # ----------------------- 1D <-> 2D VQ 适配 ----------------------------
    def _vq_forward(self, z_e_1d):
        """
        z_e_1d: (B, D, T_lat)
        借用现有 2D VQ: 把 (B, D, T) -> (B, D, T, 1) 喂进去, 再 squeeze 回来.
        """
        z_e_2d = z_e_1d.unsqueeze(-1)                       # (B, D, T_lat, 1)
        commit_loss, perplexity, z_q_2d, encodings = self.vq(z_e_2d)
        z_q_1d = z_q_2d.squeeze(-1)                         # (B, D, T_lat)
        return commit_loss, perplexity, z_q_1d, encodings

    # ----------------------------- 训练 forward ---------------------------
    def forward(self, waveform_float, waveform_mulaw, speaker_id):
        """
        waveform_float: (B, T)       float, 输入给 encoder
        waveform_mulaw: (B, T)       long,  WaveNet decoder 的 teacher-forcing 输入 + cross-entropy target
        speaker_id    : (B,)         long

        返回:
            commit_loss : scalar
            perplexity  : scalar (码本利用率)
            logits      : (B, K, T)
        """
        # ---- 1) encoder ----
        x = waveform_float.unsqueeze(1)              # (B, 1, T)
        z_e = self.encoder(x)                        # (B, D, T/64)

        # ---- 2) 向量量化 (复用 VectorQuantizerEMA) ----
        commit_loss, perplexity, z_q, _ = self._vq_forward(z_e)   # z_q: (B, D, T/64)

        # ---- 3) speaker embedding ----
        spk_emb = self.speaker_embed(speaker_id)     # (B, speaker_dim)

        # ---- 4) WaveNet decoder (teacher forcing) ----
        logits = self.decoder(waveform_mulaw, z_q, spk_emb)  # (B, K, T)

        return commit_loss, perplexity, logits

    # ----------------------------- 推理接口 -------------------------------
    @torch.no_grad()
    def encode(self, waveform_float):
        """
        waveform_float: (B, T) float
        返回 indices: (B, T/64) long
        """
        x = waveform_float.unsqueeze(1)              # (B, 1, T)
        z_e = self.encoder(x)                        # (B, D, T/64)
        _, _, _, encodings = self._vq_forward(z_e)   # encodings: (B*T_lat, K) one-hot
        B = waveform_float.size(0)
        T_lat = z_e.size(-1)
        indices = torch.argmax(encodings, dim=1).view(B, T_lat)  # (B, T_lat)
        return indices

    @torch.no_grad()
    def decode(self, indices, speaker_id):
        """
        indices    : (B, T_lat)  long
        speaker_id : (B,)        long
        返回 μ-law 波形: (B, T_lat * downsample_factor) long, ∈ [0, K-1]
        """
        # 1) indices -> z_q (用 VectorQuantizerEMA 的码本嵌入查表)
        z_q = self.vq._embedding(indices)            # (B, T_lat, D)
        z_q = z_q.permute(0, 2, 1).contiguous()      # (B, D, T_lat)

        # 2) speaker emb
        spk_emb = self.speaker_embed(speaker_id)     # (B, speaker_dim)

        # 3) 自回归生成
        wav_mulaw = self.decoder.generate(z_q, spk_emb)   # (B, T) long
        return wav_mulaw


# ============================================================================
#                  ↓↓↓  音频先验: 1D causal WaveNet on tokens ↓↓↓
#
# 输入是 VQ-VAE 编码出的 token 序列 (B, T_lat), 取值 ∈ [0, K-1].
# 把它当作"1D pixel grid", 用 causal dilated 1D conv 自回归建模 p(z | speaker).
# 训练时与 PixelCNN 一致: cross_entropy(logits, tokens).
# 采样时逐位置 multinomial.
# ============================================================================
class WaveNet1DPrior(nn.Module):
    def __init__(
        self,
        vocab_size,         # K
        dim,                # 中间通道数
        n_layers,           # block 数
        kernel_size,        # 典型 3
        num_speakers,
        speaker_dim,
    ):
        super().__init__()
        self._vocab_size = vocab_size

        self.token_embed   = nn.Embedding(vocab_size,  dim)
        self.speaker_embed = nn.Embedding(num_speakers, speaker_dim)

        # 用一个特殊的"严格因果"第一层 (相当于 PixelCNN 的 Mask-A):
        # CausalConv1d 已经只看过去 (kernel-1)*dilation 个 + 当前, 我们再把
        # 第一层 kernel 的最后一个位置 (= 当前 token) 清零, 防止模型作弊看到自己.
        # 但这样实现复杂; 更省事的方案是: 输入 right-shift 一位, 后续都用普通因果卷积.
        # 这里采用 right-shift 方案 (和 WaveNetDecoder 同思路).

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(WaveNetBlock(
                residual_channels=dim,
                skip_channels=dim,
                kernel_size=kernel_size,
                dilation=2 ** (i % 8),                # 周期性 dilation
                cond_channels=speaker_dim,            # 局部条件这里没有, 用 speaker_emb 也充当占位
                speaker_dim=speaker_dim,
            ))

        self.out_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(dim, vocab_size, kernel_size=1),
        )

    def _right_shift(self, x, fill_value):
        # x: (B, T_lat) long -> 右移一位, 起始填 fill_value
        return F.pad(x, (1, 0), value=fill_value)[:, :-1]

    def forward(self, tokens, speaker_id):
        """
        tokens    : (B, T_lat) long, ∈ [0, K-1]
        speaker_id: (B,)       long
        返回 logits: (B, K, T_lat)
        """
        # 1) shift + embed
        x_in = self._right_shift(tokens, fill_value=0)
        h = self.token_embed(x_in).permute(0, 2, 1).contiguous()   # (B, dim, T_lat)

        spk_emb = self.speaker_embed(speaker_id)                   # (B, speaker_dim)

        # 局部条件这里没有真正的 z, 简单复用 spk_emb 广播 (放在 cond_local 里)
        # 形状: (B, speaker_dim, T_lat)
        spk_broadcast = spk_emb.unsqueeze(-1).expand(-1, -1, h.size(-1))

        skip_sum = None
        for blk in self.blocks:
            h, skip = blk(h, spk_broadcast, spk_emb)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        return self.out_head(skip_sum)                             # (B, K, T_lat)

    @torch.no_grad()
    def generate(self, speaker_id, length, batch_size=None):
        """
        speaker_id: (B,) long
        length    : 要生成的 token 长度 T_lat
        返回 tokens: (B, T_lat) long
        """
        self.eval()
        device = next(self.parameters()).device
        speaker_id = speaker_id.to(device)
        B = speaker_id.size(0) if batch_size is None else batch_size

        x = torch.zeros(B, length, dtype=torch.long, device=device)
        for t in range(length):
            logits = self.forward(x, speaker_id)                   # (B, K, T_lat)
            probs = F.softmax(logits[:, :, t], dim=-1)             # (B, K)
            x[:, t] = torch.multinomial(probs, 1).squeeze(-1)
        return x
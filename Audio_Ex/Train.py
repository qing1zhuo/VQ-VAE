import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import Model
import copy
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
#                       ↓↓↓  音频先验 WaveNet 训练  ↓↓↓
#
# 在阶段一缓存的 token 序列上训练 WaveNet1DPrior, 拟合 p(z | speaker).
# 与图像版 train_prior 对照:
#   - batch = (tokens, speaker_id), 不再是 (indices_2d, class_label)
#   - model.forward(tokens, speaker_id) -> logits (B, K, T_lat)
#   - loss = CrossEntropyLoss(logits, tokens)  (按每个时间步做 K 类分类)
#   - best 判定 = val_loss 最小 (与 PixelCNN 先验一致)
# ============================================================================

def train_audio_prior(
    init_model,
    train_loader, valid_loader,
    epochs, print_epoc, lr, wd, device,
):
    """
    训练 WaveNet1DPrior 若干 epoch, 返回验证集上最优的模型与损失曲线.

    参数
    ----------
    init_model   : Model.WaveNet1DPrior
        先验网络实例 (可在调用前 load_state_dict 做 continue).
    train_loader : DataLoader
        每个 batch yield (tokens, speaker_id):
          tokens     : (B, T_lat) long, ∈ [0, K-1]
          speaker_id : (B,)        long, ∈ [0, NUM_SPEAKERS-1]
    valid_loader : DataLoader
        同上, 一般 shuffle=False.
    epochs       : int
        总训练轮数.
    print_epoc   : int
        每隔多少 epoch 打印一次 train / val loss.
    lr, wd       : float
        Adam 学习率 / weight decay.
    device       : str
        "cuda:0" 或 "cpu".

    返回
    ----------
    best_model       : 验证 loss 最低那一轮的权重 (已 load 回 init_model)
    train_loss_list  : List[float], 每 epoch 训练集平均 CE
    val_loss_list    : List[float], 每 epoch 验证集平均 CE
    """
    optimizer = optim.Adam(init_model.parameters(), lr, weight_decay=wd)
    # CrossEntropyLoss: 期望 input (N, C, d1, d2, ...) 与 target (N, d1, d2, ...)
    # 这里 logits (B, K, T_lat) vs tokens (B, T_lat) -> 对每个时间步独立算 CE 再平均
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    minn_val_loss = 1e20
    init_model = init_model.to(device)
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        epoch_loss = 0.0
        samples = 0
        init_model.train()

        for tokens, spk in train_loader:
            tokens = tokens.to(device, non_blocking=True)   # (B, T_lat)
            spk    = spk.to(device,    non_blocking=True)   # (B,)

            # forward: right-shift + causal WaveNet -> (B, K, T_lat)
            logits = init_model(tokens, spk)
            loss = criterion(logits, tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = tokens.size(0)
            epoch_loss += loss.item() * n
            samples += n

        epoch_loss /= samples
        train_loss_list.append(epoch_loss)

        # ---- 验证 ----
        init_model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for tokens, spk in valid_loader:
                tokens = tokens.to(device, non_blocking=True)
                spk    = spk.to(device,    non_blocking=True)
                logits = init_model(tokens, spk)
                loss = criterion(logits, tokens)
                n = tokens.size(0)
                val_loss += loss.item() * n
                val_samples += n

        val_loss /= val_samples
        val_loss_list.append(val_loss)

        if epoch % print_epoc == 0:
            print(f"epoch: {epoch}\ntrain={epoch_loss:.4f}  val={val_loss:.4f}\n")

        if val_loss < minn_val_loss:
            minn_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in init_model.state_dict().items()}

    if best_state is not None:
        init_model.load_state_dict(best_state)
    return init_model, train_loss_list, val_loss_list


# ============================================================================
#                       ↓↓↓  音频 VQ-VAE 训练 / 验证  ↓↓↓
#
# 与图像版 (train / train_pipeline) 对照:
#   - batch 由 (waveform_float, waveform_mulaw, speaker_id) 三元组组成;
#   - 重建损失改成 cross_entropy(logits, mulaw_target);
#   - 不再需要 mean/std (没有归一化);
#   - 监控指标依然是: recon_loss + commit_loss + perplexity.
# ============================================================================

def train_audio(
    model,                       # AudioVQVAE 实例
    train_loader,                # yield (wf, mu, spk)
    epoch, print_epoc,           # 当前 epoch 号 / 打印间隔
    device,
    optimizer, criterion,        # criterion 期望是 nn.CrossEntropyLoss()
):
    """跑一个 epoch, 返回 (total_loss, recon_loss, commit_loss, perplexity) 的样本均值."""
    model.train()
    model = model.to(device)

    epoch_total, epoch_recon, epoch_commit, epoch_perp = 0.0, 0.0, 0.0, 0.0
    samples = 0

    for wf, mu, spk in train_loader:
        # 张量上 device
        wf  = wf.to(device,  non_blocking=True)   # (B, T) float
        mu  = mu.to(device,  non_blocking=True)   # (B, T) long
        spk = spk.to(device, non_blocking=True)   # (B,)   long

        # forward
        commit_loss, perplexity, logits = model(wf, mu, spk)   # logits: (B, K, T)

        # cross_entropy 在 (B, K, T) vs (B, T) 上等价于按每个时间步算分类损失
        # PyTorch CrossEntropyLoss: input (N, C, ...) target (N, ...)
        recon_loss = criterion(logits, mu)
        loss = recon_loss + commit_loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 用样本数加权累加 (batch 间 batch_size 可能不同, drop_last=True 时其实都一样)
        n = wf.size(0)
        epoch_total  += loss.item()        * n
        epoch_recon  += recon_loss.item()  * n
        epoch_commit += commit_loss.item() * n
        epoch_perp   += perplexity.item()  * n
        samples += n

    # 求样本均值
    epoch_total  /= samples
    epoch_recon  /= samples
    epoch_commit /= samples
    epoch_perp   /= samples

    if epoch % print_epoc == 0:
        print(f"[epoch {epoch:4d}]  total={epoch_total:.4f}  "
              f"recon={epoch_recon:.4f}  commit={epoch_commit:.4f}  "
              f"perp={epoch_perp:.2f}")

    return epoch_total, epoch_recon, epoch_commit, epoch_perp


@torch.no_grad()
def valid_audio(model, valid_loader, device, criterion):
    """在验证集上跑一遍, 只返回平均 recon_loss (用于挑 best)."""
    model.eval()
    model = model.to(device)
    total_recon, samples = 0.0, 0
    for wf, mu, spk in valid_loader:
        wf, mu, spk = wf.to(device), mu.to(device), spk.to(device)
        _, _, logits = model(wf, mu, spk)
        recon = criterion(logits, mu)
        n = wf.size(0)
        total_recon += recon.item() * n
        samples += n
    return total_recon / samples


def train_audio_pipeline(
    init_model,
    train_loader, valid_loader,
    epochs, print_epoc,
    lr, wd, device,
    init_best_loss=float("inf"),    # 调用方传入: 如果之前已有 best ckpt, 把它的 val_recon 传进来
):
    """
    完整训练循环.
    返回:
        best_model       : 训练过程中 val_recon 最小的那个 model (deepcopy)
        best_val_recon   : 对应的最佳 val recon loss
        history          : dict, 各项 per-epoch 列表
    """
    optimizer = optim.Adam(init_model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_total":  [],
        "train_recon":  [],
        "train_commit": [],
        "train_perp":   [],
        "val_recon":    [],
    }

    best_model     = None
    best_val_recon = init_best_loss   # 初值 = 之前 ckpt 的 val_recon (或 inf)

    for epoch in range(1, epochs + 1):
        # ---- 训练一个 epoch ----
        tr_total, tr_recon, tr_commit, tr_perp = train_audio(
            init_model, train_loader,
            epoch, print_epoc, device, optimizer, criterion,
        )
        # ---- 验证 ----
        val_recon = valid_audio(init_model, valid_loader, device, criterion)

        # ---- 记录 ----
        history["train_total"].append(tr_total)
        history["train_recon"].append(tr_recon)
        history["train_commit"].append(tr_commit)
        history["train_perp"].append(tr_perp)
        history["val_recon"].append(val_recon)

        if epoch % print_epoc == 0:
            print(f"            val_recon={val_recon:.4f}   "
                  f"best so far={best_val_recon:.4f}")

        # ---- 保留 best (按 val_recon) ----
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            best_model = copy.deepcopy(init_model)

    return best_model, best_val_recon, history


# ============================================================================
#         ↓↓↓  音频 VQ-VAE 验证 / 可视化 / 码本统计 工具集  ↓↓↓
#
# 对应图像版的:
#   - valid(...)  -> reconstruct_audio(...) + draw_audio(...)
#   - draw(...)   -> draw_audio(...)
#   - (额外)      -> codebook_usage_audio(...)
#                  -> voice_convert_audio(...)
#
# 注意: WaveNet 解码器是自回归的, 真实生成 T 个采样点要做 T 次 forward,
# 因此提供两种重建模式:
#   (A) teacher_forced=True : 一次 forward + argmax, 快, 但是"已知 GT 上下文时的预测".
#                              用于训练期间快速肉眼/听感验证.
#   (B) teacher_forced=False: 真正的 encode -> decode 自回归生成, 慢.
#                              voice conversion 时只能用 B (没有 GT).
# ============================================================================

@torch.no_grad()
def reconstruct_audio(
    model,                  # AudioVQVAE 实例
    waveform_float,         # (B, T) float
    waveform_mulaw,         # (B, T) long, ∈ [0, K-1]; teacher_forced=True 时需要; False 时可传 None
    speaker_id,             # (B,)   long
    device,
    teacher_forced=True,
):
    """
    用 model 对一个 batch 做重建, 返回 μ-law 整数张量 (B, T) long.

    teacher_forced=True:
        forward 走一次 -> logits (B, K, T) -> argmax(dim=1) -> (B, T) long.
        相当于"已经看到每个时刻之前的真实样本时, 模型会预测什么", O(1) forward.
    teacher_forced=False:
        编码到 indices (B, T/64) -> autoregressive decode (B, T).
        慢但反映了真实的"从 latent 到波形"的能力.
    """
    model.eval()
    waveform_float = waveform_float.to(device)
    speaker_id     = speaker_id.to(device)

    if teacher_forced:
        # 训练期重建: 输入 mu-law GT 作为 right-shift 上下文
        assert waveform_mulaw is not None, "teacher_forced=True 时必须提供 waveform_mulaw"
        waveform_mulaw = waveform_mulaw.to(device)
        _, _, logits = model(waveform_float, waveform_mulaw, speaker_id)  # (B, K, T)
        # 形状: (B, K, T) -> argmax(dim=1) -> (B, T)
        return logits.argmax(dim=1)
    else:
        # 真自回归: encode -> decode
        # indices: (B, T/64)
        indices = model.encode(waveform_float)
        # wav_mulaw: (B, T) long
        return model.decode(indices, speaker_id)


def draw_audio(
    wf_orig_float,      # (B, T) float, 原始波形 (用浮点画图)
    wf_recon_float,     # (B, T) float, 重建波形 (mu_law_decode 后)
    sample_rate,        # int
    n=3,                # 画前 n 条样本
    speaker_names=None, # list[str] | None, 长度需 ≥ n; 标注在子图标题里
    title="",
):
    """
    上排原始波形, 下排重建波形.
    与图像版 draw(...) 的风格保持一致 (上下两行对照).

    形状要求: wf_orig_float / wf_recon_float 都是 CPU/GPU 上的 (B, T) float Tensor;
              函数内部会 .cpu().numpy() 转成 NumPy.
    """
    n = min(n, wf_orig_float.size(0))
    # 时间轴 (秒)
    T = wf_orig_float.size(-1)
    t = np.arange(T) / sample_rate

    o = wf_orig_float[:n].detach().cpu().numpy()    # (n, T)
    r = wf_recon_float[:n].detach().cpu().numpy()   # (n, T)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 4.2), sharex=True, sharey=True)
    # axes 在 n==1 时是 (2,), 统一升一维以便循环
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # 上排: 原始
        ax = axes[0, i]
        ax.plot(t, o[i], linewidth=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"orig | spk={speaker_names[i]}" if speaker_names else "original")
        ax.grid(alpha=0.3)
        # 下排: 重建
        ax = axes[1, i]
        ax.plot(t, r[i], linewidth=0.7, color="tab:orange")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("reconstruction")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def codebook_usage_audio(
    model,                  # AudioVQVAE 实例
    data_loader,            # yield (wf, mu, spk)
    num_embeddings,         # K
    device,
    max_batches=None,       # 最多扫几个 batch (None=全部)
):
    """
    遍历 data_loader, 统计每个码字被分配到的次数, 返回:
        counts      : (K,) long, 每个码字出现的总 token 数
        usage_ratio : 标量 ∈ [0, 1], 出现至少一次的码字比例 (实际用了 / K)
        perplexity  : 标量, 整套数据下 token 分布的 perplexity (越大越接近均匀)

    perplexity 定义:
        perplexity = exp( -∑ p_k log p_k ),  p_k = counts[k] / sum(counts)
        - 若所有 token 均匀分布, perplexity = K (理想上限)
        - 若坍缩到一个 token,    perplexity = 1
    """
    model.eval()
    counts = torch.zeros(num_embeddings, dtype=torch.long, device=device)
    total_tokens = 0

    for b, batch in enumerate(data_loader):
        if max_batches is not None and b >= max_batches:
            break
        wf, _, _ = batch                            # 我们只需要 wf
        wf = wf.to(device, non_blocking=True)
        indices = model.encode(wf)                  # (B, T_lat) long
        # bincount 统计每个 id 的出现次数
        flat = indices.view(-1)                      # (B*T_lat,)
        counts += torch.bincount(flat, minlength=num_embeddings)
        total_tokens += flat.numel()

    counts_cpu = counts.cpu()
    usage_ratio = (counts_cpu > 0).float().mean().item()

    # perplexity (按总分布)
    p = counts_cpu.float() / max(1, total_tokens)
    # 避免 log(0): 给 0 概率加一个极小项
    perplexity = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))

    return counts_cpu, usage_ratio, perplexity


@torch.no_grad()
def voice_convert_audio(
    model,                  # AudioVQVAE 实例
    waveform_float,         # (B, T) float, 源说话人的波形 (作为"内容来源")
    target_speaker_id,      # (B,) long, 目标说话人 id
    device,
):
    """
    经典 voice conversion 流程:
        z       = encode(waveform_from_speaker_A)       # 把 "内容" 抠出来
        wav_B   = decode(z, speaker_id=B)               # 用 B 的音色重新合成
    返回 μ-law 整数张量 (B, T) long. 调用 mu_law_decode 拿浮点波形.

    注意: decode 是自回归, 慢. 建议在短片段 (T<=4000) 上演示.
    """
    model.eval()
    waveform_float = waveform_float.to(device)
    target_speaker_id = target_speaker_id.to(device)

    indices = model.encode(waveform_float)                       # (B, T/64) long
    wav_mu  = model.decode(indices, target_speaker_id)           # (B, T)    long
    return wav_mu, indices
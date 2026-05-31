import os
import csv
import glob
import random
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt

# ===================================================================
# 音频相关依赖
# soundfile  : 读 wav/flac/ogg, 不依赖 TorchCodec / FFmpeg (torchaudio 2.11+ 的 load 需要它们)
# torchaudio   : 重采样 (Resample) 与 μ-law 编解码 (functional)
# ===================================================================
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT



# ============================================================================
#                       ↓↓↓  音频部分 (VQ-VAE on Audio)  ↓↓↓
# ============================================================================

# ----------------------------------------------------------------------------
# 1. μ-law 编/解码工具
#
# μ-law 是电话语音里的经典"对数式"量化方案 (ITU-T G.711):
#   对绝对值小的样本给更精细的量化分辨率, 对绝对值大的样本给粗糙的分辨率,
#   总共把连续幅度压缩成 256 个等级 (mu=255).
# 公式:
#   f(x) = sign(x) * ln(1 + mu*|x|) / ln(1 + mu),  x ∈ [-1, 1]
#   再线性映射到 {0, 1, ..., mu} 的整数.
#
# 为什么用它?
#   WaveNet (论文里 VQ-VAE 的 decoder) 不直接回归连续波形, 而是把它当作
#   "256 类分类问题": 每一步预测 next sample 落在哪个量化桶里, 用 cross-entropy.
#   这样既稳定又能学到清晰的分布.
# ----------------------------------------------------------------------------

def mu_law_encode(x: torch.Tensor, mu_law_channels: int = 256) -> torch.Tensor:
    """
    把 [-1, 1] 范围的浮点波形量化为 [0, mu_law_channels-1] 的整数张量.

    参数:
        x: 任意形状的浮点张量, 取值约束在 [-1, 1] (超过会被截断)
        mu_law_channels: 量化等级数, 论文与 WaveNet 默认 256

    返回:
        与 x 形状相同的 long 张量, 取值 ∈ {0, 1, ..., mu_law_channels-1}

    实现细节:
        torchaudio.functional.mu_law_encoding 的 quantization_channels 参数
        就是 "总等级数", 不是 mu 本身. 这里我们传 256 (即 mu=255).
    """
    # 保险起见 clamp 到 [-1, 1], 避免极端值导致 log 出 NaN
    x = x.clamp(-1.0, 1.0)
    return AF.mu_law_encoding(x, quantization_channels=mu_law_channels).long()


def mu_law_decode(x: torch.Tensor, mu_law_channels: int = 256) -> torch.Tensor:
    """
    把 [0, mu_law_channels-1] 的整数张量反量化回 [-1, 1] 的浮点波形.

    参数:
        x: long 张量, 取值 ∈ {0, ..., mu_law_channels-1}
        mu_law_channels: 必须与 encode 时一致

    返回:
        同形状的 float32 张量, 范围 ≈ [-1, 1]
    """
    # mu_law_decoding 要求 float 输入
    return AF.mu_law_decoding(x.float(), quantization_channels=mu_law_channels)


# ----------------------------------------------------------------------------
# 2. 通用多说话人音频 Dataset
#
# 目录约定:
#   <data_root>/
#       <speaker_id_1>/
#           xxx.wav  /  xxx.flac  /  xxx.mp3   (允许嵌套子目录, 会递归扫)
#       <speaker_id_2>/
#           ...
#
# 适用:
#   - VCTK (data/VCTK/wav48_silence_trimmed/p225/... ➜ 把 data_root 指到
#     wav48_silence_trimmed)
#   - LibriSpeech (speaker_id/chapter_id/*.flac, data_root 指到 train-clean-100)
#   - 任何自建的"一级子目录 = 说话人"的多说话人音频集
#
# 每条样本返回 3 个张量:
#   waveform_float : (T,)  float32, ∈ [-1, 1]      <- 喂给 Encoder
#   waveform_mulaw : (T,)  long,   ∈ [0, 255]      <- WaveNet 解码器的目标标签
#   speaker_id     : int                            <- 全局条件 (broadcasting 后给 decoder)
# ----------------------------------------------------------------------------

class AudioFolderDataset(Dataset):
    """通用 "speaker / *.audio" 形式的多说话人音频数据集."""

    # 支持的音频后缀; 大小写不敏感
    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(
        self,
        data_root: str,                                 # 数据集根目录
        segment_samples: int,                           # 每条样本裁剪后的长度 (采样点数)
        sample_rate: int = 16000,                       # 目标采样率 (论文 16 kHz)
        mu_law_channels: int = 256,                     # μ-law 量化等级
        speaker_map: Optional[Dict[str, int]] = None,   # {说话人名字: 整型id}; None 时按目录名自动生成
        files: Optional[List[Tuple[str, str]]] = None,  # 预先准备好的 [(path, speaker_name), ...]; 用于复用同一份扫描结果切分 train/val/test
        random_crop: bool = True,                       # True: 训练时随机截一段; False: 验证/测试时固定从 0 开始
    ):
        super().__init__()
        self.data_root = data_root
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.mu_law_channels = mu_law_channels
        self.random_crop = random_crop

        # ---- 1) 扫描全部 (path, speaker) ----
        # 这一步只读文件名, 不读音频内容; 慢, 但每次构建 Dataset 只跑一次.
        # 为了在 train/val/test 三个 Dataset 之间复用, 上层会先扫一次, 再 split, 把切好的 list 通过 files= 传进来.
        if files is None:
            files = self._scan(data_root)
        if len(files) == 0:
            raise FileNotFoundError(
                f"在 {data_root} 下没扫到任何音频文件; 期望结构: {data_root}/<speaker>/*.wav|.flac|.mp3"
            )

        # ---- 2) 构造 speaker → int 映射 ----
        # 给 nn.Embedding 使用; 所有 split 共享同一份映射, 否则同一个 speaker 在 train 和 test 里 id 不同.
        if speaker_map is None:
            speakers = sorted({spk for _, spk in files})
            speaker_map = {spk: i for i, spk in enumerate(speakers)}

        self.files = files
        self.speaker_map = speaker_map

        # ---- 3) Resample 缓存 ----
        # 不同 wav 文件可能采样率不同 (例如 VCTK 是 48 kHz, LibriSpeech 16 kHz);
        # torchaudio.transforms.Resample 内部会预计算 sinc 卷积核, 反复重建很贵,
        # 这里按 "源采样率 → Resample 实例" 缓存, 第一次见到才创建.
        self._resamplers: Dict[int, AT.Resample] = {}

    # --------------------------- 扫描目录 ----------------------------------
    @staticmethod
    def _scan(data_root: str) -> List[Tuple[str, str]]:
        """
        递归扫描 data_root, 返回 [(audio_path, speaker_name), ...].
        约定: data_root 下的一级子目录名 = speaker_name; 子目录内可继续嵌套.
        """
        items: List[Tuple[str, str]] = []
        if not os.path.isdir(data_root):
            return items

        for spk in sorted(os.listdir(data_root)):
            spk_dir = os.path.join(data_root, spk)
            if not os.path.isdir(spk_dir):
                continue
            for ext in AudioFolderDataset.AUDIO_EXTS:
                # glob 的 ** + recursive=True 会匹配任意层级嵌套
                pattern = os.path.join(spk_dir, "**", f"*{ext}")
                for path in glob.iglob(pattern, recursive=True):
                    items.append((path, spk))
        return items

    # soundfile 原生支持的格式; .mp3 仍走 torchaudio (需 TorchCodec)
    SOUNDFILE_EXTS = (".wav", ".flac", ".ogg")

    @staticmethod
    def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
        """读音频为 (C, T) float32 与采样率. 优先 soundfile, 避免 torchaudio 2.11+ 的 FFmpeg 依赖."""
        ext = os.path.splitext(path)[1].lower()
        if ext in AudioFolderDataset.SOUNDFILE_EXTS:
            # sf.read: (T, C); torchaudio 约定 (C, T)
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T.copy())
            return waveform, int(sr)
        waveform, sr = torchaudio.load(path)
        return waveform, int(sr)

    # ------------------------ Resampler 工厂 -------------------------------
    def _get_resampler(self, orig_sr: int) -> Optional[AT.Resample]:
        """采样率匹配时返回 None (跳过); 不匹配时返回缓存好的 Resample 模块."""
        if orig_sr == self.sample_rate:
            return None
        if orig_sr not in self._resamplers:
            # AT.Resample 内部用低通滤波 + 多相滤波器实现, 比 librosa.resample 快
            self._resamplers[orig_sr] = AT.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
        return self._resamplers[orig_sr]

    # ------------------------ Dataset 接口 ---------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, spk_name = self.files[idx]

        # ---- (a) 读音频 ----
        # soundfile 把 PCM 缩放到 float32 [-1, 1], 返回 waveform: (C, T_orig), sr: int.
        waveform, sr = self._load_audio(path)

        # ---- (b) 强制单声道 ----
        # WaveNet 解码器只处理单声道; 立体声直接平均成单声道是最稳的方案
        # (相比丢弃右声道, 平均能保留更多信息)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, T_orig)

        # ---- (c) 重采样到目标采样率 ----
        # 例如 VCTK 原始 48 kHz, 我们要 16 kHz
        # Resample 模块作为 callable, 输入输出都是 (..., T)
        resampler = self._get_resampler(sr)
        if resampler is not None:
            waveform = resampler(waveform)  # (1, T_resampled)

        # ---- (d) 摘掉声道维, 变成纯 1D 信号 ----
        waveform = waveform.squeeze(0)  # (T,)

        # ---- (e) 裁剪 / 填充到固定长度 segment_samples ----
        # 训练时网络要求 batch 内所有样本等长, 否则 1D 卷积没法 batch 化.
        T = waveform.size(0)
        if T < self.segment_samples:
            # 长度不够 -> 末尾零填充
            # F.pad 对 1D 张量按 (left, right) 在最后一维填充
            waveform = F.pad(waveform, (0, self.segment_samples - T))
        elif T > self.segment_samples:
            # 长度超出 -> 随机/固定起点截取
            if self.random_crop:
                start = random.randint(0, T - self.segment_samples)
            else:
                start = 0
            waveform = waveform[start : start + self.segment_samples]
        # T == segment_samples 时直接用原值

        waveform = waveform.contiguous()  # 保证内存连续

        # ---- (f) μ-law 量化得到分类标签 ----
        # 这一份是给解码器做 cross-entropy 的"答案"
        waveform_mulaw = mu_law_encode(waveform, self.mu_law_channels)  # (T,) long

        # ---- (g) speaker id ----
        speaker_id = self.speaker_map[spk_name]  # python int

        return waveform, waveform_mulaw, speaker_id


# ----------------------------------------------------------------------------
# 3. 一键构建 train/val/test DataLoader
#
# 切分策略:
#   按"说话人内随机切"以保证 voice conversion 实验需要的"每个 speaker 在
#   train/val/test 都出现". 这样测试集的 speaker id 不会落到 nn.Embedding 没见过的位置.
# ----------------------------------------------------------------------------

def get_audio_loaders(
    data_root: str,
    batch_size: int,
    segment_samples: int = 16000,
    sample_rate: int = 16000,
    mu_law_channels: int = 256,
    train_ratio: float = 0.9,        # 训练集占比 (每个说话人各自切)
    valid_ratio: float = 0.05,       # 验证集占比 (剩余进 test)
    seed: int = 42,
    num_workers: int = 0,            # Windows 下建议先用 0; 跑通后再开
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    返回: (train_loader, valid_loader, test_loader, speaker_map)
        - speaker_map: {说话人名: 整型id} (例如 {'p225': 0, 'p226': 1, ...})

    每个 batch yield:
        waveform_float : (B, T)  float32,  T = segment_samples
        waveform_mulaw : (B, T)  long,     ∈ [0, mu_law_channels-1]
        speaker_id     : (B,)    long
    """
    # ---- 1) 扫描全部文件 ----
    all_files = AudioFolderDataset._scan(data_root)
    if len(all_files) == 0:
        raise FileNotFoundError(
            f"在 {data_root} 下没找到音频; 请检查路径或先下载数据集"
        )

    # ---- 2) 构造全局共享的 speaker_map ----
    speakers = sorted({spk for _, spk in all_files})
    speaker_map = {spk: i for i, spk in enumerate(speakers)}

    # ---- 3) 按说话人分组, 在组内随机洗牌再切分 ----
    rng = random.Random(seed)
    by_spk: Dict[str, List[Tuple[str, str]]] = {}
    for f, spk in all_files:
        by_spk.setdefault(spk, []).append((f, spk))

    train_files, valid_files, test_files = [], [], []
    for spk, items in by_spk.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio))
        # 极端情况下一个 speaker 只有 1~2 条, 保证 train 一定不空
        train_files.extend(items[:n_train])
        valid_files.extend(items[n_train : n_train + n_valid])
        test_files.extend(items[n_train + n_valid :])

    # ---- 4) 构建三个 Dataset (共享同一个 speaker_map) ----
    common = dict(
        data_root=data_root,
        segment_samples=segment_samples,
        sample_rate=sample_rate,
        mu_law_channels=mu_law_channels,
        speaker_map=speaker_map,
    )
    train_ds = AudioFolderDataset(files=train_files, random_crop=True,  **common)
    valid_ds = AudioFolderDataset(files=valid_files, random_crop=False, **common)
    test_ds  = AudioFolderDataset(files=test_files,  random_crop=False, **common)

    # ---- 5) 包成 DataLoader ----
    # drop_last 在训练时丢掉最后一个不满 batch 的样本, 避免 BN/卷积某些层报错
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )

    print(
        f"[get_audio_loaders] {len(speakers)} speakers | "
        f"train {len(train_ds)} | valid {len(valid_ds)} | test {len(test_ds)}"
    )
    return train_loader, valid_loader, test_loader, speaker_map


def waveform_rms(waveform: torch.Tensor) -> torch.Tensor:
    """
    每条样本的 RMS 能量 (用于跳过 leading silence 为主的片段).

    参数:
        waveform: (T,) 或 (B, T) float

    返回:
        标量 或 (B,) float
    """
    if waveform.dim() == 1:
        return waveform.float().pow(2).mean().sqrt()
    return waveform.float().pow(2).mean(dim=1).sqrt()


def pick_speechy_samples_from_loader(
    loader: DataLoader,
    n: int = 3,
    min_rms: float = 0.02,
    max_batches: Optional[int] = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 DataLoader 中挑出 n 条 RMS 足够大的样本, 避免 valid/test 固定 head crop 截到纯静音.

    扫描至多 max_batches 个 batch, 按 RMS 从高到低取前 n 条.
    若没有任何样本达到 min_rms, 则退化为取得分最高的 n 条并打印警告.
    """
    candidates: List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for batch_idx, (wf, mu, spk) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        rms = waveform_rms(wf)
        for i in range(wf.size(0)):
            candidates.append(
                (rms[i].item(), wf[i : i + 1], mu[i : i + 1], spk[i : i + 1])
            )

    if len(candidates) == 0:
        raise RuntimeError("DataLoader 为空, 无法挑选样本")

    candidates.sort(key=lambda item: item[0], reverse=True)
    above = [c for c in candidates if c[0] >= min_rms]
    if len(above) == 0:
        print(
            f"[pick_speechy_samples] 警告: 前 {max_batches} 个 batch 内无 RMS>={min_rms} 的样本, "
            f"改用能量最高的 {n} 条 (最高 RMS={candidates[0][0]:.4f})"
        )
        chosen = candidates[:n]
    else:
        chosen = above[:n]

    wf_out = torch.cat([c[1] for c in chosen], dim=0)
    mu_out = torch.cat([c[2] for c in chosen], dim=0)
    spk_out = torch.cat([c[3] for c in chosen], dim=0)

    rms_list = [c[0] for c in chosen]
    print(
        f"[pick_speechy_samples] 选中 {len(chosen)} 条 | "
        f"RMS={[round(r, 4) for r in rms_list]}"
    )
    return wf_out, mu_out, spk_out

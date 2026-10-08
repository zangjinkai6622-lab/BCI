"""STFA 系列模型的通用组件。

每个组件对应一个明确的 EEG/BCI 问题：
  - ChannelAttention    : 22 个脑区通道重要性不同（C3/Cz/C4 在 MI 中更关键）
  - TemporalAttention   : trial 内判别性信息集中在某些时间段
  - FeatureAttention    : 特征图通道重要性不同（SE 风格）
  - MultiScaleTemporal  : MI 信息分布在多个时间尺度上（小核短时 / 大核长时）
  - SpatialDepthwise    : 通道对应脑区，需建模脑区间空间关系
  - FrequencyBranch     : ERD/ERS 体现在 theta/alpha/mu/beta 频带，用 STFT 显式建模
  - AdaptiveFusion      : 各分支贡献应由模型自适应学习
  - GradReverse         : 梯度反转层，支撑跨被试对抗对齐（DANN 思想）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================ 梯度反转（跨被试对抗）
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    """前向恒等、反向乘以 -lambd：让特征提取器学着"骗过"被试鉴别器。"""
    return _GradReverse.apply(x, lambd)


# ============================================================ 注意力模块
# v2：残差零初始化门控 out = x*(1 + α·w)，α 初值 0。
# 训练开始时注意力等价于恒等映射（不会像乘性 sigmoid 那样随机砍掉一半信号），
# 只有注意力确实有用时 α 才被梯度推离 0；α 本身可导出，作为"自适应强度"证据。
class ChannelAttention(nn.Module):
    """通道注意力（SE 风格）：输入 (B,1,C,T)，输出逐通道加权结果。

    权重 last_weights (B, C) 可导出做脑区重要性可视化（C3/Cz/C4...）。
    """

    def __init__(self, n_channels, reduction=config.ATTN_REDUCTION):
        super().__init__()
        hidden = max(n_channels // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_channels),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.last_weights = None

    def forward(self, x):
        w = x.mean(dim=(1, 3))                       # (B, C) 全局平均
        w = torch.sigmoid(self.fc(w))                # (B, C)
        self.last_weights = w.detach()
        return x * (1.0 + self.alpha * w.unsqueeze(1).unsqueeze(-1))


class TemporalAttention(nn.Module):
    """时间注意力：对特征图时间维逐点加权，输出权重 (B,1,T') 供分析。"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):                            # x: (B, C, T)
        w = torch.sigmoid(self.conv(x))              # (B, 1, T)
        return x * (1.0 + self.alpha * w), w.detach()


class FeatureAttention(nn.Module):
    """特征图注意力（Squeeze-and-Excitation）：自适应突出重要特征通道。"""

    def __init__(self, channels, reduction=config.ATTN_REDUCTION):
        super().__init__()
        hidden = max(channels // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):                            # x: (B, C, T)
        w = self.fc(x.mean(dim=2))                   # (B, C)
        return x * (1.0 + self.alpha * w.unsqueeze(2))


# ============================================================ 时间 / 空间卷积分支
class MultiScaleTemporalConv(nn.Module):
    """多尺度时间卷积：kernel 16/32/64 并行，拼接后输出 F*len(kernels) 张特征图。

    卷积核形状 (1, k)：只在时间轴滑动，每个通道独立做时间滤波。
    """

    def __init__(self, in_ch=1, F=config.F1, kernels=config.TEMPORAL_KERNELS):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, F, kernel_size=(1, k), padding="same", bias=False),
                nn.BatchNorm2d(F),
                nn.ELU(),
            )
            for k in kernels
        ])
        self.out_ch = F * len(kernels)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class SpatialDepthwise(nn.Module):
    """空间深度卷积：核形状 (C, 1)，跨 22 个脑区通道做卷积。

    groups=in_ch：每张时间特征图独立学习一组脑区空间权重（深度可分离思想），
    参数量小且等价于让模型自适应学习"哪些脑区组合重要"。
    """

    def __init__(self, in_ch, n_channels=config.N_CHANNELS, depth=config.DEPTH):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch * depth,
                              kernel_size=(n_channels, 1),
                              groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch * depth)
        self.out_ch = in_ch * depth

    def forward(self, x):                            # (B, in_ch, C, T)
        return F.elu(self.bn(self.conv(x)))          # (B, in_ch*depth, 1, T)


# ============================================================ 频谱分支
class FrequencyBranch(nn.Module):
    """频谱分支：torch.stft 生成 log 幅度谱（可微），保留 1-40Hz 频点，

    再经 2D 卷积（频率轴 x 时间帧）提取 theta/alpha/mu/beta 节律模式。
    输出 (B, F_out)。
    """

    def __init__(self, n_channels=config.N_CHANNELS, F_out=config.F_FREQ,
                 n_fft=config.STFT_NFFT, hop=config.STFT_HOP, sfreq=config.SFREQ):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        mask = (freqs >= config.LOWCUT) & (freqs <= config.HIGHCUT)
        self.register_buffer("freq_mask", torch.tensor(mask, dtype=torch.bool))

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, F_out, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(F_out),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_ch = F_out

    def forward(self, x):                            # x: (B,1,C,T) 或 (B,C,T)
        if x.dim() == 4:
            x = x.squeeze(1)
        B, C, T = x.shape
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x.reshape(B * C, T), self.n_fft,
            hop_length=self.hop, window=window,
            return_complex=True, center=True,
        )
        mag = torch.log1p(spec.abs())                # (B*C, F_bins, T_frames)
        mag = mag[:, self.freq_mask, :]
        n_f, n_t = mag.shape[1], mag.shape[2]
        mag = mag.reshape(B, C, n_f, n_t)
        return self.conv(mag).flatten(1)             # (B, F_out)


# ============================================================ 自适应融合
class AdaptiveFusion(nn.Module):
    """多分支自适应融合：各分支向量先投影到同一维度，

    再用 softmax 可学习权重加权求和（权重随训练自动调整，可导出分析）。
    """

    def __init__(self, dims, fusion_dim=config.FUSION_DIM):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, fusion_dim) for d in dims])
        self.weights = nn.Parameter(torch.zeros(len(dims)))
        self.last_weights = None

    def forward(self, vecs):
        assert len(vecs) == len(self.proj)
        outs = [F.elu(p(v)) for p, v in zip(self.proj, vecs)]
        w = F.softmax(self.weights, dim=0)
        self.last_weights = w.detach()
        fused = sum(wi * o for wi, o in zip(w, outs))
        return fused

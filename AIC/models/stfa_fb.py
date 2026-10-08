# -*- coding: utf-8 -*-
"""STFA-FB：滤波器组增强版 STFA-Net。

核心改进：输入从 (B,1,22,T) 变为 (B,N_BANDS,22,T)，每个频带有独立的
时间卷积分支（学习该频带的 ERD/ERS 模式），空间卷积跨频带共享（学习脑区关系，
与频带无关），最后通过自适应融合各频带特征。

数据流：
  EEG (B,N_BANDS,22,T)
    -> 各频带独立时间卷积（多尺度 kernel 16/32/64）
    -> 跨频带拼接 -> 共享空间深度卷积
    -> 池化 -> 时间/特征注意力
    -> 自适应融合 -> 分类头 -> 3 类 logits

与 STFA-Net 的区别：
  - 时间卷积对每个频带独立处理，能显式提取 μ(8-12Hz)/β(13-30Hz) 的运动想象模式
  - 空间卷积和注意力模块跨频带共享（参数量增长可控）
  - 不含频谱分支（滤波器组已显式建模频带，STFT 频谱分支冗余）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import config
from models.modules import (
    ChannelAttention, TemporalAttention, FeatureAttention,
    SpatialDepthwise, AdaptiveFusion,
)

try:
    from data.dataset import N_BANDS
except ImportError:
    N_BANDS = 3


class BandTemporalConv(nn.Module):
    """频带独立时间卷积：每个频带用独立的多尺度时间卷积核提取 ERD/ERS 模式。

    输入 (B, N_BANDS, C, T) -> 对每个频带做 Conv2d(1, F, (1,k)) ->
    拼接为 (B, N_BANDS * F * len(kernels), C, T)
    """

    def __init__(self, n_bands=N_BANDS, F=config.F1, kernels=config.TEMPORAL_KERNELS):
        super().__init__()
        self.n_bands = n_bands
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, F, kernel_size=(1, k), padding="same", bias=False),
                nn.BatchNorm2d(F),
                nn.ELU(),
            )
            for k in kernels
        ])
        self.kernels = kernels
        self.out_ch = F * len(kernels)  # 单个频带的输出通道

    def forward(self, x):
        """x: (B, N_BANDS, C, T) -> (B, N_BANDS * F * len(kernels), C, T)"""
        B, nb, C, T = x.shape
        outputs = []
        for b in range(nb):
            band = x[:, b:b+1, :, :]  # (B, 1, C, T)
            band_out = torch.cat([branch(band) for branch in self.branches], dim=1)
            outputs.append(band_out)
        return torch.cat(outputs, dim=1)  # (B, nb * F * nk, C, T)


class STFAFBNet(nn.Module):
    """滤波器组增强 STFA-Net。

    与 STFA-Net 的对应关系：
    - temporal -> BandTemporalConv（各频带独立多尺度时间卷积）
    - spatial -> SpatialDepthwise（跨频带特征图共享空间卷积）
    - pool/dropout/attn -> 完全复用
    - freq -> 去除（滤波器组已显式建模频带）
    - fusion -> AdaptiveFusion（各频带特征图融合，无频谱分支）
    """

    def __init__(self, n_channels=config.N_CHANNELS, n_classes=config.N_CLASSES,
                 n_bands=N_BANDS, profile=config.DEFAULT_PROFILE):
        super().__init__()
        self.n_bands = n_bands
        p = config.get_profile(profile)
        self.profile_name = profile
        self.fusion_dim = p["fusion_dim"]
        F1 = p["F1"]
        depth = p["depth"]
        dropout = p["dropout"]

        # 通道注意力（输入端，对 22 个脑区加权；跨频带共享）
        self.channel_attn = ChannelAttention(n_channels)

        # 频带独立时间卷积
        self.temporal = BandTemporalConv(n_bands=n_bands, F=F1)
        t_ch = self.temporal.out_ch * n_bands  # 总时间特征图数

        # 空间分支（跨所有频带特征图做深度卷积）
        self.spatial = SpatialDepthwise(t_ch, n_channels, depth)
        s_ch = t_ch * depth

        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout)

        # 时间/特征注意力
        self.temporal_attn = TemporalAttention(s_ch)
        self.feature_attn = FeatureAttention(s_ch)

        # 融合 + 分类头（无频谱分支）
        self.fusion = AdaptiveFusion([s_ch], p["fusion_dim"])
        self.head = nn.Sequential(
            nn.Linear(p["fusion_dim"], p["head_hidden"]),
            nn.BatchNorm1d(p["head_hidden"]),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(p["head_hidden"], n_classes),
        )
        self.aux = {}

    def encode(self, x):
        """特征提取：输出融合特征 (B, FUSION_DIM)。

        输入 x: (B, N_BANDS, C, T)
        """
        # 通道注意力：跨频带取均值计算权重，再广播到各频带
        x_avg = x.mean(dim=1, keepdim=True)  # (B, 1, C, T) — 跨频带平均
        _ = self.channel_attn(x_avg)  # 计算权重
        self.aux["channel_weight"] = self.channel_attn.last_weights
        w = self.channel_attn.last_weights  # (B, C)
        x = x * (1.0 + self.channel_attn.alpha * w.unsqueeze(1).unsqueeze(-1))
        # x 仍为 (B, N_BANDS, C, T)

        # 频带独立时间卷积
        h = self.temporal(x)  # (B, t_ch_total, C, T)

        # 空间卷积
        h = self.spatial(h)  # (B, s_ch, 1, T')
        h = self.pool(h)  # 时间 4 倍下采样
        h = self.dropout(h).squeeze(2)  # (B, s_ch, T'')

        # 注意力
        h, tw = self.temporal_attn(h)
        self.aux["temporal_weight"] = tw
        h = self.feature_attn(h)

        # 全局平均
        v_ts = h.mean(dim=2)  # (B, s_ch)

        fused = self.fusion([v_ts])
        self.aux["fusion_weight"] = self.fusion.last_weights
        self.aux["gate_alpha"] = self.gate_values()
        return fused

    def gate_values(self):
        g = {}
        if self.channel_attn is not None:
            g["channel_attn"] = float(self.channel_attn.alpha.detach().cpu())
        if self.temporal_attn is not None:
            g["temporal_attn"] = float(self.temporal_attn.alpha.detach().cpu())
        if self.feature_attn is not None:
            g["feature_attn"] = float(self.feature_attn.alpha.detach().cpu())
        return g

    def forward(self, x, return_aux=False):
        logits = self.head(self.encode(x))
        if return_aux:
            return logits, self.aux
        return logits

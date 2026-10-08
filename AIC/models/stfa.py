"""STFA-Net：Spatio-Temporal-Frequency Adaptive Network（时空频谱自适应网络，v2）。

数据流：
  EEG (B,1,22,T)
    -> ChannelAttention（残差零初始化门控；脑区通道加权，权重可可视化）
    -> 时间分支：多尺度时间卷积 (kernel 16/32/64)；消融时退化为单尺度 (64)
    -> 空间分支：深度空间卷积 (22,1) 学习脑区关系；消融时退化为跨通道平均
    -> 时间池化 -> TemporalAttention + FeatureAttention（同为残差零初始化门控）
    -> 时空特征向量 v_ts
    -> 频谱分支：STFT 幅度谱 -> 2D 卷积 -> v_freq，经可学习 α 门控（零初始化）
    -> AdaptiveFusion（softmax 可学习权重融合两支）
    -> 分类头 -> 3 类 logits

v2 设计原则：所有"自适应"模块（三重注意力、频谱分支）均以恒等/零贡献起步，
由训练数据决定其强度，训练后 gate_values() 可导出各 α 作为自适应性证据。

use_* 开关全部对应消融实验；encode() 输出融合特征，供 STFA-DGNet 复用。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import config
from models.modules import (
    ChannelAttention, TemporalAttention, FeatureAttention,
    MultiScaleTemporalConv, SpatialDepthwise, FrequencyBranch, AdaptiveFusion,
)


class STFANet(nn.Module):
    def __init__(self, n_channels=config.N_CHANNELS, n_samples=config.WIN_SAMPLES,
                 n_classes=config.N_CLASSES,
                 use_multiscale=True, use_spatial=True,
                 use_frequency=True, use_attention=True,
                 dropout=None, profile=config.DEFAULT_PROFILE):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_attention = use_attention

        # 容量档（standard / light）：通道数、融合维度、dropout 全部由配置派生
        p = config.get_profile(profile)
        self.profile_name = profile
        self.fusion_dim = p["fusion_dim"]
        F1, depth = p["F1"], p["depth"]
        dropout = p["dropout"] if dropout is None else dropout

        # ---- 通道注意力（输入端，对 22 个脑区加权）----
        self.channel_attn = ChannelAttention(n_channels) if use_attention else None

        # ---- 时间分支 ----
        if use_multiscale:
            self.temporal = MultiScaleTemporalConv(
                in_ch=1, F=F1, kernels=config.TEMPORAL_KERNELS)
            t_ch = F1 * len(config.TEMPORAL_KERNELS)
        else:
            # 消融对照：单一时间尺度（与 EEGNet 同核长 64）
            self.temporal = nn.Sequential(
                nn.Conv2d(1, F1, kernel_size=(1, config.EEGNET_KERNEL),
                          padding="same", bias=False),
                nn.BatchNorm2d(F1),
                nn.ELU(),
            )
            t_ch = F1

        # ---- 空间分支 ----
        if use_spatial:
            self.spatial = SpatialDepthwise(t_ch, n_channels, depth)
            s_ch = t_ch * depth
        else:
            # 消融对照：不做脑区间卷积，跨通道平均（各通道独立）
            self.spatial = None
            s_ch = t_ch

        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout)

        # ---- 特征图上的时间/特征注意力 ----
        self.temporal_attn = TemporalAttention(s_ch) if use_attention else None
        self.feature_attn = FeatureAttention(s_ch) if use_attention else None

        # ---- 频谱分支 ----
        self.freq = FrequencyBranch(n_channels, F_out=p["F_freq"]) if use_frequency else None
        # 频谱分支自适应门控：初值 0（起步不注入频谱信息），
        # 由训练数据决定该分支贡献大小；训练后 α 值可导出作为自适应证据。
        self.freq_alpha = nn.Parameter(torch.zeros(1)) if use_frequency else None

        # ---- 自适应融合 + 分类头 ----
        branch_dims = [s_ch] + ([p["F_freq"]] if use_frequency else [])
        self.fusion = AdaptiveFusion(branch_dims, p["fusion_dim"])
        if use_frequency:
            # α=0 时频谱向量为 0；强制该分支投影偏置为 0，保证融合中严格无贡献，
            # 避免 softmax 等权融合把随机偏置噪声注入主分支。
            nn.init.zeros_(self.fusion.proj[-1].bias)
        self.head = nn.Sequential(
            nn.Linear(p["fusion_dim"], p["head_hidden"]),
            nn.BatchNorm1d(p["head_hidden"]),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(p["head_hidden"], n_classes),
        )
        self.aux = {}

    def encode(self, x):
        """特征提取：输出融合特征 (B, FUSION_DIM)，并把注意力/融合权重存入 self.aux。"""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.channel_attn is not None:
            x = self.channel_attn(x)
            self.aux["channel_weight"] = self.channel_attn.last_weights

        h = self.temporal(x)                          # (B, t_ch, C, T)
        if self.spatial is not None:
            h = self.spatial(h)                       # (B, s_ch, 1, T')
        else:
            h = h.mean(dim=2, keepdim=True)           # 跨脑区平均
        h = self.pool(h)                              # 时间 4 倍下采样
        h = self.dropout(h).squeeze(2)                # (B, s_ch, T'')

        if self.temporal_attn is not None:
            h, tw = self.temporal_attn(h)
            self.aux["temporal_weight"] = tw
        if self.feature_attn is not None:
            h = self.feature_attn(h)

        v_ts = h.mean(dim=2)                          # 全局平均 -> (B, s_ch)

        vecs = [v_ts]
        if self.freq is not None:
            vecs.append(self.freq_alpha * self.freq(x))   # 自适应门控的频谱特征
        fused = self.fusion(vecs)
        self.aux["fusion_weight"] = self.fusion.last_weights
        self.aux["gate_alpha"] = self.gate_values()
        return fused

    def gate_values(self):
        """导出 v2 全部自适应门控强度（训练后）：α≈0 表示该模块被自适应抑制。"""
        g = {}
        if self.channel_attn is not None:
            g["channel_attn"] = float(self.channel_attn.alpha.detach().cpu())
        if self.temporal_attn is not None:
            g["temporal_attn"] = float(self.temporal_attn.alpha.detach().cpu())
        if self.feature_attn is not None:
            g["feature_attn"] = float(self.feature_attn.alpha.detach().cpu())
        if self.freq_alpha is not None:
            g["frequency"] = float(self.freq_alpha.detach().cpu())
        return g

    def forward(self, x, return_aux=False):
        logits = self.head(self.encode(x))
        if return_aux:
            return logits, self.aux
        return logits

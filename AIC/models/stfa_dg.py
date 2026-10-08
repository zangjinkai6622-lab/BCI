"""STFA-DGNet：STFA-Net + 对抗式跨被试特征对齐（Domain-Adversarial）。

动机：EEG 个体差异大，同一运动意图在不同被试上的信号分布不同，
      导致"见不到的被试"上准确率明显下降。
做法：
  Encoder（STFA-Net 编码器）-> 任务特征 z
    ├─ 任务分类头：z -> 左手/右手/双脚（要准）
    └─ 被试鉴别头：z -> 被试 id（经梯度反转层 GRL）
  GRL 前向恒等、反向把梯度取负：鉴别器越想识别"这是谁"，
  编码器就越学着把"被试身份信息"从 z 中抹掉，
  从而得到 subject-invariant（跨被试不变）的运动意图特征。
lambda 按训练进度从 0 warmup 到 DG_LAMBDA_MAX（DANN 标准调度）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import config
from models.modules import grad_reverse
from models.stfa import STFANet


class STFADGNet(nn.Module):
    def __init__(self, n_subjects,
                 n_channels=config.N_CHANNELS, n_samples=config.WIN_SAMPLES,
                 n_classes=config.N_CLASSES,
                 use_multiscale=True, use_spatial=True,
                 use_frequency=True, use_attention=True,
                 dropout=None, profile=config.DEFAULT_PROFILE):
        super().__init__()
        p = config.get_profile(profile)
        self.encoder = STFANet(
            n_channels=n_channels, n_samples=n_samples, n_classes=n_classes,
            use_multiscale=use_multiscale, use_spatial=use_spatial,
            use_frequency=use_frequency, use_attention=use_attention,
            dropout=dropout, profile=profile,
        )
        self.subject_head = nn.Sequential(
            nn.Linear(p["fusion_dim"], p["dg_hidden"]),
            nn.ReLU(),
            nn.Dropout(p["dropout"] if dropout is None else dropout),
            nn.Linear(p["dg_hidden"], n_subjects),
        )

    def forward(self, x, lambd=1.0, return_aux=False):
        feat = self.encoder.encode(x)                 # 跨被试不变特征
        task_logits = self.encoder.head(feat)
        subj_logits = self.subject_head(grad_reverse(feat, lambd))
        if return_aux:
            return task_logits, subj_logits, self.encoder.aux
        return task_logits, subj_logits

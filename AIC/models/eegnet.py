"""EEGNet 深度学习基线（Lawhern et al., J. Neural Eng. 2018）。

结构：Temporal Conv -> Depthwise Spatial Conv -> Separable Conv -> Classifier
作用：作为深度基线，回答"端到端深度学习相对手工特征+传统ML有没有提升"。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import config


class EEGNet(nn.Module):
    def __init__(self, n_channels=config.N_CHANNELS, n_samples=config.WIN_SAMPLES,
                 n_classes=config.N_CLASSES, F1=config.F1, depth=config.DEPTH,
                 kernel=config.EEGNET_KERNEL, dropout=config.DROPOUT):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * depth, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * depth),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * depth, F1 * depth, kernel_size=(1, 16),
                      padding="same", groups=F1 * depth, bias=False),
            nn.Conv2d(F1 * depth, F1 * depth, kernel_size=1, bias=False),
            nn.BatchNorm2d(F1 * depth),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.flatten = nn.Flatten()

        # 延迟确定展平维度（与输入长度解耦，避免手写魔法数字）；eval 模式避免污染 BN 统计量
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat_dim = self.flatten(self.separable(self.spatial(self.temporal(dummy)))).shape[1]
        if was_training:
            self.train()
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_aux=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.flatten(self.separable(self.spatial(self.temporal(x))))
        logits = self.classifier(h)
        if return_aux:
            return logits, {}
        return logits

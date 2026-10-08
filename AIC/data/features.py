"""传统机器学习基线的手工特征（与 EEG Explorer 同源，作用于 epoch 张量）。

每个 epoch、每个通道提取：
  - theta/alpha/mu/beta 频带对数能量 + 相对能量（Welch PSD）
  - Hjorth 参数（activity / mobility / complexity）
  - 谱熵（1-40Hz 归一化）
  - 时域统计（mean / std / skew / kurtosis）
特征提取本身无拟合操作，可安全作用于全量数据，划分只用于训练与评估。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

import config


def _hjorth(x):
    """Hjorth 三参数：Activity(方差) / Mobility(迁移率) / Complexity(复杂度)。"""
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = x.var() + 1e-12
    activity = var_x
    mobility = np.sqrt(dx.var() / var_x)
    complexity = np.sqrt(ddx.var() / (dx.var() + 1e-12)) / (mobility + 1e-12)
    return activity, mobility, complexity


def _spectral_entropy(psd):
    """归一化谱熵（0-1），值越大频谱越平坦、节律性越弱。"""
    p = psd / (psd.sum() + 1e-12)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log2(p)).sum() / np.log2(len(p)))


def _channel_features(x):
    freqs, psd = welch(x, fs=config.SFREQ, nperseg=min(256, x.shape[-1]))

    band_pow = {}
    for name, (lo, hi) in config.FREQ_BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        band_pow[name] = float(np.trapezoid(psd[m], freqs[m]) + 1e-12)
    total_pow = sum(band_pow.values())

    feats = []
    feats += [np.log(v) for v in band_pow.values()]                    # 对数频带能量 x4
    feats += [v / total_pow for v in band_pow.values()]                # 相对频带能量 x4
    feats += list(_hjorth(x))                                          # Hjorth x3
    band_mask = (freqs >= config.LOWCUT) & (freqs <= config.HIGHCUT)
    feats.append(_spectral_entropy(psd[band_mask]))                    # 谱熵 x1
    feats += [float(x.mean()), float(x.std()),                         # 时域统计 x4
              float(skew(x)), float(kurtosis(x))]
    return feats


def extract_epoch_features(X):
    """X: (N, 22, T) -> 特征矩阵 (N, 22 * 16) float32。

    每通道 16 个特征：4 频带对数能量 + 4 相对能量 + 3 Hjorth + 1 谱熵 + 4 时域统计。
    """
    n, c, _ = X.shape
    out = np.zeros((n, c * 16), dtype=np.float32)
    for i in range(n):
        row = []
        for ch in range(c):
            row += _channel_features(X[i, ch])
        out[i] = np.asarray(row, dtype=np.float32)
    # 防御：替换 inf/NaN（极端信号下高阶统计量可能溢出）
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

# -*- coding: utf-8 -*-
"""BCI Competition IV-2b 数据加载器（与 dataset.py 2a 口径一致，但参数独立）。

2b 与 2a 的关键差异：
  - 通道：3 EEG（C3/Cz/C4）+ 3 EOG，命名前缀 'EEG:'（2a 用 'EEG-'）
  - 类别：2 类（左手 769 / 右手 770），无双脚/舌头
  - 场次：每被试 5 场（T1/T2/T3 训练，T4/T5 评测），GDF 内嵌事件即含标签
  - 采样率/窗口/滤波与 2a 完全相同：250Hz, [0.5,4.5]s, 1-40Hz 带通 + 50Hz 陷波

跨数据集迁移的关键：C3/Cz/C4 与 2a 索引 7/9/11 一一对应，
时空频特征（mu/beta ERD/ERS）在两数据集间可迁移。
"""
import sys
import re
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import butter, filtfilt, iirnotch

import config

warnings.filterwarnings("ignore")

# 2b 专用常量（不污染 2a config）
BCI2B_N_CHANNELS = 3
BCI2B_N_CLASSES = 2
BCI2B_SFREQ = 250
BCI2B_TMIN, BCI2B_TMAX = 0.5, 4.5
BCI2B_WIN_SAMPLES = int(round((BCI2B_TMAX - BCI2B_TMIN) * BCI2B_SFREQ))   # 1000
BCI2B_LOWCUT, BCI2B_HIGHCUT = 1.0, 40.0
BCI2B_NOTCH_FREQ, BCI2B_NOTCH_Q = 50.0, 30
BCI2B_FILTER_ORDER = 4
BCI2B_EVENT_TO_LABEL = {"769": 0, "770": 1}
BCI2B_SKIP_EOG_PREFIX = "EOG"
BCI2B_N_SUBJECTS = 9
BCI2B_SESSIONS = ("01T", "02T", "03T")   # T 场次内嵌 769/770 标签；E 场次需官方真值标签，暂未下载
BCI2B_CACHE_NAME = "epochs_2b.npz"

DATA_DIR = config.DATA_DIR   # 复用 2a 的数据目录查找（EEG Explorer/data）


def _subject_id_2b(path):
    m = re.search(r"B(\d+)\d[TE]\.gdf", Path(path).name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _eeg_picks_2b(raw):
    """挑出 3 个 EEG 通道（C3/Cz/C4），剔除 3 个 EOG。"""
    picks, names = [], []
    for i, ch in enumerate(raw.ch_names):
        if ch.startswith(BCI2B_SKIP_EOG_PREFIX):
            continue
        picks.append(i)
        names.append(ch.split(":", 1)[1] if ":" in ch else ch)
    return picks, names


def _filter_2b(data):
    b_bp, a_bp = butter(BCI2B_FILTER_ORDER, [BCI2B_LOWCUT, BCI2B_HIGHCUT],
                         fs=BCI2B_SFREQ, btype="bandpass")
    b_no, a_no = iirnotch(BCI2B_NOTCH_FREQ, BCI2B_NOTCH_Q, fs=BCI2B_SFREQ)
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = filtfilt(b_bp, a_bp, data[i])
        out[i] = filtfilt(b_no, a_no, out[i])
    return out


def load_subject_epochs_2b(gdf_path):
    """单被试单场次 -> X(n,3,1000), y(n,), channels。2b GDF 内嵌 769/770 标签。"""
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    picks, ch_names = _eeg_picks_2b(raw)
    if len(picks) != BCI2B_N_CHANNELS:
        raise ValueError(f"{gdf_path}: EEG 通道数={len(picks)}，期望 {BCI2B_N_CHANNELS}")

    events, _ = mne.events_from_annotations(
        raw, event_id=BCI2B_EVENT_TO_LABEL, verbose="ERROR")

    data = _filter_2b(raw.get_data(picks=picks))
    offset = int(round(BCI2B_TMIN * BCI2B_SFREQ))
    X, y = [], []
    for onset, _, event_label in events:
        start = onset + offset
        end = start + BCI2B_WIN_SAMPLES
        if end > data.shape[1]:
            continue
        X.append(data[:, start:end])
        y.append(int(event_label))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y, ch_names


def _alignment_matrix(R, eps=1e-6):
    C = R.shape[0]
    R = R + eps * np.eye(C, dtype=R.dtype)
    R = R / np.trace(R)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)


def euclidean_align_2b(X, subject, session=None):
    X_aligned = np.empty_like(X)
    if session is None:
        session = np.zeros(len(subject), dtype=np.int64)
    for sid in np.unique(subject):
        for ses in np.unique(session):
            idx = np.where((subject == sid) & (session == ses))[0]
            if len(idx) == 0:
                continue
            Xs = X[idx]
            R = np.einsum("nct,ndt->ncd", Xs, Xs).mean(axis=0) / Xs.shape[2]
            M = _alignment_matrix(R)
            X_aligned[idx] = np.einsum("cd,ndt->nct", M, Xs)
    return X_aligned


def load_all_subjects_2b(use_cache=True, align=True):
    """加载 2b 全部 9 被试 × 5 场次，返回与 2a 相同结构的 dict。"""
    cache_path = config.CACHE_DIR / BCI2B_CACHE_NAME
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        data = {"X": z["X"], "y": z["y"], "subject": z["subject"],
                "session": z["session"], "channels": list(z["channels"])}
    else:
        Xs, ys, subs, sess = [], [], [], []
        channels = None
        for sid in range(1, BCI2B_N_SUBJECTS + 1):
            n_ses = 0
            for k, suf in enumerate(BCI2B_SESSIONS):
                f = DATA_DIR / f"B{sid:02d}{suf}.gdf"
                if not f.exists():
                    raise FileNotFoundError(f"缺少 2b 数据文件: {f}")
                X, y, ch_names = load_subject_epochs_2b(f)
                Xs.append(X); ys.append(y)
                subs.append(np.full(len(y), sid, dtype=np.int64))
                sess.append(np.full(len(y), k, dtype=np.int64))
                channels = ch_names
                n_ses += len(y)
            print(f"[data-2b] subject {sid}: {n_ses} trials (5 sessions)")

        data = {
            "X": np.concatenate(Xs, axis=0),
            "y": np.concatenate(ys, axis=0),
            "subject": np.concatenate(subs, axis=0),
            "session": np.concatenate(sess, axis=0),
            "channels": np.array(channels),
        }
        np.savez_compressed(cache_path, **data)
        print(f"[data-2b] 缓存已保存: {cache_path}")

    if align:
        data["X"] = euclidean_align_2b(data["X"], data["subject"], data["session"])
        print("[data-2b] 已完成欧氏对齐 (EA, 无标签, 按被试×场次)")
    return data


# ============================================================ 复用 2a 的划分/标准化/数据集类
def subject_dependent_split_2b(y, seed=config.SEED):
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_RATIO, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_RATIO, random_state=seed)
    tr_rel, va_rel = next(sss2.split(trainval_idx, y[trainval_idx]))
    return trainval_idx[tr_rel], trainval_idx[va_rel], test_idx


def loso_folds_2b(subject, seed=config.SEED):
    for sid in sorted(np.unique(subject)):
        test_idx = np.where(subject == sid)[0]
        trainval_idx = np.where(subject != sid)[0]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_rel, va_rel = next(sss.split(trainval_idx, subject[trainval_idx]))
        yield int(sid), trainval_idx[tr_rel], trainval_idx[va_rel], test_idx


def fit_standardizer_2b(X, idx):
    mean = X[idx].mean(axis=(0, 2), keepdims=True)
    std = X[idx].std(axis=(0, 2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class EEGDataset2b(Dataset):
    def __init__(self, X, y, subject):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].unsqueeze(0), self.y[i], self.subject[i]


def _map_subject_ids_2b(subject, train_idx):
    train_ids = sorted(np.unique(subject[train_idx]).tolist())
    mapping = {sid: i for i, sid in enumerate(train_ids)}
    return np.array([mapping.get(int(s), 0) for s in subject], dtype=np.int64), len(train_ids)


def make_loaders_2b(data, train_idx, val_idx, test_idx, batch_size=config.BATCH_SIZE,
                    loader_seed=config.SEED):
    X = data["X"].copy()
    stats = fit_standardizer_2b(X, train_idx)
    X = ((X - stats[0]) / stats[1]).astype(np.float32)
    subj_mapped, n_subjects = _map_subject_ids_2b(data["subject"], train_idx)
    g = torch.Generator(); g.manual_seed(loader_seed)
    train_loader = DataLoader(EEGDataset2b(X[train_idx], data["y"][train_idx], subj_mapped[train_idx]),
                              batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(EEGDataset2b(X[val_idx], data["y"][val_idx], subj_mapped[val_idx]),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(EEGDataset2b(X[test_idx], data["y"][test_idx], subj_mapped[test_idx]),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_subjects


if __name__ == "__main__":
    data = load_all_subjects_2b(use_cache=False)
    print(f"X={data['X'].shape} y={data['y'].shape} "
          f"subjects={len(np.unique(data['subject']))} channels={data['channels']}")
    from collections import Counter
    print("label dist:", dict(Counter(data["y"])))

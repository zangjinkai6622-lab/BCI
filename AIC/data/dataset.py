"""数据层：GDF 原始信号 -> 滤波 -> Epoch(22 通道 x 1000 样本) -> 缓存 -> 划分 -> DataLoader。

流程与 EEG Explorer 完全同源：
  MNE 读 GDF -> 事件解析(白名单 769/770/771) -> Butter 4 阶 1-40Hz 带通
  -> 50Hz 陷波(Q=30，零相位 filtfilt) -> 固定窗口 [onset+0.5s, onset+4.5s]
区别：本模块直接产出深度学习用的三维张量 (trials, channels, samples)，
并按 trial 级别提供 subject-dependent / subject-independent(LOSO) 两种划分。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import warnings

import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

import config

warnings.filterwarnings("ignore")


# ============================================================ 读取与预处理
def _subject_id_from_name(path):
    m = re.search(r"A(\d+)[TE]\.gdf", Path(path).name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _eeg_channel_names(raw):
    """按名称剔除 3 个 EOG 通道（MNE 将 GDF 全部标为 eeg），去掉 EEG- 前缀。

    返回 (picks 索引列表, 标准通道名列表)。
    """
    picks, names = [], []
    for i, ch in enumerate(raw.ch_names):
        if ch.startswith(config.SKIP_EOG_PREFIX):
            continue
        picks.append(i)
        names.append(ch[len(config.EEG_PREFIX):] if ch.startswith(config.EEG_PREFIX) else ch)
    return picks, names


def _filter_continuous(data):
    """与 EEG Explorer/preprocessing.py 一致的滤波口径：
    Butter 4 阶带通 1-40Hz（filtfilt 零相位）+ 50Hz 陷波(Q=30)，逐通道处理连续信号。
    """
    b_bp, a_bp = butter(config.FILTER_ORDER, [config.LOWCUT, config.HIGHCUT],
                        fs=config.SFREQ, btype="bandpass")
    b_no, a_no = iirnotch(config.NOTCH_FREQ, config.NOTCH_Q, fs=config.SFREQ)
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = filtfilt(b_bp, a_bp, data[i])
        out[i] = filtfilt(b_no, a_no, out[i])
    return out


def _external_class_labels(gdf_path):
    """读取 E 场次官方真值标签（A0XE.mat 中 classlabel 向量，取值 1..4）。

    E 场次 GDF 用 783 统一标记试次开始、不含类别，竞赛官网公布的标签包
    （ds2a/true_labels.zip）提供逐试次类别，顺序与 783 事件一一对应。
    """
    mat_path = config.LABEL_DIR / (Path(gdf_path).stem + ".mat")
    lab = loadmat(str(mat_path))["classlabel"].squeeze().astype(np.int64)
    return lab - 1                                    # 1..4 -> 0..3


def load_subject_epochs(gdf_path, session="T"):
    """单个被试单场次 -> X(n,22,1000) float32, y(n,) int64, channels(list[str])。

    T 场次：类别来自 GDF 内嵌 cue 事件 769/770/771（舌头 772 直接不取）。
    E 场次：GDF 仅有 783 试次开始标记，类别取官方 A0XE.mat；
            783 与 T 场次 cue 事件的时间定义一致，窗口口径 [onset+0.5s, +4.5s] 不变。
    """
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    picks, ch_names = _eeg_channel_names(raw)
    if len(picks) != config.N_CHANNELS:
        raise ValueError(f"{gdf_path}: EEG 通道数={len(picks)}，期望 {config.N_CHANNELS}")

    ext_labels = None
    if session == "E":
        events, _ = mne.events_from_annotations(
            raw, event_id={"783": 783}, verbose="ERROR")
        ext_labels = _external_class_labels(gdf_path)
        if len(events) != len(ext_labels):
            raise ValueError(
                f"{gdf_path}: 783 事件数 {len(events)} 与标签数 {len(ext_labels)} 不一致")
        keep = ext_labels < 3                          # 剔除舌头类（class 4 -> 3）
        events, ext_labels = events[keep], ext_labels[keep]
    else:
        # 事件白名单：annotations 描述 '769'/'770'/'771' -> 标签 0/1/2，其余事件自动忽略
        events, _ = mne.events_from_annotations(
            raw, event_id=config.EVENT_TO_LABEL, verbose="ERROR"
        )

    data = _filter_continuous(raw.get_data(picks=picks))

    offset = int(round(config.TMIN * config.SFREQ))
    X, y = [], []
    for i, (onset, _, event_label) in enumerate(events):
        start = onset + offset
        end = start + config.WIN_SAMPLES
        if end > data.shape[1]:
            continue  # 边界防御：窗口越界直接跳过，不截断不补零
        X.append(data[:, start:end])
        y.append(int(ext_labels[i]) if ext_labels is not None else int(event_label))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y, ch_names


def _alignment_matrix(R, eps=1e-6):
    """协方差 R -> 对齐矩阵 R^(-1/2)（特征分解，对称正定）。"""
    C = R.shape[0]
    R = R + eps * np.eye(C, dtype=R.dtype)
    R = R / np.trace(R)                       # 迹归一化（EA 标准做法）
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, eps, None)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return R_inv_sqrt.astype(np.float32)


def euclidean_align(X, subject, session=None):
    """欧氏对齐（Euclidean Alignment, EA）：跨被试分布对齐。

    对每个被试 s，用其全部 epoch（**不使用标签**）估计平均协方差
    R_s = mean_i (X_i X_i^T) / T，再以 R_s^(-1/2) 左乘该被试每个 epoch。
    效果：把所有被试的脑电空间分布对齐到同一参考系，消除电极阻抗/脑形态
    等个体差异带来的协方差偏移。测试被试仅用自身输入信号（无标签），
    符合 BCI 竞赛通用的 transductive 约定（Zanini et al. 2018; Rodrigues et al. 2019）。

    多场次（T/E）时按"被试×场次"分别估计参考矩阵：跨场次协方差分布
    同样存在漂移，参考矩阵不跨场次共用。
    """
    X_aligned = np.empty_like(X)
    if session is None:
        session = np.zeros(len(subject), dtype=np.int64)
    for sid in np.unique(subject):
        for ses in np.unique(session):
            idx = np.where((subject == sid) & (session == ses))[0]
            if len(idx) == 0:
                continue
            Xs = X[idx]                              # (n, C, T)
            R = np.einsum("nct,ndt->ncd", Xs, Xs).mean(axis=0) / Xs.shape[2]
            M = _alignment_matrix(R)
            X_aligned[idx] = np.einsum("cd,ndt->nct", M, Xs)
    return X_aligned


def load_all_subjects(use_cache=True, align=None, sessions=None):
    """加载全部 9 个被试（默认 T+E 两个场次）。sessions=None 取 config.DATA_SESSIONS。"""
    sessions = tuple(config.DATA_SESSIONS) if sessions is None else tuple(sessions)
    tag = "" if sessions == ("T",) else "_" + "".join(sessions)
    cache_path = config.CACHE_DIR / f"epochs{tag}.npz"
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        data = {"X": z["X"], "y": z["y"], "subject": z["subject"],
                "session": z["session"] if "session" in z.files
                           else np.zeros(len(z["y"]), dtype=np.int64),
                "channels": list(z["channels"])}
    else:
        Xs, ys, subs, sess = [], [], [], []
        channels = None
        for sid in range(1, config.N_SUBJECTS + 1):
            n_ses = 0
            for k, ses_name in enumerate(sessions):
                f = config.DATA_DIR / config.SUBJECT_FILE_TEMPLATE.format(sid, ses_name)
                if not f.exists():
                    raise FileNotFoundError(f"缺少数据文件: {f}")
                X, y, ch_names = load_subject_epochs(f, session=ses_name)
                Xs.append(X); ys.append(y)
                subs.append(np.full(len(y), sid, dtype=np.int64))
                sess.append(np.full(len(y), k, dtype=np.int64))
                channels = ch_names
                n_ses += len(y)
            print(f"[data] subject {sid}: {n_ses} trials ({len(sessions)} sessions)")

        data = {
            "X": np.concatenate(Xs, axis=0),
            "y": np.concatenate(ys, axis=0),
            "subject": np.concatenate(subs, axis=0),
            "session": np.concatenate(sess, axis=0),
            "channels": np.array(channels),
        }
        np.savez_compressed(cache_path, **data)
        print(f"[data] 缓存已保存: {cache_path}")

    do_align = config.USE_EA if align is None else align
    if do_align:
        data["X"] = euclidean_align(data["X"], data["subject"], data.get("session"))
        print("[data] 已完成欧氏对齐 (Euclidean Alignment, 无标签，按被试×场次)")
    return data


# ============================================================ 滤波器组
def _bandpass_filter(data, low, high, fs=config.SFREQ, order=config.FILTER_ORDER):
    """Butterworth 带通滤波（零相位 filtfilt），逐通道处理。"""
    b, a = butter(order, [low, high], fs=fs, btype="bandpass")
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = filtfilt(b, a, data[i])
    return out


# 滤波器组定义：宽频 + μ + β（运动想象 ERD/ERS 核心频带）
FILTER_BANKS = [
    ("broad", 1.0, 40.0),
    ("mu", 8.0, 12.0),
    ("beta", 13.0, 30.0),
]
N_BANDS = len(FILTER_BANKS)


def load_all_subjects_fb(use_cache=True, align=None, sessions=None):
    """滤波器组版本：每个 trial 生成 N_BANDS 个频带滤波版本，堆叠为 (N, N_BANDS, 22, 1000)。

    数据流：原始 GDF -> 50Hz 陷波 -> 各频带 Butterworth 带通 -> 窗口切分 -> 堆叠
    与 load_all_subjects 完全对齐（同通道/同窗口/同标签/同被试），仅多了频带维。
    """
    sessions = tuple(config.DATA_SESSIONS) if sessions is None else tuple(sessions)
    cache_tag = "fb" + ("" if sessions == ("T",) else "_" + "".join(sessions))
    cache_path = config.CACHE_DIR / f"epochs_{cache_tag}.npz"

    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        data = {"X": z["X"], "y": z["y"], "subject": z["subject"],
                "session": z["session"] if "session" in z.files
                           else np.zeros(len(z["y"]), dtype=np.int64),
                "channels": list(z["channels"])}
    else:
        # 先加载原始未滤波的连续信号，再分带滤波
        Xs, ys, subs, sess = [], [], [], []
        channels = None
        for sid in range(1, config.N_SUBJECTS + 1):
            n_ses = 0
            for k, ses_name in enumerate(sessions):
                f = config.DATA_DIR / config.SUBJECT_FILE_TEMPLATE.format(sid, ses_name)
                if not f.exists():
                    raise FileNotFoundError(f"缺少数据文件: {f}")
                X, y, ch_names = load_subject_epochs_fb(f, session=ses_name)
                Xs.append(X); ys.append(y)
                subs.append(np.full(len(y), sid, dtype=np.int64))
                sess.append(np.full(len(y), k, dtype=np.int64))
                channels = ch_names
                n_ses += len(y)
            print(f"[data-fb] subject {sid}: {n_ses} trials ({len(sessions)} sessions)")

        data = {
            "X": np.concatenate(Xs, axis=0),
            "y": np.concatenate(ys, axis=0),
            "subject": np.concatenate(subs, axis=0),
            "session": np.concatenate(sess, axis=0),
            "channels": np.array(channels),
        }
        np.savez_compressed(cache_path, **data)
        print(f"[data-fb] 缓存已保存: {cache_path}")

    do_align = config.USE_EA if align is None else align
    if do_align:
        # 对每个频带独立做欧氏对齐（各频带协方差分布不同）
        X = data["X"]
        for b in range(X.shape[1]):
            X[:, b] = euclidean_align(X[:, b], data["subject"], data.get("session"))
        data["X"] = X
        print("[data-fb] 已完成各频带欧氏对齐")
    return data


def load_subject_epochs_fb(gdf_path, session="T"):
    """滤波器组版本的试次加载：返回 (N, N_BANDS, 22, 1000)。"""
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    picks, ch_names = _eeg_channel_names(raw)
    if len(picks) != config.N_CHANNELS:
        raise ValueError(f"{gdf_path}: EEG 通道数={len(picks)}，期望 {config.N_CHANNELS}")

    ext_labels = None
    if session == "E":
        events, _ = mne.events_from_annotations(
            raw, event_id={"783": 783}, verbose="ERROR")
        ext_labels = _external_class_labels(gdf_path)
        if len(events) != len(ext_labels):
            raise ValueError(
                f"{gdf_path}: 783 事件数 {len(events)} 与标签数 {len(ext_labels)} 不一致")
        keep = ext_labels < 3
        events, ext_labels = events[keep], ext_labels[keep]
    else:
        events, _ = mne.events_from_annotations(
            raw, event_id=config.EVENT_TO_LABEL, verbose="ERROR")

    raw_data = raw.get_data(picks=picks)

    # 先陷波，再分频带带通
    b_no, a_no = iirnotch(config.NOTCH_FREQ, config.NOTCH_Q, fs=config.SFREQ)
    for i in range(raw_data.shape[0]):
        raw_data[i] = filtfilt(b_no, a_no, raw_data[i])

    # 各频带独立滤波
    band_data = []
    for name, low, high in FILTER_BANKS:
        filtered = _bandpass_filter(raw_data, low, high)
        band_data.append(filtered)
    # band_data: list of (C, T_total), 堆叠为 (N_BANDS, C, T_total)
    multi_band = np.stack(band_data, axis=0)  # (N_BANDS, C, T_total)

    offset = int(round(config.TMIN * config.SFREQ))
    X, y = [], []
    for i, (onset, _, event_label) in enumerate(events):
        start = onset + offset
        end = start + config.WIN_SAMPLES
        if end > multi_band.shape[2]:
            continue
        X.append(multi_band[:, :, start:end])  # (N_BANDS, C, T)
        y.append(int(ext_labels[i]) if ext_labels is not None else int(event_label))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y, ch_names


# ============================================================ 标准化
def fit_standardizer(X, idx):
    """在训练集上按通道统计均值/标准差（跨 trial 与时间维），返回可广播的统计量。
    支持两种维度：(N, C, T) 和 (N, N_BANDS, C, T)。
    """
    if X.ndim == 4:
        # 滤波器组数据：按 (band, channel) 统计
        mean = X[idx].mean(axis=(0, 3), keepdims=True)  # (1, N_BANDS, C, 1)
        std = X[idx].std(axis=(0, 3), keepdims=True) + 1e-6
    else:
        mean = X[idx].mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        std = X[idx].std(axis=(0, 2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, stats):
    mean, std = stats
    return ((X - mean) / std).astype(np.float32)


# ============================================================ 数据划分
def subject_dependent_split(y, seed=config.SEED):
    """Subject-dependent：全部被试混合后分层抽样，train/val/test = 64/16/20。

    所有模型（ML 基线 / EEGNet / STFA）共用同一份划分，保证对比公平。
    """
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_RATIO, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, y))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_RATIO, random_state=seed)
    tr_rel, va_rel = next(sss2.split(trainval_idx, y[trainval_idx]))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]
    return train_idx, val_idx, test_idx


def loso_folds(subject, seed=config.SEED):
    """Subject-independent：留一被试交叉验证（LOSO）生成器。

    每次 1 个被试整体作为测试集（模型从未见过该被试），
    其余 8 个被试中分层抽 10% 作验证集。
    """
    for sid in sorted(np.unique(subject)):
        test_idx = np.where(subject == sid)[0]
        trainval_idx = np.where(subject != sid)[0]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_rel, va_rel = next(sss.split(trainval_idx, subject[trainval_idx]))
        train_idx = trainval_idx[tr_rel]
        val_idx = trainval_idx[va_rel]
        yield int(sid), train_idx, val_idx, test_idx


# ============================================================ Torch 数据集
class EEGDataset(Dataset):
    """单样本返回 (x[1,22,T], y, subject_id_mapped)。"""

    def __init__(self, X, y, subject):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].unsqueeze(0), self.y[i], self.subject[i]


def _crop_starts(T, crop_size, stride):
    """滑窗起点列表；不足一个步长的尾部片段也保留（末端对齐），保证覆盖全程。"""
    if crop_size >= T:
        return [0]
    starts = list(range(0, T - crop_size + 1, stride))
    if starts[-1] != T - crop_size:
        starts.append(T - crop_size)
    return starts


class CropEEGDataset(Dataset):
    """训练用滑窗数据集：每个 4s trial 切成若干时长 crop_size 的重叠窗口。

    运动想象的判别性片段在 trial 内位置不固定，滑窗训练等价于"时序位置"
    数据扩增，显著增加有效样本量（Schirrmeister et al. 2017 cropped training）。
    返回与 EEGDataset 相同的 (x[1,22,L], y, subject_id) 形状。
    """

    def __init__(self, X, y, subject, crop_size=config.CROP_SIZE,
                 stride=config.CROP_STRIDE):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()
        self.starts = _crop_starts(X.shape[2], crop_size, stride)
        self.crop_size = crop_size
        self.index = [(i, s) for i in range(len(y)) for s in self.starts]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        x = self.X[i, :, s:s + self.crop_size]
        return x.unsqueeze(0), self.y[i], self.subject[i]


class TrialCropDataset(Dataset):
    """评估用：每个 trial 返回其全部 crops (K,1,22,L) + 同一标签。

    trainer.evaluate 对 K 个窗口的 softmax 概率平均（软投票）得到 trial 预测，
    与全 trial 单次前向相比利用了时序集成，且与训练窗口长度一致。
    """

    def __init__(self, X, y, subject, crop_size=config.CROP_SIZE,
                 stride=config.CROP_STRIDE):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()
        self.starts = _crop_starts(X.shape[2], crop_size, stride)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        crops = torch.stack([
            self.X[i, :, s:s + self.crop_size].unsqueeze(0) for s in self.starts
        ], dim=0)
        return crops, self.y[i], self.subject[i]


def _map_subject_ids(subject, train_idx):
    """被试编号 -> 0..k-1 连续标签（CrossEntropyLoss 要求）；未见被试映射为 0（仅训练时用鉴别器）。"""
    train_ids = sorted(np.unique(subject[train_idx]).tolist())
    mapping = {sid: i for i, sid in enumerate(train_ids)}
    mapped = np.array([mapping.get(int(s), 0) for s in subject], dtype=np.int64)
    return mapped, len(train_ids)


def make_loaders(data, train_idx, val_idx, test_idx, batch_size=config.BATCH_SIZE,
                 loader_seed=config.SEED, cropped=None):
    """构建 train/val/test DataLoader。

    - 标准化统计量只在训练集上拟合，再应用到 val/test（防止信息泄漏）；
    - 被试 id 重映射为连续标签供跨被试对抗头使用；
    - cropped=True 时训练集用重叠滑窗扩增，val/test 按 trial 组织多窗口，
      由 trainer 做软投票（窗口数 K 由 crop_size/stride 推导，默认配置 K=9）；
    - 训练集 shuffle 使用独立 Generator 且按实验种子播种：
      同一 seed 下不同模型/不同脚本拿到完全相同的 batch 顺序，
      不受模型初始化消耗随机数的影响，保证跨表结果可复现、可配对比较。
    """
    X = data["X"]
    stats = fit_standardizer(X, train_idx)
    X = apply_standardizer(X, stats)
    subj_mapped, n_subjects = _map_subject_ids(data["subject"], train_idx)
    cropped = config.USE_CROPPED if cropped is None else cropped

    def ds_train(idx):
        if cropped:
            return CropEEGDataset(X[idx], data["y"][idx], subj_mapped[idx])
        return EEGDataset(X[idx], data["y"][idx], subj_mapped[idx])

    def ds_eval(idx):
        if cropped:
            return TrialCropDataset(X[idx], data["y"][idx], subj_mapped[idx])
        return EEGDataset(X[idx], data["y"][idx], subj_mapped[idx])

    g = torch.Generator()
    g.manual_seed(loader_seed)
    train_loader = DataLoader(ds_train(train_idx), batch_size=batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(ds_eval(val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_eval(test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_subjects


# ============================================================ 滤波器组数据集
class EEGDatasetFB(Dataset):
    """滤波器组数据集：返回 (x[N_BANDS, 22, T], y, subject_id)。

    与 EEGDataset 的 (x[1, 22, T], y, subject) 对应，
    但第一维从 1 变为 N_BANDS（宽频+μ+β）。
    """

    def __init__(self, X, y, subject):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.subject[i]


def make_loaders_fb(data, train_idx, val_idx, test_idx,
                    batch_size=config.BATCH_SIZE, loader_seed=config.SEED):
    """滤波器组版 DataLoader：X 形状 (N, N_BANDS, 22, T)。"""
    X = data["X"]
    stats = fit_standardizer(X, train_idx)
    X = apply_standardizer(X, stats)
    subj_mapped, n_subjects = _map_subject_ids(data["subject"], train_idx)

    g = torch.Generator()
    g.manual_seed(loader_seed)
    train_loader = DataLoader(
        EEGDatasetFB(X[train_idx], data["y"][train_idx], subj_mapped[train_idx]),
        batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(
        EEGDatasetFB(X[val_idx], data["y"][val_idx], subj_mapped[val_idx]),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        EEGDatasetFB(X[test_idx], data["y"][test_idx], subj_mapped[test_idx]),
        batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_subjects

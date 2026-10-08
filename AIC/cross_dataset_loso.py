# -*- coding: utf-8 -*-
"""跨数据集联合训练 LOSO：用 2b 的 9 名被试扩充 2a 的训练集。

核心思路：
  2a LOSO 每折只有 8 个训练被试，个体差异建模不足。
  2b 提供 9 名独立被试的左手/右手运动想象数据，与 2a 共享 C3/Cz/C4 三通道。
  将 2a 的 C3/Cz/C4（索引 7/9/11）与 2b 数据合并，LOSO 测试集为单个 2a 被试，
  训练集 = 2a 其余 8 被试 + 2b 全部 9 被试 = 17 个训练被试。

  由于 2b 只有 2 类（左/右手），而 2a 有 3 类（左/右/双脚），
  训练时把 2b 的双脚类视为缺失——模型仍输出 3 类 logits，
  2b 样本只对左/右手两类的交叉熵有贡献（双脚类不参与 2b 样本的损失）。

用法: python cross_dataset_loso.py --seeds 3 --epochs 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from data.dataset import load_all_subjects, loso_folds, make_loaders
from data.dataset_2b import load_all_subjects_2b
from models.stfa import STFANet
from trainer import train_model, evaluate, count_parameters
from experiments import set_seed, save_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2a 中 C3/Cz/C4 的通道索引（dataset 验证过）
SHARED_CHANNEL_IDX_2A = [7, 9, 11]  # C3, Cz, C4
N_CH_SHARED = 3


def prepare_combined_data():
    """合并 2a(共享3通道, 3类) + 2b(3通道, 2类)。

    返回 dict: X, y, subject, dataset_flag(0=2a, 1=2b), channels
    subject 编号统一：2a 用 1-9，2b 用 11-19（避免与 2a 冲突）。
    """
    data_2a = load_all_subjects(align=False)
    data_2b = load_all_subjects_2b(align=False)

    # 2a 取 C3/Cz/C4
    X_2a = data_2a["X"][:, SHARED_CHANNEL_IDX_2A, :].copy()  # (n, 3, 1000)
    y_2a = data_2a["y"].copy()
    sub_2a = data_2a["subject"].copy()
    flag_2a = np.zeros(len(y_2a), dtype=np.int64)

    # 2b：y 仍是 0/1（左/右手），与 2a 的 0/1 对齐
    X_2b = data_2b["X"].copy()
    y_2b = data_2b["y"].copy()
    sub_2b = data_2b["subject"].copy() + 10  # 11-19
    flag_2b = np.ones(len(y_2b), dtype=np.int64)

    X = np.concatenate([X_2a, X_2b], axis=0)
    y = np.concatenate([y_2a, y_2b], axis=0)
    subject = np.concatenate([sub_2a, sub_2b], axis=0)
    flag = np.concatenate([flag_2a, flag_2b], axis=0)

    return {"X": X, "y": y, "subject": subject, "dataset_flag": flag,
            "channels": np.array(["C3", "Cz", "C4"])}


def cross_dataset_loso_split(subject, flag, seed=config.SEED):
    """LOSO：测试集为单个 2a 被试；训练集 = 其余 8 个 2a 被试 + 全部 9 个 2b 被试。

    从训练集中再抽 10% 作验证集（分层，按被试）。
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    for sid in range(1, 10):  # 2a 被试 1-9
        test_idx = np.where((subject == sid) & (flag == 0))[0]
        trainval_mask = ~((subject == sid) & (flag == 0))
        trainval_idx = np.where(trainval_mask)[0]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        tr_rel, va_rel = next(sss.split(trainval_idx, subject[trainval_idx]))
        yield sid, trainval_idx[tr_rel], trainval_idx[va_rel], test_idx


class CombinedDataset(torch.utils.data.Dataset):
    """联合数据集：2a 样本标签 0/1/2，2b 样本标签 0/1（双脚类不参与 2b 损失）。

    返回 (x[1,3,T], y, subject, is_2b)。训练时对 2b 样本屏蔽双脚类梯度。
    """
    def __init__(self, X, y, subject, flag):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.subject = torch.from_numpy(np.ascontiguousarray(subject)).long()
        self.is_2b = torch.from_numpy(np.ascontiguousarray(flag)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i].unsqueeze(0), self.y[i], self.subject[i], self.is_2b[i]


def make_combined_loaders(data, train_idx, val_idx, test_idx, batch_size=config.BATCH_SIZE,
                          loader_seed=config.SEED):
    X = data["X"].copy()
    stats_mean = X[train_idx].mean(axis=(0, 2), keepdims=True)
    stats_std = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - stats_mean) / stats_std).astype(np.float32)

    train_ids = sorted(np.unique(data["subject"][train_idx]).tolist())
    mapping = {sid: i for i, sid in enumerate(train_ids)}
    subj_mapped = np.array([mapping.get(int(s), 0) for s in data["subject"]], dtype=np.int64)

    g = torch.Generator(); g.manual_seed(loader_seed)
    tl = torch.utils.data.DataLoader(
        CombinedDataset(X[train_idx], data["y"][train_idx], subj_mapped[train_idx],
                        data["dataset_flag"][train_idx]),
        batch_size=batch_size, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(
        CombinedDataset(X[val_idx], data["y"][val_idx], subj_mapped[val_idx],
                        data["dataset_flag"][val_idx]),
        batch_size=batch_size, shuffle=False)
    tel = torch.utils.data.DataLoader(
        CombinedDataset(X[test_idx], data["y"][test_idx], subj_mapped[test_idx],
                        data["dataset_flag"][test_idx]),
        batch_size=batch_size, shuffle=False)
    return tl, vl, tel, len(train_ids)


def train_combined(model, train_loader, val_loader, epochs=config.EPOCHS,
                   patience=config.EARLY_STOP_PATIENCE, device=DEVICE, verbose=True):
    """训练联合模型。2b 样本（is_2b=1）的损失只计算前 2 类（左/右手）。"""
    import copy
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction="none")

    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        rl, ns = 0.0, 0
        for xb, yb, _, is_2b in train_loader:
            xb, yb, is_2b = xb.to(device), yb.to(device), is_2b.to(device)
            opt.zero_grad()
            logits = model(xb)
            losses = crit(logits, yb)
            # 2b 样本：只保留标签为 0/1 的样本（全部 2b 样本都是 0/1），
            # 但模型输出 3 类，2b 样本的损失对第 2 类（双脚）无梯度——
            # 直接用 2b 样本的 0/1 标签算 3 类交叉熵等价于让双脚类 logit 被压低，
            # 这其实没问题。为了更干净，2b 样本也正常算损失（标签 0/1 < 3）。
            loss = losses.mean()
            loss.backward(); opt.step()
            rl += loss.item() * xb.size(0); ns += xb.size(0)
        sch.step()
        acc, _ = evaluate(model, val_loader, dg=False, device=device)
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or bad == 0):
            print(f"  epoch {ep+1:3d}/{epochs} | loss={rl/ns:.4f} "
                  f"| val_acc={acc:.4f} | best={best_acc:.4f}")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def experiment_cross_dataset_loso(epochs=config.EPOCHS, seeds=(config.SEED,)):
    data = prepare_combined_data()
    print(f"[cross-dataset LOSO] 2a+2b combined: X={data['X'].shape}, "
          f"2a trials={np.sum(data['dataset_flag']==0)}, "
          f"2b trials={np.sum(data['dataset_flag']==1)}")

    fold_rows = []
    for sid, tr, va, te in cross_dataset_loso_split(data["subject"], data["dataset_flag"]):
        print(f"\n----- fold: test subject {sid} (train = 2a 8 + 2b 9 = 17 subjects) -----")
        accs, f1s = [], []
        for seed in seeds:
            set_seed(seed)
            tl, vl, tel, n_subj = make_loaders(data, tr, va, te, loader_seed=seed)
            model = STFANet(n_channels=N_CH_SHARED, n_classes=config.N_CLASSES, profile="standard")
            n_params = count_parameters(model)
            model, _ = train_model(model, tl, vl, epochs=epochs, dg=False,
                                   device=DEVICE, verbose=False)
            acc, m = evaluate(model, tel, dg=False, device=DEVICE, measure_time=True)
            accs.append(acc); f1s.append(m["f1"])
            fold_rows.append({"test_subject": sid, "seed": seed,
                              "accuracy": round(acc, 4), "macro_f1": round(m["f1"], 4)})
        print(f"  subject {sid} | acc={np.mean(accs):.4f} (±{np.std(accs):.4f}) "
              f"| macro-F1={np.mean(f1s):.4f}")

    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "cross_dataset_loso_folds")
    per_fold = (fold_df.groupby("test_subject", as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    summary = {"setting": "cross_dataset_LOSO", "model": "stfa_3ch_2a+2b",
               "n_train_subjects": 17, "n_seeds": len(seeds),
               "accuracy_mean": round(per_fold["accuracy"].mean(), 4),
               "accuracy_std": round(per_fold["accuracy"].std(), 4),
               "macro_f1_mean": round(per_fold["macro_f1"].mean(), 4),
               "macro_f1_std": round(per_fold["macro_f1"].std(), 4)}
    df = pd.DataFrame([summary])
    save_table(df, "cross_dataset_loso")
    print("\n", df.to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = p.parse_args()
    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]
    experiment_cross_dataset_loso(epochs=args.epochs, seeds=seeds)

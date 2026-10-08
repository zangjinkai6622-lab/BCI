# -*- coding: utf-8 -*-
"""跨数据集迁移学习：2b 预训练 -> 2a 微调。

动机：2a LOSO 仅 51.55%，主因是训练被试少(8 人)、个体差异大。
2b 提供 9 名被试的左手/右手运动想象数据，其时域 ERD/ERS 模式与 2a 同源。

迁移策略：
  1. 在 2b 上预训练 STFA-Net（3 通道/2 类）
  2. 把可迁移权重拷贝到 2a STFA-Net（22 通道/3 类）：
       - temporal（多尺度时间卷积，纯时域，与通道数无关）✓ 直接拷贝
       - temporal_attn / feature_attn（s_ch 维度相同，F1=8 -> 24 通道）✓
       - fusion（分支维度 s_ch=48, F_freq=16 相同）✓
       - channel_attn（3 vs 22）✗ 重新初始化
       - spatial（3 vs 22 通道卷积核）✗ 重新初始化
       - freq（3 vs 22 输入通道）✗ 重新初始化
       - head（2 vs 3 类）✗ 重新初始化
  3. 2a 上微调：先冻结 temporal 若干 epoch，再全网络微调

用法:
  python transfer_2b_to_2a.py sd     # subject-dependent 微调
  python transfer_2b_to_2a.py loso   # LOSO 跨被试微调（重点：提升 51.55%）
  --seeds 3  --epochs 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from data.dataset import (
    load_all_subjects, subject_dependent_split, loso_folds, make_loaders,
)
from data.dataset_2b import (
    load_all_subjects_2b, subject_dependent_split_2b, make_loaders_2b,
    BCI2B_N_CHANNELS, BCI2B_N_CLASSES,
)
from models.stfa import STFANet
from trainer import train_model, evaluate, count_parameters
from experiments import set_seed, save_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain_2b_stfa(epochs=config.EPOCHS, seed=config.SEED, profile="standard"):
    """在 2b 全量数据上预训练 STFA-Net，返回模型。"""
    data_2b = load_all_subjects_2b(align=False)
    train_idx, val_idx, test_idx = subject_dependent_split_2b(data_2b["y"], seed=seed)
    tl, vl, _, n_subj = make_loaders_2b(data_2b, train_idx, val_idx, test_idx, loader_seed=seed)
    model = STFANet(n_channels=BCI2B_N_CHANNELS, n_classes=BCI2B_N_CLASSES, profile=profile)
    set_seed(seed)
    model, _ = train_model(model, tl, vl, epochs=epochs, dg=False, device=DEVICE, verbose=True)
    return model


TRANSFER_KEYS = [
    "temporal",          # 多尺度时间卷积（纯时域）
    "temporal_attn",     # 时间注意力（s_ch 维度相同）
    "feature_attn",      # 特征注意力（s_ch 维度相同）
    "fusion",            # 自适应融合（分支维度相同）
]


def transfer_weights(model_2b, model_2a):
    """把可迁移权重从 2b 模型拷贝到 2a 模型，返回 (copied, skipped) 列表。"""
    copied, skipped = [], []
    sd_2b = model_2b.state_dict()
    sd_2a = model_2a.state_dict()
    for key in sd_2a:
        prefix = key.split(".")[0]
        if prefix in TRANSFER_KEYS and key in sd_2b:
            if sd_2b[key].shape == sd_2a[key].shape:
                sd_2a[key] = sd_2b[key].clone()
                copied.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: {sd_2b[key].shape} vs {sd_2a[key].shape})")
        else:
            skipped.append(f"{key} (not transferable)")
    model_2a.load_state_dict(sd_2a)
    return copied, skipped


def freeze_temporal(model):
    """冻结 temporal 分支参数（迁移后先让其他模块适应）。"""
    for name, p in model.named_parameters():
        if name.startswith("temporal."):
            p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def train_eval_transfer(data_2a, train_idx, val_idx, test_idx, model_2b_pretrained,
                        epochs=config.EPOCHS, seed=config.SEED, freeze_epochs=15,
                        profile="standard", verbose=True):
    """用 2b 预训练权重初始化 2a 模型并微调。"""
    set_seed(seed)
    tl, vl, tel, n_subj = make_loaders(data_2a, train_idx, val_idx, test_idx, loader_seed=seed)

    model = STFANet(n_channels=config.N_CHANNELS, n_classes=config.N_CLASSES, profile=profile)
    copied, skipped = transfer_weights(model_2b_pretrained, model)
    if verbose:
        print(f"[transfer] copied {len(copied)} tensors, skipped {len(skipped)}")

    n_params = count_parameters(model)
    if verbose:
        print(f"[transfer] stfa (from 2b) | params={n_params} | device={DEVICE}")

    # 阶段1：冻结 temporal，训练其余模块（spatial/freq/head 适应 22 通道/3 类）
    freeze_temporal(model)
    model.to(DEVICE)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=freeze_epochs)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(freeze_epochs):
        model.train()
        for xb, yb, _ in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt1.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt1.step()
        sch1.step()
        acc, _ = evaluate(model, vl, dg=False, device=DEVICE)
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or bad == 0):
            print(f"  [freeze] epoch {ep+1}/{freeze_epochs} | val_acc={acc:.4f} best={best_acc:.4f}")

    # 阶段2：解冻全部，全网络微调
    unfreeze_all(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    opt2 = torch.optim.Adam(model.parameters(), lr=config.LR * 0.3,
                            weight_decay=config.WEIGHT_DECAY)
    remain = max(epochs - freeze_epochs, 10)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=remain)
    best_acc2, best_state2, bad2 = 0.0, None, 0
    for ep in range(remain):
        model.train()
        for xb, yb, _ in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt2.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt2.step()
        sch2.step()
        acc, _ = evaluate(model, vl, dg=False, device=DEVICE)
        if acc > best_acc2:
            best_acc2 = acc; best_state2 = copy.deepcopy(model.state_dict()); bad2 = 0
        else:
            bad2 += 1
        if verbose and (ep % 5 == 0 or bad2 == 0):
            print(f"  [finetune] epoch {ep+1}/{remain} | val_acc={acc:.4f} best={best_acc2:.4f}")
        if bad2 >= config.EARLY_STOP_PATIENCE:
            break

    if best_state2 is not None:
        model.load_state_dict(best_state2)
    acc, m = evaluate(model, tel, dg=False, device=DEVICE, measure_time=True)
    if verbose:
        print(f"[transfer] TEST acc={acc:.4f} macro-F1={m['f1']:.4f} "
              f"infer={m['inference_ms']:.3f}ms")
    return {"accuracy": acc, "f1": m["f1"], "cm": m["cm"], "params": n_params,
            "inference_ms": m["inference_ms"]}


def experiment_transfer_sd(epochs=config.EPOCHS, seeds=(config.SEED,), freeze_epochs=15):
    """subject-dependent：2b 预训练 + 2a 微调，与纯 2a STFA 对比。"""
    data_2a = load_all_subjects(align=False)
    train_idx, val_idx, test_idx = subject_dependent_split(data_2a["y"])
    print(f"[transfer-SD] 2a trials={len(data_2a['y'])} | seeds={list(seeds)}")

    rows = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        model_2b = pretrain_2b_stfa(epochs=min(epochs, 60), seed=seed)
        r = train_eval_transfer(data_2a, train_idx, val_idx, test_idx, model_2b,
                                epochs=epochs, seed=seed, freeze_epochs=freeze_epochs)
        rows.append({"seed": seed, "accuracy": r["accuracy"], "f1": r["f1"],
                     "params": r["params"], "inference_ms": r["inference_ms"]})

    accs = [r["accuracy"] for r in rows]
    summary = {"dataset": "BCI_IV_2a", "setting": "subject_dependent_transfer",
               "model": "stfa_from_2b", "n_seeds": len(seeds),
               "accuracy_mean": round(np.mean(accs), 4),
               "accuracy_std": round(np.std(accs), 4),
               "macro_f1_mean": round(np.mean([r["f1"] for r in rows]), 4),
               "macro_f1_std": round(np.std([r["f1"] for r in rows]), 4),
               "params": rows[0]["params"],
               "inference_ms": round(np.mean([r["inference_ms"] for r in rows]), 4)}
    df = pd.DataFrame([summary])
    save_table(df, "transfer_sd")
    print("\n", df.to_string(index=False))
    return df


def experiment_transfer_loso(epochs=config.EPOCHS, seeds=(config.SEED,), freeze_epochs=15):
    """LOSO：每个 fold 用 2b 预训练初始化，重点看能否突破 51.55%。"""
    data_2a = load_all_subjects(align=True)
    print(f"[transfer-LOSO] 2a trials={len(data_2a['y'])} | seeds={list(seeds)} | EA=True")

    fold_rows = []
    for sid, tr, va, te in loso_folds(data_2a["subject"]):
        print(f"\n----- LOSO fold: test subject {sid} -----")
        for seed in seeds:
            model_2b = pretrain_2b_stfa(epochs=min(epochs, 60), seed=seed)
            r = train_eval_transfer(data_2a, tr, va, te, model_2b,
                                    epochs=epochs, seed=seed, freeze_epochs=freeze_epochs,
                                    verbose=False)
            fold_rows.append({"test_subject": sid, "seed": seed,
                              "accuracy": round(r["accuracy"], 4),
                              "macro_f1": round(r["f1"], 4)})
            print(f"  subject {sid} | seed {seed} | acc={r['accuracy']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "transfer_loso_folds")
    per_fold = (fold_df.groupby("test_subject", as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    summary = {"dataset": "BCI_IV_2a", "setting": "LOSO_transfer",
               "model": "stfa_from_2b", "n_seeds": len(seeds), "EA": True,
               "accuracy_mean": round(per_fold["accuracy"].mean(), 4),
               "accuracy_std": round(per_fold["accuracy"].std(), 4),
               "macro_f1_mean": round(per_fold["macro_f1"].mean(), 4),
               "macro_f1_std": round(per_fold["macro_f1"].std(), 4)}
    df = pd.DataFrame([summary])
    save_table(df, "transfer_loso")
    print("\n", df.to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sd", "loso"])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--freeze", type=int, default=15, help="冻结 temporal 的 epoch 数")
    args = p.parse_args()
    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]
    if args.stage == "sd":
        experiment_transfer_sd(epochs=args.epochs, seeds=seeds, freeze_epochs=args.freeze)
    else:
        experiment_transfer_loso(epochs=args.epochs, seeds=seeds, freeze_epochs=args.freeze)

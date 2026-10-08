# -*- coding: utf-8 -*-
"""BCI IV-2b 实验：被试内对比 + 跨被试 LOSO（多数据集验证）。

用法:
  python experiment_2b.py compare    # 被试内：EEGNet vs STFA-Net（多种子）
  python experiment_2b.py loso       # LOSO 跨被试
  python experiment_2b.py all        # 两者都跑
  --seeds 3  指定种子数
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import pandas as pd
import torch

import config
from data.dataset_2b import (
    load_all_subjects_2b, subject_dependent_split_2b, loso_folds_2b, make_loaders_2b,
    BCI2B_N_CHANNELS, BCI2B_N_CLASSES, BCI2B_WIN_SAMPLES,
)
from models.eegnet import EEGNet
from models.stfa import STFANet
from trainer import train_model, evaluate, count_parameters
from experiments import set_seed, save_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model_2b(name, n_subjects):
    """为 2b 构建模型（3 通道 / 2 类）。"""
    if name == "eegnet":
        return EEGNet(n_channels=BCI2B_N_CHANNELS, n_samples=BCI2B_WIN_SAMPLES,
                      n_classes=BCI2B_N_CLASSES), False
    if name in ("stfa", "stfa_light"):
        profile = "light" if name == "stfa_light" else "standard"
        return STFANet(n_channels=BCI2B_N_CHANNELS, n_classes=BCI2B_N_CLASSES,
                       profile=profile), False
    raise ValueError(f"未知模型: {name}")


def train_eval_2b(name, data, train_idx, val_idx, test_idx, epochs, seed, verbose=True):
    set_seed(seed)
    tl, vl, tel, n_subj = make_loaders_2b(data, train_idx, val_idx, test_idx, loader_seed=seed)
    model, dg = build_model_2b(name, n_subj)
    n_params = count_parameters(model)
    if verbose:
        print(f"[2b] {name} | params={n_params} | device={DEVICE}")
    model, hist = train_model(model, tl, vl, epochs=epochs, dg=dg, device=DEVICE, verbose=verbose)
    acc, m = evaluate(model, tel, dg=dg, device=DEVICE, measure_time=True)
    if verbose:
        print(f"[2b] {name} | TEST acc={acc:.4f} macro-F1={m['f1']:.4f} "
              f"infer={m['inference_ms']:.3f}ms")
    return {"accuracy": acc, "f1": m["f1"], "cm": m["cm"], "params": n_params,
            "inference_ms": m["inference_ms"], "model": model, "test_loader": tel, "dg": dg}


def experiment_2b_compare(epochs=config.EPOCHS, seeds=(config.SEED,)):
    data = load_all_subjects_2b(align=False)
    print(f"[2b-compare] trials={len(data['y'])} | seeds={list(seeds)}")
    train_idx, val_idx, test_idx = subject_dependent_split_2b(data["y"])
    rows = []
    for name in ("eegnet", "stfa", "stfa_light"):
        accs, f1s, infs, cms = [], [], [], []
        for seed in seeds:
            r = train_eval_2b(name, data, train_idx, val_idx, test_idx, epochs, seed)
            accs.append(r["accuracy"]); f1s.append(r["f1"]); infs.append(r["inference_ms"])
            cms.append(r["cm"])
        rows.append({"dataset": "BCI_IV_2b", "setting": "subject_dependent",
                     "model": name, "n_seeds": len(seeds),
                     "accuracy_mean": round(np.mean(accs), 4),
                     "accuracy_std": round(np.std(accs), 4),
                     "macro_f1_mean": round(np.mean(f1s), 4),
                     "macro_f1_std": round(np.std(f1s), 4),
                     "params": r["params"], "inference_ms": round(np.mean(infs), 4)})
    df = pd.DataFrame(rows)
    save_table(df, "comparison_2b")
    print("\n", df.to_string(index=False))
    return df


def experiment_2b_loso(epochs=config.EPOCHS, seeds=(config.SEED,)):
    data = load_all_subjects_2b(align=True)
    print(f"[2b-LOSO] trials={len(data['y'])} | seeds={list(seeds)} | EA=True")
    fold_rows = []
    for sid, tr, va, te in loso_folds_2b(data["subject"]):
        print(f"\n----- 2b LOSO fold: test subject {sid} -----")
        for name in ("eegnet", "stfa_light"):
            accs, f1s = [], []
            for seed in seeds:
                r = train_eval_2b(name, data, tr, va, te, epochs, seed, verbose=False)
                accs.append(r["accuracy"]); f1s.append(r["f1"])
                fold_rows.append({"test_subject": sid, "model": name, "seed": seed,
                                  "accuracy": round(r["accuracy"], 4),
                                  "macro_f1": round(r["f1"], 4)})
            print(f"  subject {sid} | {name:>10s} | acc={np.mean(accs):.4f} "
                  f"(±{np.std(accs):.4f})")
    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "cross_subject_folds_2b")
    per_fold = (fold_df.groupby(["model", "test_subject"], as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    summary = []
    for name in ("eegnet", "stfa_light"):
        sub = per_fold[per_fold["model"] == name]
        summary.append({"dataset": "BCI_IV_2b", "setting": "LOSO", "model": name,
                        "n_seeds": len(seeds), "EA": True,
                        "accuracy_mean": round(sub["accuracy"].mean(), 4),
                        "accuracy_std": round(sub["accuracy"].std(), 4),
                        "macro_f1_mean": round(sub["macro_f1"].mean(), 4),
                        "macro_f1_std": round(sub["macro_f1"].std(), 4)})
    df = pd.DataFrame(summary)
    save_table(df, "cross_subject_2b")
    print("\n", df.to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["compare", "loso", "all"])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = p.parse_args()
    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]
    if args.stage in ("compare", "all"):
        experiment_2b_compare(epochs=args.epochs, seeds=seeds)
    if args.stage in ("loso", "all"):
        experiment_2b_loso(epochs=args.epochs, seeds=seeds)

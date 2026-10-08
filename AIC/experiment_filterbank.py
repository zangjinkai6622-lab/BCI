# -*- coding: utf-8 -*-
"""滤波器组增强实验：LOSO + subject-dependent，对比原始 STFA-Net。

核心假设：显式 μ(8-12Hz)/β(13-30Hz) 频带分离能帮助跨被试泛化，
因为运动想象的 ERD/ERS 信号集中在这两个频带，而宽频 1-40Hz 混入了噪声。

用法: python experiment_filterbank.py --seeds 3 --epochs 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import pandas as pd
import torch

import config
from data.dataset import load_all_subjects_fb, loso_folds, make_loaders_fb
from models.stfa_fb import STFAFBNet
from models.stfa import STFANet
from data.dataset import load_all_subjects, make_loaders
from trainer import train_model, evaluate, count_parameters
from experiments import set_seed, save_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def experiment_loso_fb(epochs=config.EPOCHS, seeds=(config.SEED,)):
    """滤波器组版 LOSO。"""
    data = load_all_subjects_fb(align=True)
    print(f"[filterbank LOSO] X={data['X'].shape} | bands=[broad, mu, beta] | seeds={list(seeds)}")

    fold_rows = []
    for sid, tr, va, te in loso_folds(data["subject"]):
        print(f"\n----- fold: test subject {sid} -----")
        accs, f1s, infs = [], [], []
        for seed in seeds:
            set_seed(seed)
            tl, vl, tel, n_subj = make_loaders_fb(data, tr, va, te, loader_seed=seed)
            model = STFAFBNet(profile="standard")
            n_params = count_parameters(model)
            model, _ = train_model(model, tl, vl, epochs=epochs, dg=False,
                                   device=DEVICE, verbose=False)
            acc, m = evaluate(model, tel, dg=False, device=DEVICE, measure_time=True)
            accs.append(acc); f1s.append(m["f1"]); infs.append(m["inference_ms"])
            fold_rows.append({"test_subject": sid, "seed": seed,
                              "accuracy": round(acc, 4), "macro_f1": round(m["f1"], 4),
                              "inference_ms": round(m["inference_ms"], 4)})
        print(f"  subject {sid} | acc={np.mean(accs):.4f} (±{np.std(accs):.4f}) "
              f"| F1={np.mean(f1s):.4f} | infer={np.mean(infs):.3f}ms")

    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "filterbank_loso_folds")
    per_fold = (fold_df.groupby("test_subject", as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    summary = {"setting": "LOSO", "model": "stfa_fb",
               "n_bands": 3, "bands": "broad+mu+beta", "EA": True,
               "n_seeds": len(seeds),
               "accuracy_mean": round(per_fold["accuracy"].mean(), 4),
               "accuracy_std": round(per_fold["accuracy"].std(), 4),
               "macro_f1_mean": round(per_fold["macro_f1"].mean(), 4),
               "macro_f1_std": round(per_fold["macro_f1"].std(), 4),
               "params": n_params}
    df = pd.DataFrame([summary])
    save_table(df, "filterbank_loso")
    print("\n", df.to_string(index=False))
    return df


def experiment_subject_dep_fb(epochs=config.EPOCHS, seeds=(config.SEED,)):
    """滤波器组版 subject-dependent。"""
    from data.dataset import subject_dependent_split
    data = load_all_subjects_fb(align=False)
    print(f"[filterbank SD] X={data['X'].shape} | seeds={list(seeds)}")

    accs, f1s, infs = [], [], []
    for seed in seeds:
        set_seed(seed)
        tr, va, te = subject_dependent_split(data["y"], seed=seed)
        tl, vl, tel, _ = make_loaders_fb(data, tr, va, te, loader_seed=seed)
        model = STFAFBNet(profile="standard")
        n_params = count_parameters(model)
        model, _ = train_model(model, tl, vl, epochs=epochs, dg=False,
                               device=DEVICE, verbose=False)
        acc, m = evaluate(model, tel, dg=False, device=DEVICE, measure_time=True)
        accs.append(acc); f1s.append(m["f1"]); infs.append(m["inference_ms"])
        print(f"  seed {seed} | acc={acc:.4f} | F1={m['f1']:.4f}")

    summary = {"setting": "subject_dep", "model": "stfa_fb",
               "n_bands": 3, "bands": "broad+mu+beta", "EA": False,
               "n_seeds": len(seeds),
               "accuracy_mean": round(np.mean(accs), 4),
               "accuracy_std": round(np.std(accs), 4),
               "macro_f1_mean": round(np.mean(f1s), 4),
               "macro_f1_std": round(np.std(f1s), 4),
               "params": n_params}
    df = pd.DataFrame([summary])
    save_table(df, "filterbank_subject_dep")
    print("\n", df.to_string(index=False))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--mode", choices=["loso", "sd", "both"], default="both")
    args = p.parse_args()
    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]

    if args.mode in ("loso", "both"):
        experiment_loso_fb(epochs=args.epochs, seeds=seeds)
    if args.mode in ("sd", "both"):
        experiment_subject_dep_fb(epochs=args.epochs, seeds=seeds)

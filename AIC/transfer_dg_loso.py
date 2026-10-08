# -*- coding: utf-8 -*-
"""迁移 + 域对抗：2b 预训练 STFA 编码器 -> 2a STFA-DG 微调（LOSO）。

在 transfer_2b_to_2a.py 的迁移基础上，把 2a 模型换成 STFA-DG（带被试鉴别器），
看域对抗 + 预训练的组合能否进一步提升跨被试精度。
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
from data.dataset import load_all_subjects, loso_folds, make_loaders
from data.dataset_2b import load_all_subjects_2b, subject_dependent_split_2b, make_loaders_2b
from models.stfa import STFANet
from models.stfa_dg import STFADGNet
from trainer import train_model, evaluate, count_parameters, _lambda_schedule
from experiments import set_seed, save_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFER_KEYS = ["temporal", "temporal_attn", "feature_attn", "fusion"]


def pretrain_2b(epochs=60, seed=config.SEED):
    data_2b = load_all_subjects_2b(align=False)
    tr, va, te = subject_dependent_split_2b(data_2b["y"], seed=seed)
    tl, vl, _, _ = make_loaders_2b(data_2b, tr, va, te, loader_seed=seed)
    from data.dataset_2b import BCI2B_N_CHANNELS, BCI2B_N_CLASSES
    model = STFANet(n_channels=BCI2B_N_CHANNELS, n_classes=BCI2B_N_CLASSES, profile="standard")
    set_seed(seed)
    model, _ = train_model(model, tl, vl, epochs=epochs, dg=False, device=DEVICE, verbose=False)
    return model


def transfer_encoder_weights(model_2b, model_2a_dg):
    """把 2b STFA 编码器的可迁移权重拷到 2a STFA-DG 的编码器。"""
    sd_2b = model_2b.state_dict()
    sd_2a = model_2a_dg.state_dict()
    copied = 0
    for key in sd_2a:
        if not key.startswith("encoder."):
            continue
        enc_key = key[len("encoder."):]
        prefix = enc_key.split(".")[0]
        if prefix in TRANSFER_KEYS and enc_key in sd_2b:
            if sd_2b[enc_key].shape == sd_2a[key].shape:
                sd_2a[key] = sd_2b[enc_key].clone()
                copied += 1
    model_2a_dg.load_state_dict(sd_2a)
    return copied


def train_eval_transfer_dg(data_2a, train_idx, val_idx, test_idx, model_2b,
                            epochs=config.EPOCHS, seed=config.SEED, freeze_epochs=15):
    set_seed(seed)
    tl, vl, tel, n_subj = make_loaders(data_2a, train_idx, val_idx, test_idx, loader_seed=seed)
    model = STFADGNet(n_subjects=n_subj, n_channels=config.N_CHANNELS,
                      n_classes=config.N_CLASSES, profile="standard")
    copied = transfer_encoder_weights(model_2b, model)

    # 冻结 encoder.temporal，训练其余（含 DG 鉴别头）
    for name, p in model.named_parameters():
        if "encoder.temporal." in name:
            p.requires_grad = False

    model.to(DEVICE)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=freeze_epochs)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(freeze_epochs):
        model.train()
        for xb, yb, sb in tl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt1.zero_grad()
            lambd = _lambda_schedule(ep / max(freeze_epochs - 1, 1))
            task_logits, subj_logits = model(xb, lambd=lambd)
            loss = crit(task_logits, yb) + lambd * crit(subj_logits, sb)
            loss.backward(); opt1.step()
        sch1.step()
        acc, _ = evaluate(model, vl, dg=True, device=DEVICE)
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1

    # 解冻全部，全网络微调
    for p in model.parameters():
        p.requires_grad = True
    if best_state is not None:
        model.load_state_dict(best_state)
    opt2 = torch.optim.Adam(model.parameters(), lr=config.LR * 0.3, weight_decay=config.WEIGHT_DECAY)
    remain = max(epochs - freeze_epochs, 10)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=remain)
    best_acc2, best_state2, bad2 = 0.0, None, 0
    for ep in range(remain):
        model.train()
        for xb, yb, sb in tl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt2.zero_grad()
            lambd = _lambda_schedule((freeze_epochs + ep) / max(epochs - 1, 1))
            task_logits, subj_logits = model(xb, lambd=lambd)
            loss = crit(task_logits, yb) + lambd * crit(subj_logits, sb)
            loss.backward(); opt2.step()
        sch2.step()
        acc, _ = evaluate(model, vl, dg=True, device=DEVICE)
        if acc > best_acc2:
            best_acc2 = acc; best_state2 = copy.deepcopy(model.state_dict()); bad2 = 0
        else:
            bad2 += 1
        if bad2 >= config.EARLY_STOP_PATIENCE:
            break
    if best_state2 is not None:
        model.load_state_dict(best_state2)
    acc, m = evaluate(model, tel, dg=True, device=DEVICE, measure_time=True)
    return {"accuracy": acc, "f1": m["f1"], "cm": m["cm"],
            "inference_ms": m["inference_ms"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--freeze", type=int, default=15)
    args = p.parse_args()
    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]

    data_2a = load_all_subjects(align=True)
    print(f"[transfer+DG LOSO] 2a trials={len(data_2a['y'])} | seeds={list(seeds)} | EA=True")

    fold_rows = []
    for sid, tr, va, te in loso_folds(data_2a["subject"]):
        print(f"\n----- fold: test subject {sid} -----")
        accs = []
        for seed in seeds:
            model_2b = pretrain_2b(epochs=60, seed=seed)
            r = train_eval_transfer_dg(data_2a, tr, va, te, model_2b,
                                       epochs=args.epochs, seed=seed, freeze_epochs=args.freeze)
            accs.append(r["accuracy"])
            fold_rows.append({"test_subject": sid, "seed": seed,
                              "accuracy": round(r["accuracy"], 4),
                              "macro_f1": round(r["f1"], 4)})
        print(f"  subject {sid} | acc={np.mean(accs):.4f} (±{np.std(accs):.4f})")

    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "transfer_dg_loso_folds")
    per_fold = (fold_df.groupby("test_subject", as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean")))
    summary = {"setting": "LOSO_transfer_DG", "model": "stfa_dg_from_2b",
               "n_seeds": len(seeds), "EA": True,
               "accuracy_mean": round(per_fold["accuracy"].mean(), 4),
               "accuracy_std": round(per_fold["accuracy"].std(), 4),
               "macro_f1_mean": round(per_fold["macro_f1"].mean(), 4),
               "macro_f1_std": round(per_fold["macro_f1"].std(), 4)}
    df = pd.DataFrame([summary])
    save_table(df, "transfer_dg_loso")
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()

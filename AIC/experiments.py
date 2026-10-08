"""实验编排：模型对比 / 消融实验 / 跨被试(LOSO) / 注意力可视化，结果统一落盘 results/。

公平性原则：
  - 所有模型共用同一套数据划分（固定 seed=42）；
  - 标准化统计量只在训练集拟合；测试集全程不参与训练/选模型；
  - 先固定划分 -> 跑 baseline -> 跑 EEGNet -> 逐个加模块，每步记录结果。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import config
from data.dataset import (
    load_all_subjects, subject_dependent_split, loso_folds, make_loaders,
)
from baselines import run_ml_baselines
from models.eegnet import EEGNet
from models.stfa import STFANet
from models.stfa_dg import STFADGNet
from trainer import train_model, evaluate, count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================ 工具函数
def set_seed(seed=config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_subjects(data, n_subjects):
    """冒烟测试用：只取前 n 个被试。"""
    keep = data["subject"] <= n_subjects
    return {"X": data["X"][keep], "y": data["y"][keep],
            "subject": data["subject"][keep], "channels": data["channels"]}


DL_MODEL_NAMES = ["eegnet", "stfa", "stfa_dg", "stfa_light", "stfa_dg_light"]


def build_dl_model(name, n_subjects, ablation_kwargs=None):
    """name: eegnet / stfa / stfa_dg / stfa_light / stfa_dg_light。

    stfa_light / stfa_dg_light 使用 config 的 light 容量档（约 5 千参数，
    与 EEGNet 同量级）；ablation_kwargs 透传给 STFA 系列。
    """
    kw = ablation_kwargs or {}
    if name == "eegnet":
        # 裁剪训练时输入长度为 CROP_SIZE，分类头展平维度随之改变
        n_samples = config.CROP_SIZE if config.USE_CROPPED else config.WIN_SAMPLES
        return EEGNet(n_samples=n_samples), False
    if name in ("stfa", "stfa_light"):
        profile = "light" if name == "stfa_light" else "standard"
        return STFANet(profile=profile, **kw), False
    if name in ("stfa_dg", "stfa_dg_light"):
        profile = "light" if name == "stfa_dg_light" else "standard"
        return STFADGNet(n_subjects=n_subjects, profile=profile, **kw), True
    raise ValueError(f"未知模型: {name}，可选: {DL_MODEL_NAMES}")


def aggregate_seed_results(rs):
    """同一种模型多种子重复结果的聚合：mean±std + 混淆矩阵跨种子平均。"""
    accs = [r["accuracy"] for r in rs]
    f1s = [r["f1"] for r in rs]
    infs = [r["inference_ms"] for r in rs]
    return {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s)), 4),
        "inference_ms": round(float(np.mean(infs)), 4),
        "cm": np.round(np.mean([r["cm"] for r in rs], axis=0), 1),
        "params": rs[0]["params"],
    }


def train_and_eval_dl(name, data, train_idx, val_idx, test_idx,
                      epochs=config.EPOCHS, ablation_kwargs=None, verbose=True,
                      seed=config.SEED):
    """在给定划分上训练一个深度学习模型并在测试集评估。

    seed 同时控制权重初始化与 DataLoader shuffle（后者经独立 Generator 播种，
    保证同一 seed 下不同模型见到相同 batch 顺序）。
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, n_subjects = make_loaders(
        data, train_idx, val_idx, test_idx, loader_seed=seed)
    model, dg = build_dl_model(name, n_subjects, ablation_kwargs)
    n_params = count_parameters(model)
    if verbose:
        print(f"[DL] {name} | params={n_params} | subjects(train)={n_subjects} | device={DEVICE}")

    model, hist = train_model(model, train_loader, val_loader,
                              epochs=epochs, dg=dg, device=DEVICE, verbose=verbose)
    acc, m = evaluate(model, test_loader, dg=dg, device=DEVICE, measure_time=True)
    # v2：导出最终自适应门控强度（无门控的模型/消融配置返回 {}）
    gates = {}
    enc = getattr(model, "encoder", model)
    if hasattr(enc, "gate_values"):
        gates = enc.gate_values()
    if verbose:
        print(f"[DL] {name} | TEST acc={acc:.4f} macro-F1={m['f1']:.4f} "
              f"infer={m['inference_ms']:.3f} ms/sample"
              + (f" | gates={ {k: round(v, 3) for k, v in gates.items()} }" if gates else ""))
    return {
        "model": model, "dg": dg, "test_loader": test_loader,
        "accuracy": acc, "f1": m["f1"], "cm": m["cm"],
        "params": n_params, "inference_ms": m["inference_ms"],
        "val_acc": hist["best_val_acc"], "gates": gates,
    }


def _df_to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_table(df, name):
    df.to_csv(config.RESULTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    with open(config.RESULTS_DIR / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(_df_to_md(df) + "\n")
    print(f"[result] {name}: {config.RESULTS_DIR / (name + '.csv')}")


def save_confusion(cm, tag):
    out = config.RESULTS_DIR / "confusion"
    out.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in config.CLASS_NAMES],
        columns=[f"pred_{n}" for n in config.CLASS_NAMES],
    )
    cm_df.to_csv(out / f"{tag}.csv", encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(config.N_CLASSES), config.CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(config.N_CLASSES), config.CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(config.N_CLASSES):
            for j in range(config.N_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black" if cm[i, j] < cm.max() * 0.6 else "white")
        ax.set_title(tag)
        fig.tight_layout()
        fig.savefig(out / f"{tag}.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] 混淆矩阵图未生成（matplotlib 不可用？）: {e}")


@torch.no_grad()
def collect_channel_attention(model, loader, dg, device=DEVICE):
    """在测试集 loader（已用训练集统计量标准化）上收集通道注意力平均权重。

    返回 (22,) numpy（脑区重要性）；无通道注意力的模型（如 EEGNet）返回 None。
    """
    model.eval()
    ws = []
    for xb, _, _ in loader:
        xb = xb.to(device)
        if xb.dim() == 5:                    # 滑窗评估 loader：(B,K,1,C,L)
            xb = xb.reshape(xb.shape[0] * xb.shape[1], *xb.shape[2:])
        if dg:
            _, _, aux = model(xb, return_aux=True)
        else:
            _, aux = model(xb, return_aux=True)
        if "channel_weight" in aux:
            ws.append(aux["channel_weight"].cpu().numpy())
    if not ws:
        return None
    return np.concatenate(ws, axis=0).mean(axis=0)


def save_channel_attention(weights, channels, tag):
    if weights is None:
        return
    df = pd.DataFrame({"channel": list(channels), "attention_weight": weights})
    df = df.sort_values("attention_weight", ascending=False).reset_index(drop=True)
    df.to_csv(config.RESULTS_DIR / f"{tag}.csv", index=False, encoding="utf-8-sig")
    print("[attention] 通道重要性 Top5:")
    for _, r in df.head(5).iterrows():
        print(f"  {r['channel']:>4s}  {r['attention_weight']:.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(weights)[::-1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.array(channels)[order], weights[order], color="steelblue")
        ax.set_ylabel("attention weight"); ax.set_title(f"Channel attention ({tag})")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / f"{tag}.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass


# 被试内正式对比链：EEGNet 基线 -> STFA-Net 主模型 -> STFA-DGNet 跨被试对照。
# light 档是 LOSO 专用容量档，不进被试内主表（需要时用 --models 显式指定）。
COMPARISON_MODELS = ("eegnet", "stfa", "stfa_dg")


# ============================================================ 实验①：模型对比（subject-dependent）
def experiment_comparison(epochs=config.EPOCHS, n_subjects=None, seeds=(config.SEED,),
                          model_names=COMPARISON_MODELS):
    # subject-dependent：不做 EA（测试被试训练时见过，无需跨被试对齐）
    data = load_all_subjects(align=False)
    if n_subjects:
        data = subset_subjects(data, n_subjects)
    print(f"[experiment] 模型对比 | trials={len(data['y'])} | seeds={list(seeds)} | "
          f"DL models={list(model_names)} | device={DEVICE}")

    train_idx, val_idx, test_idx = subject_dependent_split(data["y"])
    rows = []

    # ---- 传统 ML 基线（手工特征；确定性算法，不随种子变化，n_seeds=1）----
    print("\n=== Traditional ML Baseline (handcrafted features) ===")
    ml_res = run_ml_baselines(data["X"], data["y"], train_idx, test_idx)
    for name, r in ml_res.items():
        save_confusion(r["cm"], f"comparison_{name}")
        rows.append({"setting": "subject_dependent", "model": name.upper(),
                     "n_seeds": 1,
                     "accuracy_mean": round(r["accuracy"], 4), "accuracy_std": 0.0,
                     "macro_f1_mean": round(r["f1"], 4), "macro_f1_std": 0.0,
                     "params": r["params"], "inference_ms": round(r["inference_ms"], 4)})

    # ---- 深度学习模型（默认核心三模型；--models 可扩展到 light 档）----
    gate_rows = []
    for name in model_names:
        print(f"\n=== {name} ===")
        rs = []
        for seed in seeds:
            r = train_and_eval_dl(name, data, train_idx, val_idx, test_idx,
                                  epochs=epochs, seed=seed)
            rs.append(r)
            for gate_name, alpha in r["gates"].items():
                gate_rows.append({"model": name, "seed": seed,
                                  "gate": gate_name, "alpha": round(alpha, 6)})
        agg = aggregate_seed_results(rs)
        save_confusion(agg["cm"], f"comparison_{name}")
        rows.append({"setting": "subject_dependent", "model": name,
                     "n_seeds": len(seeds),
                     "accuracy_mean": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
                     "macro_f1_mean": agg["macro_f1_mean"], "macro_f1_std": agg["macro_f1_std"],
                     "params": agg["params"], "inference_ms": agg["inference_ms"]})
        if name in ("stfa", "stfa_dg", "stfa_light", "stfa_dg_light"):
            # 通道注意力取最后一个种子的模型（权重结构各种子一致）
            w = collect_channel_attention(rs[-1]["model"], rs[-1]["test_loader"], rs[-1]["dg"])
            save_channel_attention(w, data["channels"], f"attention_{name}")

    df = pd.DataFrame(rows)
    save_table(df, "comparison")
    if gate_rows:
        gate_df = pd.DataFrame(gate_rows)
        gate_df.to_csv(config.RESULTS_DIR / "gate_values.csv", index=False, encoding="utf-8-sig")
        print(f"[result] gate_values: {config.RESULTS_DIR / 'gate_values.csv'} "
              f"（v2 自适应门控强度，α≈0 = 模块被自适应抑制）")
    print("\n", df.to_string(index=False))
    return df


# ============================================================ 实验②：消融实验
# 完整 STFA-Net -> 去多尺度时间 -> 去空间 -> 去频谱 -> 去注意力，
# 末行 stfa_dg_ref 为跨被试对抗(GRL) 参照（其价值由实验③ LOSO 裁决）。
# 模块消融以 subject-dependent 正式模型 STFA-Net 为基线：
# GRL 只在"测试被试未见过"(LOSO) 场景有意义，用它当消融基线会污染模块贡献判断。
ABLATION_STEPS = [
    ("stfa_full",          "stfa",    {}),                          # 完整 STFA-Net
    ("wo_multiscale_time", "stfa",    {"use_multiscale": False}),   # 去多尺度时间
    ("wo_spatial",         "stfa",    {"use_spatial": False}),      # 去空间卷积
    ("wo_frequency",       "stfa",    {"use_frequency": False}),    # 去频谱分支
    ("wo_attention",       "stfa",    {"use_attention": False}),    # 去三重注意力
    # 跨被试对抗(GRL) 的对照行：其价值不由 subject-dependent 消融裁决，
    # 而由实验③ LOSO（stfa vs stfa_dg 在未见被试上）判定。
    ("stfa_dg_ref",        "stfa_dg", {}),
]


def experiment_ablation(epochs=config.EPOCHS, n_subjects=None, seeds=(config.SEED,)):
    # subject-dependent：不做 EA，与对比实验同口径
    data = load_all_subjects(align=False)
    if n_subjects:
        data = subset_subjects(data, n_subjects)
    print(f"[experiment] 消融实验 | trials={len(data['y'])} | seeds={list(seeds)}")

    train_idx, val_idx, test_idx = subject_dependent_split(data["y"])
    rows = []
    for tag, model_name, kw in ABLATION_STEPS:
        print(f"\n=== ablation: {tag} ({model_name}, {kw or 'full'}) ===")
        rs = []
        for seed in seeds:
            rs.append(train_and_eval_dl(model_name, data, train_idx, val_idx, test_idx,
                                        epochs=epochs, ablation_kwargs=kw, seed=seed))
        agg = aggregate_seed_results(rs)
        save_confusion(agg["cm"], f"ablation_{tag}")
        rows.append({"ablation": tag, "model": model_name,
                     "n_seeds": len(seeds),
                     "accuracy_mean": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
                     "macro_f1_mean": agg["macro_f1_mean"], "macro_f1_std": agg["macro_f1_std"],
                     "params": agg["params"], "inference_ms": agg["inference_ms"]})

    df = pd.DataFrame(rows)
    save_table(df, "ablation")
    print("\n", df.to_string(index=False))
    return df


# ============================================================ 实验③：跨被试 Subject-Independent (LOSO)
def experiment_loso(epochs=config.EPOCHS, model_names=("eegnet", "stfa", "stfa_dg"),
                    data=None, seeds=(config.SEED,)):
    """LOSO 跨被试实验。seeds 给多个种子时，每个 fold×model 重复训练多次，

    先对同一 fold 的多种子结果取平均，再在 9 个 fold 上统计 mean±std
    （单 seed 的 LOSO 被试间方差高达 33%~70%，必须带误差棒结论才站得住）。
    """
    data = data if data is not None else load_all_subjects()
    n_folds = len(np.unique(data["subject"]))
    print(f"[experiment] 跨被试 LOSO | {n_folds} folds | seeds={list(seeds)} | "
          f"EA={config.USE_EA} | device={DEVICE}")

    fold_rows = []
    for sid, train_idx, val_idx, test_idx in loso_folds(data["subject"]):
        print(f"\n----- LOSO fold: test subject = {sid} -----")
        for name in model_names:
            accs, f1s, infs = [], [], []
            for seed in seeds:
                r = train_and_eval_dl(name, data, train_idx, val_idx, test_idx,
                                      epochs=epochs, verbose=False, seed=seed)
                accs.append(r["accuracy"])
                f1s.append(r["f1"])
                infs.append(r["inference_ms"])
                fold_rows.append({"test_subject": sid, "model": name, "seed": seed,
                                  "accuracy": round(r["accuracy"], 4),
                                  "macro_f1": round(r["f1"], 4),
                                  "inference_ms": round(r["inference_ms"], 4)})
            print(f"  subject {sid} | {name:>7s} | "
                  f"acc={np.mean(accs):.4f} (±{np.std(accs):.4f}) "
                  f"| macro-F1={np.mean(f1s):.4f}")

    fold_df = pd.DataFrame(fold_rows)
    save_table(fold_df, "cross_subject_folds")

    # 先按 fold 聚合多种子，再在 fold 间统计（两层口径不能混）
    per_fold = (fold_df.groupby(["model", "test_subject"], as_index=False)
                .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                     inference_ms=("inference_ms", "mean")))

    summary_rows = []
    for name in model_names:
        sub = per_fold[per_fold["model"] == name]
        summary_rows.append({
            "setting": "subject_independent_LOSO", "model": name,
            "n_seeds": len(seeds), "EA": config.USE_EA,
            "accuracy_mean": round(sub["accuracy"].mean(), 4),
            "accuracy_std": round(sub["accuracy"].std(), 4),
            "macro_f1_mean": round(sub["macro_f1"].mean(), 4),
            "macro_f1_std": round(sub["macro_f1"].std(), 4),
            "inference_ms": round(sub["inference_ms"].mean(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, "cross_subject")
    print("\n", summary_df.to_string(index=False))
    return summary_df

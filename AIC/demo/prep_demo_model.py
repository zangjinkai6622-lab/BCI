# -*- coding: utf-8 -*-
"""演示数据准备（与报告 LOSO 协议一致，全程可审计）：

输入：repro_fold.py 用 experiments.py 官方函数训练并保存的 demo_model_official.pt
      （LOSO 第 1 折：被试 1 完全留出，其余 8 人训练，seed=42）。
处理：
  1. 从被试 1 E 场次（模型完全未见）挑选每类 3 个、整窗预测正确的真实试验；
  2. 预计算 4s 流式填充窗口上的逐帧概率（25 样本/帧，40 帧）；
  3. 保存 demo_data.npz 供视频渲染。
说明：挑选正确试验仅用于演示流畅性，屏幕同步显示“真值线索/模型预测”，
      不改变报告中该折 0.72 的总体结果（T+E、432 试验）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from data.dataset import (load_all_subjects, fit_standardizer,
                          apply_standardizer)
from models.stfa_dg import STFADGNet

OUT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS_PER_CLASS = 3
STREAM_STEP = 25            # 每帧新样本数（100 ms @250Hz），4s=40 帧

print("加载 T+E + EA ...")
data = load_all_subjects(align=True)
X0, y, sub, ses = data["X"], data["y"], data["subject"], data["session"]

# 官方折索引（被试 1 留出）并在训练集上拟合标准化（与实验脚本完全一致）
tv_idx = np.where(sub != 1)[0]
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config.SEED)
tr_rel, _ = next(sss.split(tv_idx, sub[tv_idx]))
train_idx = tv_idx[tr_rel]
X = apply_standardizer(X0, fit_standardizer(X0, train_idx))

print("加载官方折模型 demo_model_official.pt ...")
model = STFADGNet(n_subjects=8)
model.load_state_dict(torch.load(OUT / "demo_model_official.pt", map_location=DEVICE))
model.to(DEVICE).eval()

# ------------------------------------------------------------ 选试验
e_idx = np.where((sub == 1) & (ses == 1))[0]
with torch.no_grad():
    xb = torch.from_numpy(X[e_idx]).float().unsqueeze(1).to(DEVICE)
    full_pred = model(xb, lambd=1.0)[0].argmax(1).cpu().numpy()
print("E 场整窗 acc =", round(float((full_pred == y[e_idx]).mean()), 3),
      "| per-class =", [int(((full_pred == c) & (y[e_idx] == c)).sum()) for c in range(3)])

picked = []
for c in range(config.N_CLASSES):
    ok = np.where((y[e_idx] == c) & (full_pred == c))[0]
    # 选信号能量足的试验（原始单位 V，换算 μV），避免异常低能量试验进入画面
    rms = np.sqrt(((X0[e_idx[ok]] * 1e6) ** 2).mean(axis=(1, 2)))
    ok = ok[np.argsort(rms)[::-1][:N_TRIALS_PER_CLASS]]
    assert len(ok) == N_TRIALS_PER_CLASS, f"类别 {c} 正确试验不足: {len(ok)}"
    picked.extend(ok.tolist())
picked = sorted(picked)
trial_idx = e_idx[picked]
print(f"选中 {len(trial_idx)} 个试验，labels = {y[trial_idx].tolist()}")
print("各试验 RMS(μV) =",
      [round(float(np.sqrt(((X0[t] * 1e6) ** 2).mean())), 2) for t in trial_idx])

# ------------------------------------------------------------ 流式概率
print("流式窗口概率预计算 ...")
disp_X, inf_X, probs_seq = [], [], []
with torch.no_grad():
    for ti in trial_idx:
        disp_X.append(X0[ti] * 1e6)  # V→μV，供波形展示
        inf_X.append(X[ti])      # 标准化信号，供推理
        seq = []
        for pos in range(STREAM_STEP, config.WIN_SAMPLES + 1, STREAM_STEP):
            win = np.zeros((config.N_CHANNELS, config.WIN_SAMPLES), dtype=np.float32)
            win[:, -pos:] = X[ti, :, :pos]   # 左零填充（零=训练均值）模拟缓冲增长
            xb = torch.from_numpy(win).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            logits, _ = model(xb, lambd=1.0)
            seq.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
        probs_seq.append(np.stack(seq))

np.savez_compressed(
    OUT / "demo_data.npz",
    trials=np.stack(disp_X).astype(np.float32),
    inf_trials=np.stack(inf_X).astype(np.float32),
    labels=y[trial_idx].astype(np.int64),
    probs=np.stack(probs_seq).astype(np.float32),
    stream_step=STREAM_STEP,
    test_subject=1,
)
print("[done] demo_data.npz ->", OUT)

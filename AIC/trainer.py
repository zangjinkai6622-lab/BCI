"""训练与评估：统一的训练循环（Adam + Cosine LR + 早停）、指标计算。

指标：Accuracy / Macro-F1 / 参数量 / 单样本推理耗时(ms) / 混淆矩阵。
跨被试对抗模型（dg=True）额外训练被试鉴别器，lambda 按 DANN 调度 warmup。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config


def count_parameters(model):
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _lambda_schedule(progress, max_lambda=config.DG_LAMBDA_MAX):
    """DANN 对抗权重调度：progress∈[0,1] -> lambda 从 0 平滑升到 max_lambda。"""
    return max_lambda * 2.0 / (1.0 + np.exp(-10.0 * progress)) - max_lambda


def _model_forward(model, x, dg, lambd):
    """兼容普通模型（返回 logits）与 DG 模型（返回 task_logits, subj_logits）。"""
    if dg:
        task_logits, subj_logits = model(x, lambd=lambd)
        return task_logits, subj_logits
    return model(x), None


def train_model(model, train_loader, val_loader,
                epochs=config.EPOCHS, lr=config.LR,
                weight_decay=config.WEIGHT_DECAY,
                patience=config.EARLY_STOP_PATIENCE,
                dg=False, device="cpu", verbose=True):
    """训练模型，按验证集准确率选最优权重（早停）。返回 (model, history)。"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            optimizer.zero_grad()

            lambd = _lambda_schedule(epoch / max(epochs - 1, 1)) if dg else 0.0
            logits, subj_logits = _model_forward(model, xb, dg, lambd)
            loss = criterion(logits, yb)
            if dg:
                loss = loss + lambd * criterion(subj_logits, sb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        scheduler.step()

        val_acc, _ = evaluate(model, val_loader, dg=dg, device=device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (epoch % 5 == 0 or bad_epochs == 0):
            tag = f"lambda={lambd:.3f}" if dg else ""
            print(f"  epoch {epoch + 1:3d}/{epochs} | train_loss={running_loss / n_seen:.4f} "
                  f"| val_acc={val_acc:.4f} | best={best_acc:.4f} {tag}")

        if bad_epochs >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch + 1} (best val_acc={best_acc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_acc": float(best_acc)}


@torch.no_grad()
def evaluate(model, loader, dg=False, device="cpu", measure_time=False):
    """在给定 loader 上评估，返回 (accuracy, metrics_dict)。"""
    model.eval()
    y_true, y_pred = [], []
    total_time, n_samples, warmup = 0.0, 0, True

    for xb, yb, _sb in loader:
        xb = xb.to(device)
        if measure_time:
            t0 = time.perf_counter()

        if xb.dim() == 5:
            # 滑窗评估：(B,K,1,C,L) -> 每窗口 softmax -> 窗口间概率平均（软投票）
            B, K = xb.shape[:2]
            x_flat = xb.reshape(B * K, *xb.shape[2:])
            win_logits, _ = _model_forward(model, x_flat, dg, lambd=1.0)
            probs = torch.softmax(win_logits, dim=1).view(B, K, -1).mean(dim=1)
        else:
            logits, _ = _model_forward(model, xb, dg, lambd=1.0)
            probs = torch.softmax(logits, dim=1)

        if measure_time:
            dt = time.perf_counter() - t0
            if not warmup:          # 首个 batch 含编译/缓存开销，不计入
                total_time += dt
                n_samples += xb.size(0)
            warmup = False

        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(config.N_CLASSES))),
        "inference_ms": float(total_time / max(n_samples, 1) * 1000.0) if measure_time else None,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return float(accuracy_score(y_true, y_pred)), metrics

"""传统机器学习基线：手工特征 -> StandardScaler -> PCA(99% 方差) -> LR/RF/SVM。

与深度学习使用完全相同的 trial 划分（同一 train/test 索引、同一 seed），
保证 "手工特征 + 传统ML" 与 "端到端深度学习" 的对比是公平的。
网格搜索仅在训练集内部做 5 折交叉验证，测试集全程不参与。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config
from data.features import extract_epoch_features

# (分类器, 网格) —— 与 EEG Explorer/machine_learning.py 同一思路，网格适度收窄以控时长
ML_GRIDS = {
    "lr": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.1, 1, 10]},
    ),
    "rf": (
        RandomForestClassifier(random_state=config.SEED),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [None, 10, 20]},
    ),
    "svm": (
        SVC(kernel="rbf"),
        {"model__C": [0.1, 1, 10, 100],
         "model__gamma": ["scale", 0.01, 0.001]},
    ),
}


def run_ml_baselines(X, y, train_idx, test_idx, verbose=True):
    """在指定划分上跑 LR/RF/SVM，返回 {model_name: metrics_dict}。

    metrics: accuracy / f1(macro) / cm / params / inference_ms / best_params
    """
    feats = extract_epoch_features(X)
    X_train, X_test = feats[train_idx], feats[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    for name, (clf, grid) in ML_GRIDS.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99)),
            ("model", clf),
        ])
        gs = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=1)
        t0 = time.perf_counter()
        gs.fit(X_train, y_train)
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = gs.predict(X_test)
        infer_ms = (time.perf_counter() - t0) / len(X_test) * 1000.0

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="macro")),
            "cm": confusion_matrix(y_test, y_pred,
                                   labels=list(range(config.N_CLASSES))),
            "params": "-",
            "inference_ms": float(infer_ms),
            "best_params": gs.best_params_,
            "train_s": float(train_s),
        }
        if verbose:
            print(f"[ML] {name:>3s} | acc={results[name]['accuracy']:.4f} "
                  f"| macro-F1={results[name]['f1']:.4f} "
                  f"| {gs.best_params_}")
    return results

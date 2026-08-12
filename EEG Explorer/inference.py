import machine_learning
import logging
logger = logging.getLogger(__name__)
import data_pipeline
import numpy as np
import label
import prediction as pred
import pathlib
import config
import pandas as pd


def predict_file(file_path, model_name: str = config.DEFAULT_MODEL):
    """
    逐 trial 推理：
        A09T.gdf
          ↓
        216 trials（每个 trial 一个窗口）
          ↓
        216 特征样本
          ↓
        模型逐 trial 预测（不再全文件投票）
          ↓
        返回 216 行详细结果（含真值 / 预测 / 是否正确）
    """
    file_stem = pathlib.Path(file_path).name

    # 1. 特征提取：feature_df = (216 行 × 593 列)，含 "label" 列（真值）
    feature_df = data_pipeline.extract_feature(file_path)
    n_trials = len(feature_df)

    # 2. 模型预测：对 n_trials 个样本一次性预测，得到 (n_trials,) 数组
    predictions = machine_learning.predict_one_sample(feature_df, model_name)

    # 3. 逐 trial 组装明细（从 feature_df 里拿出对应的 event 信息暂缺，用索引代替，留扩展位）
    true_labels = feature_df["label"].astype(int).tolist()
    details = []
    for trial_idx in range(n_trials):
        true_lbl = true_labels[trial_idx]
        pred_lbl = int(predictions[trial_idx])
        true_name = label.decode_label(true_lbl)
        pred_name = label.decode_label(pred_lbl)
        correct = (true_lbl == pred_lbl)
        details.append({
            "file": file_stem,
            "trial_idx": trial_idx,
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "true_label_name": true_name,
            "pred_label_name": pred_name,
            "correct": correct,
        })

    # 4. 统计汇总
    n_correct = sum(1 for d in details if d["correct"])
    accuracy = n_correct / n_trials * 100 if n_trials > 0 else 0.0
    pred_dist = pd.Series([d["pred_label_name"] for d in details]).value_counts()
    # 按类别分别统计准确率
    per_class_acc = {}
    for lbl_int, lbl_name in sorted(label.LABEL_NAME.items()):
        subset = [d for d in details if d["true_label"] == lbl_int]
        if subset:
            acc_cls = sum(1 for d in subset if d["correct"]) / len(subset) * 100
            per_class_acc[lbl_name] = (len(subset), acc_cls)

    # 5. 控制台打印：逐 trial 一行（紧凑格式）+ 尾部汇总
    print("\n" + "=" * 90)
    print(f"Inference Report | File: {file_stem} | Model: {model_name} | Total Trials: {n_trials}")
    print("=" * 90)
    print(f"{'#':>4}  {'True':<10}  {'Pred':<10}  {'Match'}")
    print("-" * 90)
    for d in details:
        mark = "✅" if d["correct"] else "❌"
        print(
            f"{d['trial_idx']:>4}  "
            f"{d['true_label_name']:<10}  "
            f"{d['pred_label_name']:<10}  "
            f"{mark}"
        )
    print("=" * 90)
    print(f"Correct: {n_correct} / {n_trials}  |  Overall Accuracy: {accuracy:.2f}%")
    print("-" * 90)
    print("Per-Class Accuracy:")
    for cls_name, (n_total, acc_cls) in per_class_acc.items():
        # 对应 pred_label_name 的预测数量
        n_pred = pred_dist.get(cls_name, 0)
        print(f"  {cls_name:<10}  true_count={n_total:<4}  pred_count={n_pred:<4}  acc={acc_cls:.2f}%")
    print("=" * 90 + "\n")

    # 6. 保存 md 明细报告 + csv 明细表
    pred.save_prediction_details_md(
        file_name=file_path,
        model_name=model_name,
        details=details,
        n_correct=n_correct,
        accuracy=accuracy,
        per_class_acc=per_class_acc,
        pred_dist=pred_dist,
    )
    pred.save_prediction_csv(details)

    # 7. 逐 trial 结果返回（不再是一个单一 "both_feet" 字符串，而是 216 条明细）
    return {
        "file": file_stem,
        "model": model_name,
        "n_trials": n_trials,
        "n_correct": n_correct,
        "accuracy_pct": round(accuracy, 2),
        "details": details,
    }


if __name__ == "__main__":
    file = "EEG Explorer/data/A09T.gdf"
    result = predict_file(file)
    # 简化输出（不在只打印一个类别）：打印一行汇总，明细已在函数内部打出
    print(
        f"【Result】File={result['file']}  Trials={result['n_trials']}  "
        f"Correct={result['n_correct']}  Accuracy={result['accuracy_pct']}%"
    )

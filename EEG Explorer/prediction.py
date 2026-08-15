import config
import pandas as pd
import pathlib


def save_prediction_md(file_name: str, model_name: str, prediction: str):
    """单结果预测报告（向后兼容，旧接口保留）"""
    save_dir = config.PREDICTION_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    file_stem = pathlib.Path(file_name).stem
    save_path = save_dir / f"{file_stem}.md"
    with open(save_path, "w", encoding="utf8") as f:
        f.write("# Prediction Report\n\n")
        f.write("## File\n")
        f.write(f"{file_name}\n\n")
        f.write("## Model\n")
        f.write(f"{model_name}\n\n")
        f.write("## Prediction\n")
        f.write(f"{prediction}\n")
        f.write("\n")
        f.write("## Time\n")
        f.write(f"{pd.Timestamp.now()}\n")


def save_prediction_csv(results: list):

    df = pd.DataFrame(results)
    save_path = config.PREDICTION_DIR / "prediction.csv"
    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )
    return save_path


def save_prediction_details_md(
    file_name: str,
    model_name: str,
    details: list,
    n_correct: int,
    accuracy: float,
    per_class_acc: dict,
    pred_dist: pd.Series,
):

    save_dir = config.PREDICTION_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    file_stem = pathlib.Path(file_name).stem
    save_path = save_dir / f"{file_stem}.md"

    n_trials = len(details)

    with open(save_path, "w", encoding="utf8") as f:
        f.write("# Inference Report (Per-Trial)\n\n")
        f.write("## File\n")
        f.write(f"`{file_name}`\n\n")
        f.write("## Model\n")
        f.write(f"`{model_name}`\n\n")
        f.write("## Trials\n")
        f.write(f"Total: **{n_trials}** trials\n\n")
        f.write("## Summary\n")
        f.write(f"- Correct Predictions: **{n_correct} / {n_trials}**\n")
        f.write(f"- Overall Accuracy: **{accuracy:.2f}%**\n\n")

        f.write("### Per-Class Accuracy\n")
        f.write("| Class | True Count | Pred Count | Accuracy |\n")
        f.write("|-------|-----------:|-----------:|---------:|\n")
        # 遍历 details 里可能出现的所有类别名（per_class_acc 的 key 是真值存在的类）
        all_names = set(per_class_acc.keys()) | set(pred_dist.index.tolist())
        for cls_name in sorted(all_names):
            true_count, acc_cls = per_class_acc.get(cls_name, (0, 0.0))
            pred_count = int(pred_dist.get(cls_name, 0))
            acc_str = f"{acc_cls:.2f}%" if true_count > 0 else "N/A"
            f.write(f"| {cls_name} | {true_count} | {pred_count} | {acc_str} |\n")
        f.write("\n")

        f.write("### Prediction Distribution (Predicted Counts)\n")
        f.write("| Class | Count |\n")
        f.write("|-------|------:|\n")
        for cls_name, cnt in pred_dist.items():
            f.write(f"| {cls_name} | {int(cnt)} |\n")
        f.write("\n")

        f.write("## Per-Trial Details\n")
        f.write("| # | True Label | Pred Label | Match |\n")
        f.write("|--:|------------|------------|-------|\n")
        for d in details:
            trial_idx = d["trial_idx"]
            true_name = d["true_label_name"]
            pred_name = d["pred_label_name"]
            match = "✅" if d["correct"] else "❌"
            f.write(f"| {trial_idx} | {true_name} | {pred_name} | {match} |\n")
        f.write("\n")

        f.write("## Time\n")
        f.write(f"{pd.Timestamp.now()}\n")

    return save_path

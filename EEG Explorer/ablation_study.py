"""
Ablation Study（特征消融实验）
==============================
目标：固定「模型=SVM + train/test划分」，只改变特征子集，看哪类特征对准确率贡献最大。

四个实验（固定模型 SVM + PCA 99% + 统一的 train/test 行索引）：
    A. Time Domain （时域）     → 列名含 "_time_domain_"
    B. Frequency   （频域）     → 列名含 _peak_frequency / _band_power_ / _relative_band_power_ / _band_ratio_
    C. Time + Freq （时域+频域）→ A ∪ B （排除 Hjorth / 熵 / 不对称）
    D. All         （全部特征） → 现在完整 592 维（含 Hjorth、熵、不对称）—— 作为 baseline

输出：
    - output/ablation/ablation_compare.csv ：4 行排行榜（含 原始维度/PCA维度/准确率/CV分数/最佳参数）
    - output/ablation/ablation_compare.md  ：Markdown 排行榜 + 每个实验的特征维度信息 + 结论
"""
import traceback
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

import config
import data_pipeline
import machine_learning
import glob


# ============================================================
# 特征子集筛选规则（A / B / C / D）
# ============================================================
def _is_time_domain_col(col: str) -> bool:
    return "_time_domain_" in col


def _is_freq_col(col: str) -> bool:
    return any([
        "_peak_frequency" in col,
        "_band_power_" in col,
        "_relative_band_power_" in col,
        "_band_ratio_" in col,
    ])


def select_feature_subset(dataset: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """
    根据实验名从 dataset 中挑出对应特征列，同时保留 label 列。
    experiment ∈ {"A", "B", "C", "D"}
    """
    if "label" not in dataset.columns:
        raise ValueError("dataset 必须包含 label 列")

    feature_cols = [c for c in dataset.columns if c != "label"]

    if experiment == "A":
        keep = [c for c in feature_cols if _is_time_domain_col(c)]
    elif experiment == "B":
        keep = [c for c in feature_cols if _is_freq_col(c)]
    elif experiment == "C":
        keep = [c for c in feature_cols if _is_time_domain_col(c) or _is_freq_col(c)]
    elif experiment == "D":
        keep = list(feature_cols)   # 全部
    else:
        raise ValueError(f"未知实验名 '{experiment}'，可选值: A/B/C/D")

    subset = dataset[keep + ["label"]].copy()
    return subset


EXPERIMENTS = [
    # (id, 中文名, 描述)
    ("A", "Time Domain", "时域特征（time_domain_mean/var/std/rms）"),
    ("B", "Frequency",   "频域特征（peak_freq + band_power(absolute/relative) + band_ratio）"),
    ("C", "Time + Freq", "时域 + 频域（排除 Hjorth / 熵 / C3-C4 不对称）"),
    ("D", "All Features", "全部特征（时域+频域+Hjorth+Entropy+Asymmetry）= Baseline"),
]


# ============================================================
# 结果保存
# ============================================================
def save_csv(results: list):
    rows = []
    for r in results:
        rows.append({
            "experiment": r["experiment_id"],
            "name": r["experiment_name"],
            "description": r["description"],
            "n_features_raw": r["n_features_raw"],
            "n_features_pca": r["n_features_pca"],
            "accuracy": "" if r["accuracy"] is None else f"{r['accuracy']:.4f}",
            "best_score": "" if r["best_score"] is None else f"{r['best_score']:.4f}",
            "best_params": "" if r["best_params"] is None else str(r["best_params"]),
            "error": "" if r.get("error") is None else r["error"],
        })
    df = pd.DataFrame(rows)
    save_path = config.ABLATION_DIR / "ablation_compare.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"Ablation CSV saved: {save_path}")
    return save_path


def save_markdown(results: list):
    save_path = config.ABLATION_DIR / "ablation_compare.md"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf8") as f:
        f.write("# Feature Ablation Study\n\n")
        f.write("## Setup\n\n")
        f.write("- Model: SVM (RBF kernel, GridSearchCV 5-fold)\n")
        f.write("- Pipeline: StandardScaler → PCA(99% variance) → SVM\n")
        f.write("- Train/Test: fixed split (random_state=42, 80/20, stratified) — **all 4 experiments share exactly the same rows**\n")
        f.write(f"- Dataset: 9 T-files (A01T~A09T) → total {results[0]['n_samples_total']} trials × 3 labels\n\n")

        f.write("## Ranking by Accuracy\n\n")
        f.write("|Rank|ID|Experiment|Raw Dim|PCA Dim|Accuracy|CV Score|Best Params|Status|\n")
        f.write("|---|---|---|------:|------:|-------:|-------:|---|---|\n")
        for rank, r in enumerate(results, start=1):
            if r.get("error"):
                status = f"❌ Error: `{r['error']}`"
                acc_str = cv_str = params_str = "N/A"
                raw_dim = pca_dim = "N/A"
            else:
                status = "✅ OK"
                acc_str = f"{r['accuracy']:.4f}"
                cv_str  = f"{r['best_score']:.4f}"
                params_str = f"`{str(r['best_params'])}`"
                raw_dim = f"{r['n_features_raw']}"
                pca_dim = f"{r['n_features_pca']}"
            f.write(
                f"|{rank}|"
                f"**{r['experiment_id']}**|"
                f"{r['experiment_name']}|"
                f"{raw_dim}|"
                f"{pca_dim}|"
                f"{acc_str}|"
                f"{cv_str}|"
                f"{params_str}|"
                f"{status}|\n"
            )

        # -------------------- 每个实验的详细说明 --------------------
        f.write("\n## Experiments Detail\n\n")
        for r in results:
            f.write(f"### [{r['experiment_id']}] {r['experiment_name']}\n\n")
            f.write(f"- **Description**: {r['description']}\n")
            if r.get("error"):
                f.write(f"- **Error**: `{r['error']}`\n\n")
                continue
            f.write(f"- **Raw features**: {r['n_features_raw']}\n")
            f.write(f"- **After PCA (99%)**: {r['n_features_pca']}\n")
            f.write(f"- **Accuracy**: **{r['accuracy']:.4f}**\n")
            f.write(f"- **CV Score**: {r['best_score']:.4f}\n")
            f.write(f"- **Best Params**: `{str(r['best_params'])}`\n\n")
            # 和 baseline (D) 的差值
            baseline = next((x for x in results if x["experiment_id"] == "D"), None)
            if baseline and baseline.get("error") is None and r["experiment_id"] != "D":
                delta = (r["accuracy"] - baseline["accuracy"]) * 100
                arrow = "🔺" if delta > 0 else ("🔻" if delta < 0 else "➖")
                f.write(f"- **Δ vs D (Baseline)**: {arrow} {delta:+.2f} pp (percentage points)\n\n")

        # -------------------- 结论 --------------------
        f.write("\n## Conclusion\n\n")
        ok_results = [r for r in results if r.get("error") is None]
        if ok_results:
            best = ok_results[0]
            baseline = next((x for x in ok_results if x["experiment_id"] == "D"), None)
            f.write(
                f"The best feature subset is **[{best['experiment_id']}] {best['experiment_name']}**, "
                f"with **{best['accuracy']:.4f}** accuracy "
                f"(raw dim = {best['n_features_raw']}, PCA dim = {best['n_features_pca']}).\n\n"
            )
            # 时域 vs 频域 对比
            exp_a = next((x for x in ok_results if x["experiment_id"] == "A"), None)
            exp_b = next((x for x in ok_results if x["experiment_id"] == "B"), None)
            if exp_a and exp_b:
                diff = (exp_b["accuracy"] - exp_a["accuracy"]) * 100
                if diff > 0:
                    f.write(
                        f"- **Frequency (B) > Time Domain (A)** by **{diff:+.2f} pp** → "
                        f"频域特征（频段能量/比值）比时域统计量更有判别力。\n"
                    )
                elif diff < 0:
                    f.write(
                        f"- **Time Domain (A) > Frequency (B)** by **{diff:+.2f} pp** → "
                        f"时域统计量比频域特征更有判别力。\n"
                    )
                else:
                    f.write("- **Time Domain (A) ≈ Frequency (B)** → 两类特征贡献接近。\n")
            # C = Time+Freq  vs D = All
            exp_c = next((x for x in ok_results if x["experiment_id"] == "C"), None)
            if baseline and exp_c:
                diff_d = (baseline["accuracy"] - exp_c["accuracy"]) * 100
                if diff_d > 0.5:
                    f.write(
                        f"- **Hjorth + Entropy + Asymmetry (the extra features in D) contribute +{diff_d:+.2f} pp** → "
                        f"保留全部特征是值得的。\n"
                    )
                elif diff_d < -0.5:
                    f.write(
                        f"- **Hjorth + Entropy + Asymmetry actually hurt accuracy by {diff_d:+.2f} pp** → "
                        f"这些特征可能引入噪声，可以考虑移除。\n"
                    )
                else:
                    f.write(
                        f"- **Hjorth + Entropy + Asymmetry contribute {diff_d:+.2f} pp (within ±0.5pp)** → "
                        f"对当前模型和数据集影响不大，保留或移除都可。\n"
                    )
        else:
            f.write("⚠️ All experiments failed. Check the Status column for details.\n")

        failed = [r for r in results if r.get("error")]
        if failed:
            f.write("\n## Errors\n\n")
            for r in failed:
                f.write(f"### [{r['experiment_id']}] {r['experiment_name']}\n\n")
                f.write(f"```\n{r['error']}\n```\n\n")
    logger.info(f"Ablation Markdown saved: {save_path}")
    return save_path


# ============================================================
# 主流程
# ============================================================
def load_full_dataset():
    """和 train.py 一致：扫 9 个 T.gdf → concat 成一个超大 DataFrame"""
    all_gdf = sorted(glob.glob(str(config.DATA_DIR / "*.gdf")))
    files = [f for f in all_gdf if f.endswith("T.gdf")]
    if not files:
        files = sorted(glob.glob(str(config.DATA_DIR / "*.edf")))
    features_list = []
    for file in files:
        result = data_pipeline.extract_feature(file)
        if result is None:
            continue
        features_list.append(result)
    dataset = pd.concat(features_list, axis=0, ignore_index=True)
    return dataset


def run_ablation(model_type: str = "svm"):
    # -------------------- 1. 加载全量数据 --------------------
    dataset = load_full_dataset()
    n_samples_total = len(dataset)
    logger.info("=" * 70)
    logger.info(f"Feature Ablation Study | total samples = {n_samples_total} | model = {model_type}")
    logger.info("=" * 70)

    # -------------------- 2. 统一计算 train/test 行索引（所有 4 个实验共享） --------------------
    all_idx = np.arange(n_samples_total)
    y_all = dataset["label"].values.astype(int)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=y_all
    )
    sample_split = (train_idx, test_idx)
    logger.info(
        f"Fixed split: train={len(train_idx)} trials, test={len(test_idx)} trials "
        f"(shared across all 4 experiments)"
    )

    # -------------------- 3. 逐个实验跑 --------------------
    results = []
    total_exps = len(EXPERIMENTS)
    for idx, (exp_id, exp_name, desc) in enumerate(EXPERIMENTS, start=1):
        logger.info("-" * 70)
        logger.info(f"[{idx}/{total_exps}] Experiment {exp_id} → {exp_name}")
        logger.info(f"     Description: {desc}")
        logger.info("-" * 70)

        # 3.1 按实验筛选特征子集
        subset = select_feature_subset(dataset, exp_id)
        n_features_raw = subset.shape[1] - 1   # 去掉 label
        logger.info(f"     Raw feature dim (before pipeline) = {n_features_raw}")

        # 3.2 训练（使用外部传入的 sample_split，保证和其他实验 100% 同 train/test 行）
        model_name = f"svm_ablation_{exp_id}"
        try:
            train_result = machine_learning.train_pipeline(
                dataset=subset,
                model_type=model_type,
                model_name=model_name,
                sample_split=sample_split,
            )
            # 拿到 PCA 维度（从 best_pipeline.named_steps 读）
            best_pipeline = train_result["model"]
            if "pca" in best_pipeline.named_steps:
                pca_dim = best_pipeline.named_steps["pca"].n_components_
            else:
                pca_dim = n_features_raw

            results.append({
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "description": desc,
                "n_samples_total": n_samples_total,
                "n_features_raw": n_features_raw,
                "n_features_pca": pca_dim,
                "accuracy": train_result["accuracy"],
                "best_score": train_result["best_score"],
                "best_params": train_result["best_params"],
                "error": None,
            })
            logger.info(
                f"[{idx}/{total_exps}] ✅ {exp_id} done. "
                f"raw={n_features_raw} → pca={pca_dim} → acc={train_result['accuracy']:.4f}"
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            logger.error(f"[{idx}/{total_exps}] ❌ {exp_id} FAILED: {err_msg}")
            logger.error(f"Traceback:\n{tb}")
            results.append({
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "description": desc,
                "n_samples_total": n_samples_total,
                "n_features_raw": n_features_raw,
                "n_features_pca": None,
                "accuracy": None,
                "best_score": None,
                "best_params": None,
                "error": err_msg,
            })

    # -------------------- 4. 按准确率降序排（失败放最后） --------------------
    def rank_key(r):
        acc = r.get("accuracy")
        return (acc is not None, acc if acc is not None else -1.0)
    results.sort(key=rank_key, reverse=True)

    # -------------------- 5. 保存 + 控制台 summary --------------------
    save_csv(results)
    save_markdown(results)

    logger.info("=" * 70)
    logger.info("Ablation Summary (sorted by accuracy desc):")
    for rank, r in enumerate(results, start=1):
        if r.get("error"):
            logger.info(f"  #{rank}  [{r['experiment_id']}] {r['experiment_name']:<13}  ❌ ERROR: {r['error']}")
        else:
            logger.info(
                f"  #{rank}  [{r['experiment_id']}] {r['experiment_name']:<13}  "
                f"raw={r['n_features_raw']:>4} → pca={r['n_features_pca']:>3}  "
                f"acc={r['accuracy']:.4f}  cv={r['best_score']:.4f}  "
                f"params={r['best_params']}"
            )
    logger.info("=" * 70)
    logger.info(f"Results saved to {config.ABLATION_DIR}")
    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    run_ablation(model_type="svm")

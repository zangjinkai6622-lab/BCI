"""AIC 比赛项目入口。

用法（在 AIC 目录下）：
  python main.py prepare       # 仅预处理并缓存 epochs（GDF -> (N,22,1000)）
  python main.py compare       # 实验①：LR/RF/SVM vs EEGNet vs STFA-Net vs STFA-DGNet
  python main.py ablation      # 实验②：STFA-Net 逐模块消融
  python main.py loso          # 实验③：跨被试留一验证（Subject-Independent）
  python main.py all           # 依次跑 compare -> ablation -> loso
  python main.py compare --quick    # 冒烟测试：3 个被试、3 个 epoch，验证全流程

  --seeds 3                    # 所有实验通用：3 个随机种子重复，结果带 mean±std
  --models eegnet,stfa_light   # 指定 DL 模型；light 档（~5千参数）用于 LOSO 跨被试

结果落盘：results/ 下的 csv / md / 混淆矩阵 / 通道注意力图。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import config
from data.dataset import load_all_subjects
import experiments


def main():
    parser = argparse.ArgumentParser(description="STFA-Net / STFA-DGNet 实验入口")
    parser.add_argument("stage", choices=["prepare", "compare", "ablation", "loso", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="冒烟测试：只用前 3 个被试、训练 3 个 epoch")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"覆盖训练 epoch 数（默认 {config.EPOCHS}）")
    parser.add_argument("--seeds", type=int, default=1,
                        help="每个模型/折的随机种子重复次数（1 或 3）；所有实验通用")
    parser.add_argument("--models", type=str, default=None,
                        help="深度学习模型列表（逗号分隔）。"
                             "compare 默认 eegnet,stfa,stfa_dg；"
                             "loso 默认 eegnet,stfa,stfa_dg；"
                             "可选追加 stfa_light,stfa_dg_light")
    args = parser.parse_args()

    seeds = config.SEEDS[:args.seeds] if args.seeds > 1 else [config.SEED]

    model_names = None
    if args.models:
        model_names = tuple(m.strip() for m in args.models.split(",") if m.strip())
        bad = [m for m in model_names if m not in experiments.DL_MODEL_NAMES]
        if bad:
            parser.error(f"未知模型 {bad}，可选: {experiments.DL_MODEL_NAMES}")

    epochs = 3 if args.quick else args.epochs
    n_subjects = 3 if args.quick else None

    print(f"=== AIC | stage={args.stage} | epochs={epochs} | "
          f"subjects={'3(quick)' if n_subjects else 'all 9'} | data={config.DATA_DIR} ===")

    if args.stage == "prepare":
        data = load_all_subjects(use_cache=False)
        print(f"X={data['X'].shape}, y={data['y'].shape}, "
              f"subject={data['subject'].shape}, channels={len(data['channels'])}")
    elif args.stage == "compare":
        cmp_kw = {"epochs": epochs, "n_subjects": n_subjects, "seeds": seeds}
        if model_names:
            cmp_kw["model_names"] = model_names
        experiments.experiment_comparison(**cmp_kw)
    elif args.stage == "ablation":
        experiments.experiment_ablation(epochs=epochs, n_subjects=n_subjects, seeds=seeds)
    elif args.stage == "loso":
        loso_kw = {"epochs": epochs, "seeds": seeds}
        if model_names:
            loso_kw["model_names"] = model_names
        if args.quick:
            # quick 模式下 LOSO 只在前 3 个被试间做留一
            data = experiments.subset_subjects(load_all_subjects(), 3)
            loso_kw["data"] = data
        experiments.experiment_loso(**loso_kw)
    elif args.stage == "all":
        cmp_kw = {"epochs": epochs, "n_subjects": n_subjects, "seeds": seeds}
        if model_names:
            cmp_kw["model_names"] = model_names
        experiments.experiment_comparison(**cmp_kw)
        experiments.experiment_ablation(epochs=epochs, n_subjects=n_subjects, seeds=seeds)
        loso_kw = {"epochs": epochs, "seeds": seeds}
        if model_names:
            loso_kw["model_names"] = model_names
        if args.quick:
            data = experiments.subset_subjects(load_all_subjects(), 3)
            loso_kw["data"] = data
        experiments.experiment_loso(**loso_kw)


if __name__ == "__main__":
    main()

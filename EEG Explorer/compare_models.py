import machine_learning
import pandas as pd
import config    
import glob
import data_pipeline
import label
def compare_models(dataset:pd.DataFrame):
    models = [
        ("svm_v1", "svm"),
        ("rf_v1", "rf"),
        ("lr_v1", "lr")
    ]
    results = []
    for model_name, model_type in models:
        result = machine_learning.train_pipeline(dataset,model_type,model_name)
        results.append(result)
    results = sort_results(results)
    save_csv(results)
    save_markdown(results)
    print("=" * 50)
    print("Model comparison completed.")
    print(f"Results saved to {config.COMPARE_DIR}")
    print("=" * 50)
    return results

def sort_results(results:list):
    results.sort(
        key=lambda x:x["accuracy"],
        reverse=True
    )
    return results


def save_csv(results:list):
    df = pd.DataFrame(
        [
            {
                "model_name": r["model_name"],
                "accuracy": r["accuracy"],
                "best_score": r["best_score"],
                "best_params": str(r["best_params"])
            }
            for r in results
        ]
    )
    save_path = config.COMPARE_DIR / "model_compare.csv"
    df.to_csv(save_path,index=False,encoding="utf-8-sig")


def save_markdown(results:list):
    save_path = config.COMPARE_DIR / "model_compare.md"
    with open(save_path,"w",encoding="utf8") as f:
        f.write("# Model Comparison\n\n")
        f.write("|Rank|Model|Accuracy|CV Score|Best Params|\n")
        f.write("|---|---|---|---|---|\n")
        for i, result in enumerate(results):
            f.write(
                f"|{i+1}|"
                f"{result['model_name']}|"
                f"{result['accuracy']:.4f}|"
                f"{result['best_score']:.4f}|"
                f"{str(result['best_params'])}|\n"
            )
        best = results[0]
        f.write("\n## Conclusion\n\n")
        f.write(
            f"The best model is **{best['model_name']}**, "
            f"with an accuracy of **{best['accuracy']:.4f}** "
            f"and a cross-validation score of "
            f"**{best['best_score']:.4f}**.\n"
        )

if __name__ == "__main__":
    files = glob.glob("EEG Explorer/data/*.edf")[:3]
    features_list = []
    for file in files:
        result = data_pipeline.process_one_file(file)
        if result is None:
            continue
        feature_df = result[0]
        feature_df["label"] = label.get_label(file)
        features_list.append(feature_df)
    dataset = pd.concat(features_list, ignore_index=True)
    compare_models(dataset)
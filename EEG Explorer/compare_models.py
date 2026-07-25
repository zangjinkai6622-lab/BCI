import machine_learning
import pandas as pd
import config
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
    save_csv(results,'model_compare.csv')
    save_markdown(results,"model_compare.md")
    return results

def sort_results(results:list):
    results.sort(
        key=lambda x:x["accuracy"],
        reverse=True
    )
    return results


def save_csv(results:list):
    df=pd.DataFrame(results)
    save_path = config.COMPARE_DIR / "model_compare.csv"
    df.to_csv(save_path,index=False,encoding="utf-8-sig")


def save_markdown(results:list):
    save_path = config.COMPARE_DIR / "model_compare.md"
    with open(save_path,"w",encoding="utf8") as f:
        f.write("# Model Comparison\n\n")
        f.write("|Rank|Model|Accuracy|CV Score|Best Params|\n")
        f.write("|---|---|---|---|---|\n")
        for i,result in enumerate(results):
            f.write(
                f"|{i+1}|"
                f"{result['model_name']}|"
                f"{result['accuracy']:.4f}|"
                f"{result['best_score']:.4f}|"
                f"{result['best_params']}|\n"
            )
        best = results[0]
        f.write("\n## Conclusion\n\n")
        f.write(
            f"The best model is **{best['model_name']}**, "
            f"with an accuracy of **{best['accuracy']:.4f}** "
            f"and a cross-validation score of "
            f"**{best['best_score']:.4f}**.\n"
        )
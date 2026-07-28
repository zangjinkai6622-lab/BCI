import config
import pandas as pd

def generate_report(
    analysis_result: dict,
    visualization_result: dict,
    compare_result: list = None,
    prediction_result: dict = None,
    filename: str = "report.md"
):
    with open(config.REPORT_DIR / filename, "w", encoding="utf-8") as file:
        file.write("# EEG Explorer Report\n\n")

        file.write("## 1. Dataset Information\n")
        file.write(f"Rows: {analysis_result['basic']['dataset']['rows']}\n")
        file.write(f"Columns: {analysis_result['basic']['dataset']['columns']}\n\n")

        file.write("## 2. Preprocessing\n")

        file.write("### 2.1 Band-pass Filter\n")
        file.write(f"![](../figures/{visualization_result['preprocess_figures']['bandpass']})\n\n")

        file.write("### 2.2 Notch Filter\n")
        file.write(f"![](../figures/{visualization_result['preprocess_figures']['notch']})\n\n")

        file.write("## 3. Statistics\n\n")
        file.write(analysis_result["basic"]["statistics"].to_markdown())
        file.write("\n\n")

        file.write("## 4. Missing Values\n\n")
        file.write(analysis_result["basic"]["missing_values"].to_markdown())
        file.write("\n\n")

        file.write("## 5. Data Types\n\n")
        file.write(analysis_result["basic"]["data_type"].to_markdown())
        file.write("\n\n")

        file.write("## 6. Time Domain Features\n\n")

        time_df = pd.DataFrame(
            analysis_result["features"]["time_features"]
        ).T

        file.write(time_df.to_markdown())
        file.write("\n\n")

        file.write("## 7. Hjorth Parameters\n\n")
        hjorth_df = pd.DataFrame(
            analysis_result["features"]["hjorth"]
        ).T

        file.write(hjorth_df.to_markdown())
        file.write("\n\n")

        file.write("## 8. Entropy Features\n\n")

        entropy_df = pd.DataFrame(
            analysis_result["features"]["entropy"]
        ).T

        file.write(entropy_df.to_markdown())
        file.write("\n\n")

        file.write("## 9. Time Figures\n\n")

        for name in visualization_result["time_figures"]:
            file.write(f"![](../figures/{name})\n")

        file.write("\n")

        file.write("## 10. Frequency Domain Features\n\n")
        fft_df = pd.DataFrame(
            analysis_result["signals"]["fft"]
        )
        file.write(fft_df.to_markdown())
        file.write("\n\n")
        psd_df = pd.DataFrame(
            analysis_result["signals"]["psd"]
        )
        file.write(psd_df.to_markdown())
        file.write("\n\n")

        file.write("## 11. Frequency Figures\n\n")
        for name in visualization_result["frequency_figures"]:
            file.write(f"![](../igures/{name})\n")

        file.write("\n")
        
        file.write("## 12. Band Power\n\n")
        band_df = pd.DataFrame(
            analysis_result["features"]["band_power"]
        )
        file.write(band_df.to_markdown())
        file.write("\n\n")

        file.write("## 13. Interpretation\n\n")
        interpretation = analysis_result["interpretation"]
        for channel, texts in interpretation.items():
            file.write(f"### {channel}\n")

            for text in texts:
                file.write(f"- {text}\n")
            file.write("\n")

        if compare_result is not None:
            file.write("## 14. Model Comparison\n\n")
            file.write("|Rank|Model|Accuracy|CV Score|\n")
            file.write("|---|---|---|---|\n")
            for i, result in enumerate(compare_result):
                file.write(
                    f"|{i+1}|"
                    f"{result['model_name']}|"
                    f"{result['accuracy']:.4f}|"
                    f"{result['best_score']:.4f}|\n"
                )
            file.write("\n")
            best = compare_result[0]
            file.write("### Best Model\n\n")
            file.write(f"- Model: {best['model_name']}\n")
            file.write(f"- Accuracy: {best['accuracy']:.4f}\n")
            file.write(f"- CV Score: {best['best_score']:.4f}\n")
            file.write(f"- Best Params: {best['best_params']}\n\n")

        if prediction_result is not None:

            file.write("## 15. Prediction\n\n")

            file.write(f"- File: {prediction_result['file']}\n")
            file.write(f"- Model: {prediction_result['model']}\n")
            file.write(f"- Result: {prediction_result['prediction']}\n\n")

        file.write("## 16. Conclusion\n\n")
        file.write(
            "The EEG Explorer pipeline successfully completed EEG preprocessing, "
            "feature extraction and visualization.\n\n"
        )
        if compare_result is not None:

            best = compare_result[0]

            file.write(
                f"The best-performing model was **{best['model_name']}**, "
                f"achieving an accuracy of **{best['accuracy']:.4f}** "
                f"with a cross-validation score of **{best['best_score']:.4f}**.\n\n"
            )
        if prediction_result is not None:

            file.write(
                f"The prediction module classified the EEG file as "
                f"**{prediction_result['prediction']}**.\n"
            )
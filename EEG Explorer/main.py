import train
import inference
import data_pipeline
import visualization
import report
import compare_models
import prediction

def main():
    result = data_pipeline.process_file(
        "EEG Explorer/data/S001R05.edf"
    )
    prediction_result = inference.predict_feature(
        result["feature_df"]
    )
    report.generate_report(
        analysis_result=result["analysis_result"],
        prediction_result=prediction_result
    )

if __name__ == '__main__':
    main()
import inference
import data_pipeline
import report


def main():
    result = data_pipeline.process_file(
        "EEG Explorer/data/S001R05.edf"
    )
    prediction_result = inference.predict_file(
        "EEG Explorer/data/S001R05.edf"
    )
    report.generate_report(
        analysis_result=result["analysis_result"],
        visualization_result=result["visualization_result"],
        prediction_result=prediction_result
    )

if __name__ == '__main__':
    main()
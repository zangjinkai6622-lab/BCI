import main
import channel
import feature


def load_data(path: str):
    return main.reader.read_edf(path)

def process_one_file(file_path):
    raw_df = load_data(file_path)
    if raw_df is None:
        return
    raw_df = channel.normalize_channel_names(raw_df)
    channels=channel.get_available_channels(raw_df)
    clean_df,preprocess_result=main.preprocess(raw_df,channels)
    analysis_result = main.get_features(clean_df,channels)
    windows=feature.split_windows(clean_df)
    feature_df=feature.create_feature_dataframe(windows,channels)
    visualization_result = main.get_visualization(preprocess_result, analysis_result,channels)
    return (
        feature_df,
        preprocess_result,
        analysis_result,
        visualization_result
    )
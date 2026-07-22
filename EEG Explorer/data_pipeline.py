import reader
import channel
import feature
import main
import preprocessing
import analyser
import visualization
def process_one_file(file_path):
    raw_df = reader.read_edf(file_path)
    if raw_df is None:
        return
    raw_df = channel.normalize_channel_names(raw_df)
    channels=channel.get_available_channels(raw_df)
    clean_df,preprocess_result=preprocessing.preprocess(raw_df,channels)
    analysis_result = analyser.get_features(clean_df,channels)
    windows=feature.split_windows(clean_df)
    feature_df=feature.create_feature_dataframe(windows,channels)
    visualization_result = visualization.get_visualization(preprocess_result, analysis_result,channels)
    return (
        feature_df,
        preprocess_result,
        analysis_result,
        visualization_result
    )

def extract_feature(file_path:str):
    raw_df = reader.read_edf(file_path)
    if raw_df is None:
        return
    raw_df = channel.normalize_channel_names(raw_df)
    channels=channel.get_available_channels(raw_df)
    clean_df,_=main.preprocess(raw_df,channels)
    windows=feature.split_windows(clean_df)
    feature_df=feature.create_feature_dataframe(windows,channels)
    return feature_df
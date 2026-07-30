import reader
import channel
import feature
import preprocessing
import analyser
import visualization

"""
训练流程使用。
返回训练所需的特征及分析结果。
"""
def process_one_file(file_path:str):
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
    return {
        'feature_df':feature_df,
        'preprocess_result':preprocess_result,
        'analysis_result':analysis_result,
        'visualization_result':visualization_result
    }

def extract_feature(file_path:str):
    raw_df = reader.read_edf(file_path)
    if raw_df is None:
        return
    raw_df = channel.normalize_channel_names(raw_df)
    channels=channel.get_available_channels(raw_df)
    clean_df,_=preprocessing.preprocess(raw_df,channels)
    windows=feature.split_windows(clean_df)
    feature_df=feature.create_feature_dataframe(windows,channels)
    return feature_df

"""
完整数据处理流程。
返回原始数据、预处理结果、分析结果、
可视化结果和机器学习特征。
"""
def process_file(file:str):
    raw_df = reader.read_edf(file)
    if raw_df is None:
        return None
    raw_df = channel.normalize_channel_names(raw_df)
    channels = channel.get_available_channels(raw_df)
    filtered_df, preprocess_result = preprocessing.preprocess(raw_df, channels)
    analysis_result = analyser.get_features(filtered_df, channels)
    windows = feature.split_windows(filtered_df)
    feature_df = feature.create_feature_dataframe(windows, channels)
    visualization_result = visualization.get_visualization(preprocess_result, analysis_result, channels)
    return {
        "raw_df": raw_df,
        "filtered_df": filtered_df,
        "preprocess_result": preprocess_result,
        "analysis_result": analysis_result,
        "visualization_result": visualization_result,
        "feature_df": feature_df,
        "channels": channels,
        "file_path": file
    }
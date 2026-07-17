import  pandas as pd

def creat_feature_dataframe(analysis_result:dict):
    feature={}
    # 时域
    for channel,values in analysis_result['features']['time_features'].items():
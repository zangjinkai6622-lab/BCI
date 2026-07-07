import pandas as pd
import numpy as np

def get_dataset_info(df: pd.DataFrame):
    return {
        'rows':df.shape[0],
        'columns':df.shape[1],
        "column_names": list(df.columns), 
        "memory_usage": df.memory_usage(deep=True).sum()/1024**2,
    }
def get_missing_values(df: pd.DataFrame):
    return df.isnull().sum()
def get_data_type(df: pd.DataFrame):
    return df.dtypes
    
def get_statistics(df:pd.DataFrame):
    return df.describe()
def get_time_domain_features(df:pd.DataFrame):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    features={}
    for column in numerical_columns:
        channel_feature={
            "mean": df[column].mean(),
            "variance": df[column].var(),
            "std": df[column].std(),
            "rms": np.sqrt(np.mean(df[column]**2)),
            "peak_to_peak": np.ptp(df[column]),
            "zero_crossing_rate": ...
        }
        features[column]=channel_feature

    return features
        

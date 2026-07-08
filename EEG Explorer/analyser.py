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

def get_mean(df:pd.DataFrame,column:str):
    return df[column].mean()

def get_variance(df:pd.DataFrame,column:str):
    return df[column].var()

def get_standard_deviation(df:pd.DataFrame,column:str):
    return df[column].std()

def get_rms(df:pd.DataFrame,column:str):
    return np.sqrt(np.mean(df[column]**2))

def get_ptp(df:pd.DataFrame,column:str):
    return np.ptp(df[column])

def get_zero_crossing_rate(df:pd.DataFrame,column:str):
    signal=df[column].to_numpy()
    sign=np.sign(signal)
    count=0
    previous_sign = None
    for i in range(len(sign)):
        if sign[i] != 0:
            if previous_sign is not None:
                if sign[i] != previous_sign:
                    count += 1
                    
            previous_sign = sign[i]

        if sign[i] == 0:
            continue
    return count
                   

def get_time_domain_features(df:pd.DataFrame):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    features={}
    for column in numerical_columns:
        channel_feature={
            "mean": get_mean(df,column),
            "variance": get_variance(df,column),
            "std": get_standard_deviation(df,column),
            "rms": get_rms(df,column),
            "peak_to_peak": get_ptp(df,column),
            "zero_crossing_rate": get_zero_crossing_rate(df,column)           
        }
        features[column]=channel_feature

    return features
        

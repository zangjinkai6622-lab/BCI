import pandas as pd

def get_shape(df: pd.DataFrame):
    return df.shape
def get_columns(df: pd.DataFrame):
    return list(df.columns)
def get_missing_values(df: pd.DataFrame):
    return df.isnull().sum()
def get_data_type(df: pd.DataFrame):
    return df.dtypes
    
def get_statistics(df:pd.DataFrame):
    return df.describe()
# def get_eeg_feature(df:pd.DataFrame):
#     return df.iloc[:,1:]
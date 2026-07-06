import pandas as pd


def get_dataset_info(df: pd.DataFrame):
    return {
        'rows':df.shape[0],
        'columns':df.shape[1],
        "column_names": list(df.columns), 
        "memory_usage": df.memory_usage(deep=True).sum()/1024**2,
    }
# def get_shape(df: pd.DataFrame):
#     return {
#         'rows':df.shape[0],
#         'columns':df.shape[1]
#     }
# def get_columns(df: pd.DataFrame):
#     return list(df.columns)
def get_missing_values(df: pd.DataFrame):
    return df.isnull().sum()
def get_data_type(df: pd.DataFrame):
    return df.dtypes
    
def get_statistics(df:pd.DataFrame):
    return df.describe()
# def get_eeg_feature(df:pd.DataFrame):
#     return df.iloc[:,1:]
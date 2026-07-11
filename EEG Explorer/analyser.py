import pandas as pd
import numpy as np
import scipy.signal
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
        
def get_fft(df:pd.DataFrame,column:str,sampling_rate:int): # sampling_rate是采样率，一共记录多少个点，即多少Hz
    signal=df[column].to_numpy() # 获得这一列数据，转为np计算
    fft_result=np.fft.fft(signal) # 直接调用FTT
    # FFT返回的是复数数组，模长代表振幅，平方为能量，相位。
    amplitude=np.abs(fft_result) # 获得振幅，abs取模长
    frequency=np.fft.fftfreq(len(signal),1/sampling_rate) # 获得频率，N，两点之间的时间间隔
    positive_amplitude=amplitude[0:len(amplitude)//2+1]
    positive_frequency=frequency[0:len(frequency)//2+1]
    return {
        "amplitude":positive_amplitude,
        "frequency":positive_frequency
    }


# psd有归一化,单位频率上的功率，标准统一，便于比较
def get_psd(df:pd.DataFrame,column:str,sampling_rate:int): # 功率谱密度
    signal=df[column].to_numpy()
    frequency,psd=scipy.signal.welch(signal,fs=sampling_rate)
    return {
        "frequency":frequency,
        "psd":psd
    }

#  频带功率，对psd进行积分,在psd算法基础上进行
def get_band_power(psd_result:dict,bands:dict):
    result={}
    frequency=psd_result['frequency']
    psd=psd_result['psd']
    for band,(low,high) in bands.items():
        mask=(frequency>=low)&(frequency<=high) # 返回一个符合范围条件的TRUE FALSE的数组
        power=np.trapezoid(psd[mask],frequency[mask])
        result[band]=power
        
    return result





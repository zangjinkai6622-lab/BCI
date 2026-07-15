import scipy
import numpy as np
import pandas as pd
import config
def apply_bandpass_filter(df: pd.DataFrame, column:str,lowcut:float,highcut:float,sampling_rate:int):
    signal=df[column].to_numpy()
    # 滤波,order参数越大，效果越好，但同时有计算开销,b,a返回顺序不能反,fs=sampling_rate直接使用避免了归一化的过程
    b,a=scipy.signal.butter(config.FILTER_ORDER,[lowcut,highcut],fs=sampling_rate,btype='bandpass')
    # 用filtfilt()方法因为这个方法会正反各一次，抵消掉相位，但是lfilter()方法没有，会有相位的影响
    filtered_signal=scipy.signal.filtfilt(b,a,signal)
    # 防止直接修改df，创建一个副本
    filtered_df = df.copy()
    filtered_df[column] = filtered_signal
    return filtered_df

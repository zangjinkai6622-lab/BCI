import scipy
import numpy as np
import pandas as pd
import config
def apply_bandpass_filter(df: pd.DataFrame, column:str,lowcut:float,highcut:float,sampling_rate:int):
    signal=df[column].to_numpy()
    # 带通滤波是取中间，有范围
    # 滤波,order参数越大，效果越好，但同时有计算开销,b,a返回顺序不能反,fs=sampling_rate直接使用避免了归一化的过程
    b,a=scipy.signal.butter(config.FILTER_ORDER,[lowcut,highcut],fs=sampling_rate,btype='bandpass')
    # 用filtfilt()方法因为这个方法会正反各一次，抵消掉相位，但是lfilter()方法没有，会有相位的影响
    filtered_signal=scipy.signal.filtfilt(b,a,signal)
    # 防止直接修改df，创建一个副本
    filtered_df = df.copy()
    filtered_df[column] = filtered_signal
    return filtered_df


def apply_notch_filter(df:pd.DataFrame,column:str,notch_freq:int,sampling_rate:int):
    signal=df[column].to_numpy()
    b,a=scipy.signal.iirnotch(w0=notch_freq,Q=config.NOTCH_Q,fs=sampling_rate) 
    # 陷波滤波，是取出中间一块，中心为w0，宽度也就是带宽由Δf确定。
    # Q = f₀ / Δf,f₀ (或 w0) = 陷波中心频率（Hz）;Δf = 带宽（Bandwidth），单位 Hz;Q = 品质因子（无量纲）
    # w0是要去除的频率，Q大，陷波窄，保留更多信号，但可能不完全去除噪声，Q小：陷波宽，去噪效果好，但可能去除有用信号
    filtered_signal=scipy.signal.filtfilt(b,a,signal)
    filtered_df=df.copy()
    filtered_df[column]=filtered_signal
    return filtered_df

import pandas as pd
import numpy as np
import scipy.signal
import config
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
    numerical_columns = [
        col for col in numerical_columns
        if col != "time"
    ]
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

def get_hjorth(df:pd.DataFrame,column:str):
    signal=df[column].to_numpy()
    first_derivative=np.diff(signal)
    second_derivative=np.diff(first_derivative)
    activity=np.var(signal)
    mobility=np.sqrt(np.var(first_derivative)/activity)
    complexity=np.sqrt(np.var(second_derivative)/np.var(first_derivative))
    return {
        "Activity":activity,
        "Mobility":mobility,
        "Complexity":complexity
    }

def get_entropy(df:pd.DataFrame,column:str):
    siganl=df[column].to_numpy()
    hist,bins=np.histogram(siganl,bins=10,density=True)
    #  数据是连续的，如果统计单个数据都是1/N的概率，没有意义，所以分段统计，返回的hist是一个bins里的数据数量或者概率，bins返回是段边界，
    # density=True：概率归一化，即面积和为1 ，切此时hist对应概率；density=False：返回的是数量，此时hist对应数据数量
    prob=hist[hist>0]  # 删除概率为0的概率
    entropy=-np.sum(prob*np.log2(prob)) # 0-1范围的log是负数，所有要加-
    return {
        'entropy':entropy
    }


def get_interpretation(band_power_result:dict,hjorth_result:dict,entropy_result:dict,column:str):
    interpretation=[]

    max_band=max(band_power_result,key=band_power_result.get)
    interpretation.append(f"{max_band.capitalize()} band has the highest power.")

    entropy = entropy_result['entropy']

    if entropy < 1:
        interpretation.append("Signal complexity is low.")
    elif entropy < 2:
        interpretation.append("Signal complexity is moderate.")
    else:
        interpretation.append("Signal complexity is high.")

    if hjorth_result['Mobility'] > 0.5:
        interpretation.append("Signal changes rapidly.")
    else:
        interpretation.append("Signal changes slowly.")

    if hjorth_result['Complexity'] > 1.0:
        interpretation.append("Signal waveform is more complex.")
    else:
        interpretation.append("Signal waveform is relatively simple.")


    return interpretation



# =====【修改3-新增MI特征1】相对频带功率：每个频带功率/总功率，减少个体差异影响
def get_relative_band_power(band_power: dict) -> dict:
    total = sum(band_power.values())
    if total <= 0:
        return {f"relative_{k}": 0.0 for k in band_power.keys()}
    return {f"relative_{k}": v / total for k, v in band_power.items()}

# =====【修改3-新增MI特征2】频带比值：MI常用指标，如alpha/beta, theta/beta, beta/(alpha+theta)等
def get_band_ratios(band_power: dict) -> dict:
    delta = band_power.get("delta", 1e-10)
    theta = band_power.get("theta", 1e-10)
    alpha = band_power.get("alpha", 1e-10)
    beta  = band_power.get("beta",  1e-10)
    gamma = band_power.get("gamma", 1e-10)
    # 防止除零
    eps = 1e-10
    return {
        "ratio_alpha_beta":   alpha / (beta + eps),    # 注意力/放松指标
        "ratio_theta_beta":   theta / (beta + eps),    # 困意/认知负荷
        "ratio_beta_alpha":   beta  / (alpha + eps),
        "ratio_beta_theta":   beta  / (theta + eps),
        "ratio_alpha_theta":  alpha / (theta + eps),
        "ratio_gamma_beta":   gamma / (beta + eps),
        "ratio_mu_beta":      alpha / (beta + eps)    # mu = 8-13Hz与alpha重合，MI关键
    }

# =====【修改3-新增MI特征3】ERD/ERS近似特征：将窗口分成前后两半，计算后段相对前段的功率变化
# ERD = Event-Related Desynchronization (事件相关去同步，MI时mu/beta功率↓)
# ERS = Event-Related Synchronization  (事件相关同步，MI后期或对侧功率↑)
# ERD/ERS公式：(后段功率 - 前段功率) / 前段功率，负值为ERD，正值为ERS
def get_erd_ers(window_df: pd.DataFrame, channel: str, sampling_rate: int, bands: dict) -> dict:
    signal = window_df[channel].to_numpy()
    n = len(signal)
    half = n // 2
    if half < 10:
        return {f"erd_ers_{band}": 0.0 for band in bands.keys()}
    # 切分前后半段
    sig_first = signal[:half]
    sig_last  = signal[half:]
    # 分别计算两段的频带功率
    freqs_f, psd_f = scipy.signal.welch(sig_first, fs=sampling_rate, nperseg=min(256, len(sig_first)))
    freqs_l, psd_l = scipy.signal.welch(sig_last,  fs=sampling_rate, nperseg=min(256, len(sig_last)))
    result = {}
    eps = 1e-10
    for band, (low, high) in bands.items():
        mask_f = (freqs_f >= low) & (freqs_f <= high)
        mask_l = (freqs_l >= low) & (freqs_l <= high)
        power_f = np.trapezoid(psd_f[mask_f], freqs_f[mask_f])
        power_l = np.trapezoid(psd_l[mask_l], freqs_l[mask_l])
        # ERD/ERS
        result[f"erd_ers_{band}"] = (power_l - power_f) / (power_f + eps)
    return result

# =====【修改3-新增MI特征4】频谱质心(centroid)和边缘频率，描述频谱形状
def get_spectral_shape(psd_result: dict) -> dict:
    freq = psd_result["frequency"]
    psd  = psd_result["psd"]
    total_power = np.sum(psd) + 1e-10
    # 频谱质心
    centroid = np.sum(freq * psd) / total_power
    # 边缘频率：累积功率达到50%和95%时的频率
    cum = np.cumsum(psd) / total_power
    def find_freq(cum_ratio):
        idx = np.searchsorted(cum, cum_ratio)
        idx = min(idx, len(freq) - 1)
        return freq[idx]
    se50 = find_freq(0.50)
    se95 = find_freq(0.95)
    return {
        "spectral_centroid": centroid,
        "spectral_edge_50":  se50,
        "spectral_edge_95":  se95,
        "spectral_bandwidth": se95 - se50
    }

# =====【修改3-新增MI特征5】左右运动皮层不对称特征：C3/C4通道的功率差与比值
# 这是左手/右手MI最核心的判别特征，左手动→C3(左)ERD更强或C4(右)占优
# 这里作为通用"通道对"函数，后面调用时传入具体的C3/C4、Cz等
def get_channel_asymmetry(band_power_dict_left: dict, band_power_dict_right: dict, name_left: str, name_right: str) -> dict:
    results = {}
    eps = 1e-10
    for band in band_power_dict_left.keys():
        pl = band_power_dict_left[band]
        pr = band_power_dict_right[band]
        # DAR = Differential Asymmetry Ratio (L-R)/(L+R)，范围[-1,1]
        dar = (pl - pr) / (pl + pr + eps)
        # 简单比值
        ratio = pl / (pr + eps)
        # 差值
        diff = pl - pr
        results[f"dar_{name_left}_{name_right}_{band}"] = dar
        results[f"ratio_{name_left}_{name_right}_{band}"] = ratio
        results[f"diff_{name_left}_{name_right}_{band}"]  = diff
    return results


def get_features(df:pd.DataFrame,channels:list):
    fft_result={}
    psd_result={}
    band_power_result={}    
    hjorth_result={}
    entropy_result={}
    interpretation_result={}

    for channel in channels:
        fft_result[channel]=get_fft(df,channel,config.SAMPLING_RATE)
        psd_result[channel]=get_psd(df,channel,config.SAMPLING_RATE)
        band_power_result[channel]=get_band_power(psd_result[channel],config.bands)
        hjorth_result[channel]=get_hjorth(df,channel)
        entropy_result[channel]=get_entropy(df,channel)
        interpretation_result[channel]=get_interpretation(band_power_result[channel],hjorth_result[channel],entropy_result[channel],channel)
        

    analysis_result={
        'basic':{
            'dataset':
                get_dataset_info(df),
            'statistics':
                get_statistics(df),
            'missing_values':
                get_missing_values(df),
            'data_type':
                get_data_type(df)
        },
        'features':{
            'time_features':get_time_domain_features(df),
            'band_power':band_power_result,
            'hjorth':hjorth_result,
            'entropy':entropy_result

        },
        'signals':{
            'fft':fft_result,
            'psd':psd_result
        },
        
        'interpretation': interpretation_result

    }
    return analysis_result

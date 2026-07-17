import  pandas as pd
import config
import analyser
import numpy as np

def split_windows(df:pd.DataFrame):
    windows=[]
    for start in range(0,len(df)-config.window_size+1,config.window_size): # 防止出现不足一个size的情况
        end=start+config.window_size
        window=df.iloc[start:end]
        windows.append(window)
    return windows

def creat_feature_dataframe(windows:list):
    features=[]
    for window in windows:
        # 一行就是一个样本特征即一次循环
        sample={}
        for channel in config.EEG_CHANNELS:
            # 时域
            feature=analyser.get_time_domain_features(window[channel])
            for key,value in feature.items():
                sample[f'{channel}_time_domain_{key}']=value
            # FFT
            fft_result=analyser.get_fft(window,channel,config.SAMPLING_RATE)
            peak_freq=fft_result['frequency'][np.argmax(fft_result['amplitude'])]
            sample[f'{channel}_peak_frequency']=peak_freq
            # bandpower
            psd_result=analyser.get_psd(window,channel,config.SAMPLING_RATE)
            band_power=analyser.get_band_power(psd_result,config.bands)          
            for key,value in band_power.items():
                sample[f'{channel}_band_power_{key}']=value
            # Hjorth
            hjorth=analyser.get_hjorth(window,channel)
            for key,value in hjorth.items():
                sample[f'{channel}_hjorth_{key}']=value
            # entropy
            sample[f'{channel}_entropy']=analyser.get_entropy(window,channel)
        # 一个窗口结束，下一行
        features.append(sample)
    return pd.DataFrame(features)

            
            
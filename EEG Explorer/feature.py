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

def create_feature_dataframe(windows:list,channels:list):
    features=[]
    for window in windows:
        # 一行就是一个样本特征即一次循环
        sample={}
        time_features=analyser.get_time_domain_features(window)
        # 时域，放在循环外面是如果多通道会循环调用get_time_domain_features，增大计算量
        # window是df传入get_time_domain_features得到的是两层字典：各个通道，通道对应的特征值，需要两层循环，第一层先取通道和特征值字典，再循环取出特征值名称和值
        for ch,feature in time_features.items():
            for key,value in feature.items():
                sample[f'{ch}_time_domain_{key}']=value
        
        for channel in channels:
            # FFT
            fft_result=analyser.get_fft(window,channel,config.SAMPLING_RATE)
            freq=fft_result['frequency']
            amp=fft_result['amplitude']
            mask=(freq>=1)&(freq<=40)
            peak_freq=freq[mask][np.argmax(amp[mask])]
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
            entropy=analyser.get_entropy(window,channel)
            for key,value in entropy.items():
                sample[f'{channel}_entropy_{key}']=value
            # 一个窗口结束，下一行
        features.append(sample)
        feature_df = pd.DataFrame(features)
        feature_df = feature_df.astype(float)
        return feature_df

            
            
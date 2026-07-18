import reader  
import analyser 
import visualization 
import report 
import numpy as np
import config
import pandas as pd
import preprocessing
import feature
import channel
def load_data(path: str):
    return reader.read_edf(path)
def main():
    # raw->preprocess->get_features / windows(两个分支)
    raw_df = load_data(r'EEG Explorer\data\S001R01.edf')
    if raw_df is None:
        return
    raw_df = channel.normalize_channel_names(raw_df)
    channels=channel.get_available_channels(raw_df)
    clean_df,preprocess_result=preprocess(raw_df,channels)
    analysis_result = get_features(clean_df,channels)
    windows=feature.split_windows(clean_df)
    feature_df=feature.create_feature_dataframe(windows,channels)
    print(feature_df.shape)
    print(feature_df.head())
    print(feature_df.dtypes.value_counts())
    visualization_result = get_visualization(preprocess_result, analysis_result,channels)
    generate_report(
        analysis_result,
        visualization_result,
        "report.md"
    )
def preprocess(df:pd.DataFrame,channels:list):
    preprocess_result={}
    # 两次循环放在下一次循环将之前的覆盖，现在会保留raw，bandpass，notch结果
    preprocess_result['raw']=df.copy()
    for channel in channels:
        df=preprocessing.apply_bandpass_filter(df,channel,config.LOWCUT,config.HIGHCUT,config.SAMPLING_RATE)
    preprocess_result['bandpass']=df.copy()
    for channel in channels:
        df=preprocessing.apply_notch_filter(df,channel,config.NOTCH_FREQ,config.SAMPLING_RATE)
    preprocess_result['notch']=df.copy()
    return df,preprocess_result
def generate_report(analysis_result:dict,visualization_result:dict,report_name:str):
    report.generate_report(analysis_result,visualization_result,report_name)

def get_features(df:pd.DataFrame,channels:list):
    fft_result={}
    psd_result={}
    band_power_result={}    
    hjorth_result={}
    entropy_result={}
    interpretation_result={}

    for channel in channels:
        fft_result[channel]=analyser.get_fft(df,channel,config.SAMPLING_RATE)
        psd_result[channel]=analyser.get_psd(df,channel,config.SAMPLING_RATE)
        band_power_result[channel]=analyser.get_band_power(psd_result[channel],config.bands)
        hjorth_result[channel]=analyser.get_hjorth(df,channel)
        entropy_result[channel]=analyser.get_entropy(df,channel)
        interpretation_result[channel]=analyser.get_interpretation(band_power_result[channel],hjorth_result[channel],entropy_result[channel],channel)
        

    analysis_result={
        'basic':{
            'dataset':
                analyser.get_dataset_info(df),
            'statistics':
                analyser.get_statistics(df),
            'missing_values':
                analyser.get_missing_values(df),
            'data_type':
                analyser.get_data_type(df)
        },
        'features':{
            'time_features':analyser.get_time_domain_features(df),
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

def get_visualization(preprocess_result:dict,analysis_result:dict,channels:list):
    visualization_result={
        'preprocess_figures':{
            'bandpass':[],
            'notch':[]
            
        },
        'time_figures':{
            'line':[],
            'hist':[],
            'hjorth':[]
        },
        'frequency_figures':{
            'fft':[],
            'psd':[],
            'band_power':[],
            'entropy':[]
            
        }
    }

    for channel in channels:        
        visualization_result['time_figures']['line'].append(
            visualization.plot_line(preprocess_result['notch'],channel,f'{channel}_line1.png')
        )
        visualization_result['time_figures']['hist'].append(
            visualization.plot_histogram(preprocess_result['notch'],channel,f'{channel}_hist1.png')
        )
        visualization_result['time_figures']['hjorth'].append(
            visualization.plot_hjorth(analysis_result['features']['hjorth'][channel],channel,f'{channel}_hjorth_parameters.png')
        )
        visualization_result['frequency_figures']['fft'].append(
            visualization.plot_fft(analysis_result['signals']['fft'][channel],channel,f'{channel}_fft1.png')
        )
        visualization_result['frequency_figures']['psd'].append(
            visualization.plot_psd(analysis_result['signals']['psd'][channel],channel,f'{channel}_psd1.png')
        )
        visualization_result['frequency_figures']['band_power'].append(
            visualization.plot_band_power(analysis_result['features']['band_power'][channel],channel,f'{channel}_band_power.png')
        )
        visualization_result['frequency_figures']['entropy'].append(
            visualization.plot_entropy(analysis_result['features']['entropy'][channel],channel,f'{channel}_entropy.png')
        )
        visualization_result['preprocess_figures']['bandpass'].append(
            visualization.plot_bandpass(preprocess_result['raw'],preprocess_result['bandpass'],channel,f'{channel}_bandpass.png')
        )
        visualization_result['preprocess_figures']['notch'].append(
            visualization.plot_notch(preprocess_result['bandpass'],preprocess_result['notch'],channel,f'{channel}_notch.png')
        )


    return visualization_result


if __name__ == '__main__':
    # path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main()



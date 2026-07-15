import reader  
import analyser 
import visualization 
import report 
import numpy as np
import config
import pandas as pd
import preprocessing
def load_data(path: str):
    return reader.read_csv(path)
def main():
    path = config.EEGLZ
    df = load_data(path)
    if df is None:
        return
    df=preprocess(df)
    analysis_result = get_features(df)
    visualization_result = get_visualization(df, analysis_result)
    generate_report(
        analysis_result,
        visualization_result,
        "report.md"
    )
    
def preprocess(df:pd.DataFrame):
    for channel in config.EEG_CHANNELS:
        df=preprocessing.apply_bandpass_filter(df,channel,config.LOWCUT,config.HIGHCUT,config.SAMPLING_RATE)
    return df
def generate_report(analysis_result:dict,visualization_result:dict,report_name:str):
    report.generate_report(analysis_result,visualization_result,report_name)

def get_features(df:pd.DataFrame):
    fft_result={}
    psd_result={}
    band_power_result={}    
    hjorth_result={}
    entropy_result={}
    interpretation_result={}

    for channel in config.EEG_CHANNELS:
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

def get_visualization(df:pd.DataFrame,analysis_result:dict):
    time_figures=[]
    frequency_figures=[]

    for channel in config.EEG_CHANNELS:        
        time_figures.append(
            visualization.plot_line(df,channel,f'{channel}_line1.png')
        )
        time_figures.append(
            visualization.plot_histogram(df,channel,f'{channel}_hist1.png')
        )
        time_figures.append(
            visualization.plot_hjorth(analysis_result['features']['hjorth'][channel],channel,f'{channel}_hjorth_parameters.png')
        )
        frequency_figures.append(
            visualization.plot_fft(analysis_result['signals']['fft'][channel],channel,f'{channel}_fft1.png')
        )
        frequency_figures.append(
            visualization.plot_psd(analysis_result['signals']['psd'][channel],channel,f'{channel}_psd1.png')
        )
        frequency_figures.append(
            visualization.plot_band_power(analysis_result['features']['band_power'][channel],channel,f'{channel}_band_power.png')
        )
        frequency_figures.append(
            visualization.plot_entropy(analysis_result['features']['entropy'][channel],channel,f'{channel}_entropy.png')
        )

    visualization_result={
        'time_figures':time_figures,
        'frequency_figures':frequency_figures
    }
    return visualization_result


if __name__ == '__main__':
    # path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main()



import matplotlib.pyplot as plt
import config
import pandas as pd
import numpy as np

def plot_line(df:pd.DataFrame,column:str,filename:str):
    plt.figure(figsize=(10,4))
    plt.title(column)
    plt.xlabel('index')
    plt.ylabel(column)
    plt.plot(df[column])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename
def plot_histogram(df:pd.DataFrame,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.hist(df[column]) # 直方图，x连续
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_fft(fft_result:dict,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.plot(fft_result['frequency'],fft_result['amplitude'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_psd(psd_result:dict,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel('frequency')
    plt.ylabel('psd')
    plt.plot(psd_result['frequency'],psd_result['psd'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_band_power(band_power_result:dict,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel('band')
    plt.ylabel('power')
    plt.bar(band_power_result.keys(),band_power_result.values()) # 柱状图，分散的数据
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_hjorth(hjorth_result:dict,column:str,filename:str):
    plt.figure()
    plt.title(f'Hjorth Parameters - {column}')
    plt.xlabel('Hjorth Parameters')
    plt.ylabel('Value')
    plt.bar(list(hjorth_result.keys()),list(hjorth_result.values()))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_entropy(entropy_result:dict,column:str,filename:str):
    plt.figure()
    plt.title(f'Entropy - {column}')
    plt.xlabel('Feature')
    plt.ylabel('Entropy Value')
    plt.bar(list(entropy_result.keys()),list(entropy_result.values()))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_bandpass(raw_df:pd.DataFrame, bandpass_df:pd.DataFrame, column:str, filename:str):
    plt.figure(figsize=(10,4))
    plt.plot(raw_df[column], label="Raw")
    plt.plot(bandpass_df[column], label="Filtered")
    plt.title(f"Band-pass Filter - {column}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

def plot_notch(bandpass_df:pd.DataFrame, notch_df:pd.DataFrame, column:str, filename:str):
    plt.figure(figsize=(10,4))
    plt.plot(bandpass_df[column], label="Raw")
    plt.plot(notch_df[column], label="Filtered")
    plt.title(f"Notch Filter - {column}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

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
            plot_line(preprocess_result['notch'],channel,f'{channel}_line1.png')
        )
        visualization_result['time_figures']['hist'].append(
            plot_histogram(preprocess_result['notch'],channel,f'{channel}_hist1.png')
        )
        visualization_result['time_figures']['hjorth'].append(
            plot_hjorth(analysis_result['features']['hjorth'][channel],channel,f'{channel}_hjorth_parameters.png')
        )
        visualization_result['frequency_figures']['fft'].append(
            plot_fft(analysis_result['signals']['fft'][channel],channel,f'{channel}_fft1.png')
        )
        visualization_result['frequency_figures']['psd'].append(
            plot_psd(analysis_result['signals']['psd'][channel],channel,f'{channel}_psd1.png')
        )
        visualization_result['frequency_figures']['band_power'].append(
            plot_band_power(analysis_result['features']['band_power'][channel],channel,f'{channel}_band_power.png')
        )
        visualization_result['frequency_figures']['entropy'].append(
            plot_entropy(analysis_result['features']['entropy'][channel],channel,f'{channel}_entropy.png')
        )
        visualization_result['preprocess_figures']['bandpass'].append(
            plot_bandpass(preprocess_result['raw'],preprocess_result['bandpass'],channel,f'{channel}_bandpass.png')
        )
        visualization_result['preprocess_figures']['notch'].append(
            plot_notch(preprocess_result['bandpass'],preprocess_result['notch'],channel,f'{channel}_notch.png')
        )


    return visualization_result

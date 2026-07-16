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

import matplotlib.pyplot as plt
from analyser import *
import pandas as pd

def plot_line(df:pd.DataFrame,column:str):
    plt.figure()
    plt.title(column)
    plt.xlabel('index')
    plt.ylabel(column)
    plt.plot(df[column])
def plot_histogram(df:pd.DataFrame,column:str):
    plt.figure()
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.hist(df[column])
def save_plot(filename:str):
    plt.savefig(f'D:/BCI/EEG Explorer/output/figures/{filename}')
    plt.close()
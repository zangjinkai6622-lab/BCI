import matplotlib.pyplot as plt
import config
import pandas as pd

def plot_line(df:pd.DataFrame,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel('index')
    plt.ylabel(column)
    plt.plot(df[column])
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename
def plot_histogram(df:pd.DataFrame,column:str,filename:str):
    plt.figure()
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.hist(df[column])
    plt.savefig(config.FIGURE_DIR/filename)
    plt.close()
    return filename

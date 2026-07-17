import  pandas as pd
import config
def split_windows(df, window_size):
    windows=[]
    for start in range(0,len(df),window_size):
        end=start+window_size
        window=df.iloc[start:end]
        windows.append(window)
    return windows



        

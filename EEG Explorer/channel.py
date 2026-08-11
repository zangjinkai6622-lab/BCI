import pandas as pd


def normalize_channel_names(df: pd.DataFrame):
    new_columns = []
    for col in df.columns:
        # 去掉结尾的 .
        col = col.rstrip(".")
        # 去掉 BCI IV 2a 等数据集的 "EEG-" 前缀 -> 统一成 C3/C4/Cz/Fz/Pz 等标准命名
        if col.startswith("EEG-"):
            col = col[len("EEG-"):]
        new_columns.append(col)
    df.columns = new_columns
    return df

def get_available_channels(df):
    channels = [
        c for c in df.columns
        if c != "time"                  # 排除时间列
        and not c.startswith("EOG")     # 排除眼电通道：EOG-left / EOG-central / EOG-right
    ]
    return channels
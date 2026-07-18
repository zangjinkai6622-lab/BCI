import pandas as pd


def normalize_channel_names(df: pd.DataFrame):

    new_columns = []

    for col in df.columns:

        # 去掉结尾的 .
        col = col.rstrip(".")

        new_columns.append(col)


    df.columns = new_columns

    return df

def get_available_channels(df):

    exclude=[
        "time"
    ]

    channels=[
        c for c in df.columns
        if c not in exclude
    ]

    return channels
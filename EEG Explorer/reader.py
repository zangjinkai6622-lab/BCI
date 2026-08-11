import pandas as pd
import mne
import logging
logger = logging.getLogger(__name__)
"""
Read csv file.

Args:
    path: csv path

Returns:
    pandas.DataFrame
"""
# 开始->检查文件是否存在->try读取->except处理错误->读取成功->打印：shape,columns->返回DataFrame

def read_csv(path: str):
    try:
        df=pd.read_csv(path,encoding='utf-8')
    except FileNotFoundError:
        logger.error("File not found.")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {path}")
        return None
    except UnicodeDecodeError:
        logger.error(f"Encoding error: {path}")
        return None
    return df

def read_edf(path: str):
    try:
        raw=mne.io.read_raw_edf(path,preload=True,encoding='utf-8')
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {path}")
        return None
    except UnicodeDecodeError:
        logger.error(f"Encoding error: {path}")
        return None
    df=raw.to_data_frame()
    return raw, df

def read_gdf(path: str):
    try:
        raw=mne.io.read_raw_gdf(path,preload=True) # 没有encoding参数
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {path}")
        return None
    except UnicodeDecodeError:
        logger.error(f"Encoding error: {path}")
    df=raw.to_data_frame()
    return raw,df


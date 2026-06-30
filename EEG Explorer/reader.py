import pandas as pd
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
        print("Error: file not found.")
        return None

    except pd.errors.EmptyDataError:
        print("Error: empty file.")
        return None

    except UnicodeDecodeError:
        print("Error: encoding error.")
        return None
    return df
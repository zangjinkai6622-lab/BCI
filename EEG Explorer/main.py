from reader import read_csv
from analyser import *


def main(path: str):
    df=read_csv(path)
    if df is None:
        return
    # print(df.shape)
    # print(df.columns)
    # print("读取成功")
    rows,columns=get_shape(df)
    print("rows:",rows)
    print("columns:",columns)
    print(get_columns(df))
    print(get_missing_values(df))
    print(get_data_type(df))
    print(get_statistics(df))


if __name__ == '__main__':
    path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main(path)



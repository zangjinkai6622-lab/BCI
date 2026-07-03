from reader import read_csv
from analyser import *
from visualization import *

def main(path: str):
    df=read_csv(path)
    # Day1
    if df is None:
        return
    # print(df.shape)
    # print(df.columns)
    # print("读取成功")
    # Day2
    # rows,columns=get_shape(df)
    # print("rows:",rows)
    # print("columns:",columns)
    # print(get_columns(df))
    # print(get_missing_values(df))
    # print(get_data_type(df))
    # print(get_statistics(df))

    plot_line(df,'Age')    
    save_plot('age_line.png')
    # plt.show()
    plot_histogram(df,'Age')    
    save_plot('age_histe.png')
    # plt.show()


if __name__ == '__main__':
    path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main(path)



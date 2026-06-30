from reader import read_csv

if __name__ == '__main__':
    df=read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print(df.shape)
    print(df.columns)
    print("读取成功")

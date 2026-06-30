from reader import read_csv



def main():
    path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df=read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    if df is None:
        return
    print(df.shape)
    print(df.columns)
    print("读取成功")
if __name__ == '__main__':
    main()



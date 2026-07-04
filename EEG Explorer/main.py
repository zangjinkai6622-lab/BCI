from reader import read_csv
import analyser 
import visualization 
import report


def main(path: str):
    df=read_csv(path)
    if df is None:
        return
    statistics=analyser.get_statistics(df)
    missing_values=analyser.get_missing_values(df)
    data_type=analyser.get_data_type(df)
    age_line1= visualization.plot_line(df,'Age','age_line1.png')
    age_hist1= visualization.plot_histogram(df,'Age','age_hist1.png')
    figures=[age_line1,age_hist1]
    analysis_result={
        'statistics':statistics,
        'missing_values':missing_values,
        'data_type':data_type,
        'figures':figures
    }
    report.generate_report(analysis_result)
    


if __name__ == '__main__':
    path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main(path)



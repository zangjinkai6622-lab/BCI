import reader  
import analyser 
import visualization 
import report

def main(path: str):
    df=reader.read_csv(path)
    if df is None:
        return


    analysis_result={
        'dataset':
            analyser.get_dataset_info(df),
        'statistics':
            analyser.get_statistics(df),
        'missing_values':
            analyser.get_missing_values(df),
        'data_type':
            analyser.get_data_type(df),
        'figures':[visualization.plot_line(df,'Age','age_line1.png'),
                  visualization.plot_histogram(df,'Age','age_hist1.png')
                  ],
        'interpretation':[],
        'time_features':analyser.get_time_domain_features(df),
    }
    report.generate_report(analysis_result,'report.md')
    
if __name__ == '__main__':
    path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    main(path)



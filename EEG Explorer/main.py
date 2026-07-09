import reader  
import analyser 
import visualization 
import report 
import numpy as np
import config
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
        'figures':[visualization.plot_line(df,'Fp1','Fp1_line1.png'),
                  visualization.plot_histogram(df,'Fp1','Fp1_hist1.png')
                  ],
        'interpretation':[],
        'time_features':analyser.get_time_domain_features(df),
    }
    report.generate_report(analysis_result,'report.md')

    result=analyser.get_fft(
        df,
        "Fp1",
        100
    )
    idx=np.argmax(result["amplitude"])

    print(result["frequency"][idx])

if __name__ == '__main__':
    # path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    path=config.EEGLZ
    main(path)



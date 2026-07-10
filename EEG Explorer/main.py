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
    fft_result=analyser.get_fft(df,'Fp1',100)
    psd_result=analyser.get_psd(df,'Fp1',100)


    analysis_result={
        'dataset':
            analyser.get_dataset_info(df),
        'statistics':
            analyser.get_statistics(df),
        'missing_values':
            analyser.get_missing_values(df),
        'data_type':
            analyser.get_data_type(df),
        'interpretation':[],
        'time_features':analyser.get_time_domain_features(df),
        'fft':fft_result,
        'psd':psd_result,

    }

    visualization_result={
        'time_domain_features':[
            visualization.plot_line(df,'Fp1','Fp1_line1.png'),
            visualization.plot_histogram(df,'Fp1','Fp1_hist1.png'),
        ],
        'frequency_domain_features':[
            visualization.plt_fft(fft_result,'Fp1','Fp1_fft1.png'),
            visualization.plt_fft(psd_result,'Fp1','Fp1_psd1.png')
        ]
    }

    report.generate_report(analysis_result,visualization_result,'report.md')


if __name__ == '__main__':
    # path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    path=config.EEGLZ
    main(path)



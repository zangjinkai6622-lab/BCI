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

    fft_result={}
    psd_result={}
    band_power_result={}
    for channel in config.EEG_CHANNELS:
        fft_result[channel]=analyser.get_fft(df,channel,config.SAMPLING_RATE)
        psd_result[channel]=analyser.get_psd(df,channel,config.SAMPLING_RATE)
        band_power_result[channel]=analyser.get_band_power(psd_result[channel],config.bands)

    hjorth_result={}
    for channel in config.EEG_CHANNELS:
        hjorth_result[channel]=analyser.get_hjorth(df,channel)

    analysis_result={
        'basic':{
            'dataset':
                analyser.get_dataset_info(df),
            'statistics':
                analyser.get_statistics(df),
            'missing_values':
                analyser.get_missing_values(df),
            'data_type':
                analyser.get_data_type(df)
        },
        'features':{
            'time_features':analyser.get_time_domain_features(df),
            'band_power':band_power_result,
            'hjorth':hjorth_result
        },
        'signals':{
            'fft':fft_result,
            'psd':psd_result
        },
        'interpretation':[],


        

    }
    
    time_figures=[]
    frequency_figures=[]

    for channel in config.EEG_CHANNELS:        
        time_figures.append(
            visualization.plot_line(df,channel,f'{channel}_line1.png')
        )
        time_figures.append(
            visualization.plot_histogram(df,channel,f'{channel}_hist1.png')
        )
        time_figures.append(
            visualization.plot_hjorth(hjorth_result[channel],channel,f'{channel}_hjorth_parameters.png')
        )
        frequency_figures.append(
            visualization.plot_fft(fft_result[channel],channel,f'{channel}_fft1.png')
        )
        frequency_figures.append(
            visualization.plot_psd(psd_result[channel],channel,f'{channel}_psd1.png')
        )
        frequency_figures.append(
            visualization.plot_band_power(band_power_result[channel],channel,f'{channel}_band_power.png')
        )

    visualization_result={
        'time_figures':time_figures,
        'frequency_figures':frequency_figures
    }

    report.generate_report(analysis_result,visualization_result,'report.md')


if __name__ == '__main__':
    # path='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    path=config.EEGLZ
    main(path)



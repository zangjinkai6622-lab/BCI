import config
import pathlib
import pandas as pd
def generate_report(analysis_result:dict,visualization_result:dict,filename:str):

    with open(config.REPORT_DIR/filename,"w",encoding="utf-8") as file:
        file.write(f"# EEG Explorer Report\n")

        file.write(f"## 1. Dataset Information\n")    
        file.write(f"Rows: {analysis_result['basic']['dataset']['rows']}\n")
        file.write(f"Columns: {analysis_result['basic']['dataset']['columns']}\n")

        file.write(f"## 2. Statistics\n\n")
        file.write(analysis_result['basic']['statistics'].to_markdown()+"\n")

        file.write(f"## 3. Missing Values\n")
        file.write(f"{analysis_result['basic']['missing_values'].to_markdown()}\n")

        file.write(f"## 4. Data Types\n")               
        file.write(f"{analysis_result['basic']['data_type'].to_markdown()}\n") 

        file.write(f"## 5. Time Domain Features\n")
        time_feature_df=pd.DataFrame(analysis_result['features']['time_features']).T # 行是通道，列是特征
        file.write(f"{time_feature_df.to_markdown()}")
        file.write("\n\n")     

        file.write(f"## 6. Hjorth Parameters\n")      
        hjorth_df=pd.DataFrame(analysis_result['features']['hjorth']).T     
        file.write(f"{hjorth_df.to_markdown()}")
        file.write("\n\n")     

        file.write("## 7. Entropy Features\n")
        entropy_df=pd.DataFrame(analysis_result['features']['entropy']).T
        file.write(f"{entropy_df.to_markdown()}")
        file.write("\n\n")

        file.write(f"## 8.-Time-Figures\n")
        for name in visualization_result['time_figures']:
            file.write(f"- ![](../figures/{name})\n")            


        # 字典不能直接.to_markdown()，先转成df，再.to_markdown()
        file.write(f"## 9. frequency-domain features\n")
        fft_df=pd.DataFrame(analysis_result['signals']['fft'])
        file.write(f"{fft_df.to_markdown()}")
        file.write("\n\n")
        psd_df=pd.DataFrame(analysis_result['signals']['psd'])
        file.write(f"{psd_df.to_markdown()}")
        file.write("\n\n")

        file.write(f"## 10. Frequency domain figures\n")
        for name in visualization_result['frequency_figures']:
            file.write(f"- ![](../figures/{name})\n")    
        file.write(f"## 11. Band Power\n")
        band_power_df=pd.DataFrame(analysis_result['features']['band_power'])
        file.write(f"{band_power_df.to_markdown()}")
        file.write("\n\n")
        file.write(f"## 12. Interpretatio\n")
        file.write(f"{analysis_result['interpretation']}\n")
        file.write(f"## 13. Conclusion\n")

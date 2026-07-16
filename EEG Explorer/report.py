import config
import pathlib
import pandas as pd
def generate_report(analysis_result:dict,visualization_result:dict,filename:str):

    with open(config.REPORT_DIR/filename,"w",encoding="utf-8") as file:
        file.write(f"# EEG Explorer Report\n")

        file.write(f"## 1. Dataset Information\n")    
        file.write(f"Rows: {analysis_result['basic']['dataset']['rows']}\n")
        file.write(f"Columns: {analysis_result['basic']['dataset']['columns']}\n")

        file.write(f"## 2. Preprocessing\n")
        file.write(f"### 2.1 Band-pass Filter\n")
        file.write(f"- ![](../figures/{visualization_result['preprocess_figures']['bandpass']})\n")    
        file.write("\n\n")

        file.write(f"### 2.2 Notch Filter\n")
        file.write(f"- ![](../figures/{visualization_result['preprocess_figures']['notch']})\n")    
        file.write("\n\n")

        file.write(f"## 3. Statistics\n\n")
        file.write(analysis_result['basic']['statistics'].to_markdown()+"\n")

        file.write(f"## 4. Missing Values\n")
        file.write(f"{analysis_result['basic']['missing_values'].to_markdown()}\n")

        file.write(f"## 5. Data Types\n")               
        file.write(f"{analysis_result['basic']['data_type'].to_markdown()}\n") 

        file.write(f"## 6. Time Domain Features\n")
        time_feature_df=pd.DataFrame(analysis_result['features']['time_features']).T # 行是通道，列是特征
        file.write(f"{time_feature_df.to_markdown()}")
        file.write("\n\n")     

        file.write(f"## 7. Hjorth Parameters\n")      
        hjorth_df=pd.DataFrame(analysis_result['features']['hjorth']).T     
        file.write(f"{hjorth_df.to_markdown()}")
        # file.write(f"- ![](../figures/{visualization_result['time_figures']['hjorth']})\n")    
        file.write("\n\n")     

        file.write("## 8. Entropy Features\n")
        entropy_df=pd.DataFrame(analysis_result['features']['entropy']).T
        file.write(f"{entropy_df.to_markdown()}")
        # file.write(f"- ![](../figures/{visualization_result['frequency_figures']['entropy']})\n")    
        file.write("\n\n")

        file.write(f"## 9.-Time-Figures\n")
        for name in visualization_result['time_figures']:
            file.write(f"- ![](../figures/{name})\n")            


        # 字典不能直接.to_markdown()，先转成df，再.to_markdown()
        file.write(f"## 10. frequency-domain features\n")
        fft_df=pd.DataFrame(analysis_result['signals']['fft'])
        file.write(f"{fft_df.to_markdown()}")
        file.write("\n\n")
        psd_df=pd.DataFrame(analysis_result['signals']['psd'])
        file.write(f"{psd_df.to_markdown()}")
        file.write("\n\n")

        file.write(f"## 11. Frequency domain figures\n")
        for name in visualization_result['frequency_figures']:
            file.write(f"- ![](../figures/{name})\n")    
        file.write(f"## 12. Band Power\n")
        band_power_df=pd.DataFrame(analysis_result['features']['band_power'])
        file.write(f"{band_power_df.to_markdown()}")
        file.write("\n\n")
        file.write(f"## 13. Interpretatio\n")
        interpretation=analysis_result['interpretation']
        for channel, texts in interpretation.items():
            file.write(f"### {channel}\n")
            for text in texts:
                file.write(f"- {text}\n")
        file.write(f"## 14. Conclusion\n")

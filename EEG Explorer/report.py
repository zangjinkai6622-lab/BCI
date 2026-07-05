import config
import pathlib
def generate_report(analysis_result:dict,filename:str):

    with open(config.REPORT_DIR/filename,"w",encoding="utf-8") as file:
        file.write(f"# EEG Explorer Report\n")
        file.write(f"## 1. Dataset Information\n")    
        file.write(f"Rows: {analysis_result['dataset']['rows']}\n")
        file.write(f"Columns: {analysis_result['dataset']['columns']}\n")
        file.write(f"## 2. Statistics\n\n")
        file.write(analysis_result['statistics'].to_markdown()+"\n")
        file.write(f"## 3. Missing Values\n")
        file.write(f"{analysis_result['missing_values']}\n")
        file.write(f"## 4. Data Types\n")
        file.write(f"{analysis_result['data_type']}\n")
        file.write(f"## 5. Figures\n")
        for name in analysis_result['figures']:
            file.write(f"- ![](../figures/{name})\n")
        
        file.write(f"## 6. Interpretatio\n")
        file.write(f"EEG feature analysis will be added in future versions.\n")
        file.write(f"## 7. Conclusion\n")

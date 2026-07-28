import machine_learning
import data_pipeline
import numpy as np
import label
import prediction as pred
import pathlib

# 运用已有模型进行预测
def predict_file(file_path, model_name:str="svm_v1"):
    feature_df = data_pipeline.extract_feature(file_path)
    prediction = machine_learning.predict_one_sample(feature_df,model_name)
    final_label = vote_prediction(prediction)
    prediction_name = label.decode_label(final_label)
    print(prediction_name)
    pred.save_prediction_md(file_name=file_path,model_name=model_name,prediction=prediction_name)
    
    return {
        "file": pathlib.Path(file_path).name,
        "model": model_name
    }
# 字典的get方法，如果key不存在，则返回默认值，key存在则返回对应的value

# 统计最多类别，最终返回
def vote_prediction(predictions:np.array):
    unique_values,counts=np.unique(predictions,return_counts=True)
    result=dict(zip(unique_values,counts))
    max_key=max(result,key=result.get)
    return max_key

if __name__=="__main__":
    file="EEG Explorer/data/S001R05.edf"
    result=predict_file(file)
    pred.save_prediction_csv([result])
    print(result["prediction"])
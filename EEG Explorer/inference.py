import machine_learning
import data_pipeline
import numpy as np
# 运用已有模型进行预测
def predict_file(file_path:str):
        feature_df=data_pipeline.extract_feature(file_path)
        prediction=machine_learning.pridict_one_sample(feature_df)
        return prediction

# 将预测结果转化为文字
def decode_prediction(label:int):
    label_map = {
        0: "Rest",
        1: "Left Fist",
        2: "Right Fist"
    }
    return label_map.get(label, "Unknown")
# 字典的get方法，如果key不存在，则返回默认值，key存在则返回对应的value

# 统计最多类别，最终返回
def vote_prediction(predictions:np.array):
    unique_values,counts=np.unique(predictions,return_counts=True)
    result=dict(zip(unique_values,counts))
    max_key=max(result,key=result.get)
    return max_key

def predict_file(file_path):
    feature_df=data_pipeline.extract_feature(file_path)
    predictions=machine_learning.predict_one_sample(feature_df,"svm_v1")
    final_label=vote_prediction(predictions)
    result=decode_prediction(final_label)
    return result
import machine_learning
import logging
logger = logging.getLogger(__name__)
import data_pipeline
import numpy as np
import label
import prediction as pred
import pathlib
import config

# 运用已有模型进行预测
def predict_file(file_path, model_name:str=config.DEFAULT_MODEL):
    feature_df = data_pipeline.extract_feature(file_path)
    prediction = machine_learning.predict_one_sample(feature_df,model_name)        
    logger.info("Prediction Details:")
    for i, p in enumerate(prediction):
        logger.info(
            "Trial %02d : %s",
            i + 1,
            label.decode_label(p)
        )
    logger.info("Prediction Summary:")
    unique, counts = np.unique(prediction, return_counts=True)
    for cls, cnt in zip(unique, counts):
        logger.info(
            "%-12s : %d",
            label.decode_label(cls),
            cnt
        )
    final_label = vote_prediction(prediction)
    prediction_name = label.decode_label(final_label)

    pred.save_prediction_md(file_name=file_path,model_name=model_name,prediction=prediction_name)
    
    return {
        "file": pathlib.Path(file_path).name,
        "prediction": prediction_name,
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
    file="EEG Explorer/data/S001R14.edf"
    result=predict_file(file)
    pred.save_prediction_csv([result])
    print(result["prediction"])
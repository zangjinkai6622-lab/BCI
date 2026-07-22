import numpy as np
import pandas as pd
import inference
import machine_learning
import data_pipeline
import glob
import label

def train_main():
    files=glob.glob('EEG Explorer/data/*.edf')
    features_list=[]

    for file in files:
        result = data_pipeline.process_one_file(file)
        feature_df, preprocess_result, analysis_result, visualization_result = result
        feature_df = result[0]
        feature_df['label']=label.get_label(file)
        features_list.append(feature_df)
    dataset=pd.concat(features_list,axis=0,ignore_index=True) # 行拼接
    model,scaler,pca=machine_learning.train_pipeline(dataset,'svm_v1')

def inference_main():
    file='EEG Explorer/data/test.edf'
    result=inference.predict_file(file)
    return result



if __name__ == '__main__':
    mode=input("train or predict:")
    if mode=="train":
        train_main()
    elif mode=="predict":
        result=inference.predict_file("EEG Explorer/data/test.edf")
        print(result)

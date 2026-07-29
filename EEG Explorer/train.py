import glob
import pandas as pd
import data_pipeline
import label
import machine_learning
import config

def train_main():
    files=glob.glob("EEG Explorer/data/*.edf")[:3]
    features_list=[]
    for file in files:
        result=data_pipeline.process_one_file(file)
        if result is None:
            continue
        feature_df=result[0]
        feature_df['label']=label.get_label(file)
        features_list.append(feature_df)
    dataset=pd.concat(features_list,axis=0,ignore_index=True)
    machine_learning.train_pipeline(dataset=dataset,model_type='svm', model_name=config.DEFAULT_MODEL)

def train_all(dataset:pd.DataFrame):
    for model_name,model_type in config.MODEL_LIST:
        model=machine_learning.create_model(model_type)
        machine_learning.train_pipeline(dataset,model,model_name)

if __name__=="__main__":
    train_main()
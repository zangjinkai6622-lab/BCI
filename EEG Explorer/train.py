import glob
import pandas as pd
import data_pipeline
import label
import machine_learning

def train_main():
    files=glob.glob("EEG Explorer/data/*.edf")
    features_list=[]
    for file in files:
        result=data_pipeline.process_one_file(file)
        if result is None:
            continue
        feature_df=result[0]
        feature_df['label']=label.get_label(file)
        features_list.append(feature_df)
    dataset=pd.concat(features_list,axis=0,ignore_index=True)
    machine_learning.train_pipeline(dataset,"svm_v1")



if __name__=="__main__":
    train_main()
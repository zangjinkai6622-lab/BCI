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
    print(dataset.drop("label",axis=1).shape)
    model=machine_learning.create_model("svm")
    machine_learning.train_pipeline(dataset=dataset,model=model, model_name="svm_v1")


if __name__=="__main__":
    train_main()
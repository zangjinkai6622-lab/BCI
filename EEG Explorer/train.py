import glob
import pandas as pd
import data_pipeline
import label
import machine_learning
import config

def train_main():
    files=glob.glob("EEG Explorer/data/*.edf")[:13]
    features_list=[]
    for file in files:
        result=data_pipeline.process_one_file(file)
        if result is None:
            continue
        feature_df=result['feature_df']
        features_list.append(feature_df)
        print(file)
        print(feature_df["label"].value_counts().sort_index())
        print("-" * 40)
    dataset=pd.concat(features_list,axis=0,ignore_index=True)
    print("=" * 50)
    print("Dataset Shape")
    print(dataset.shape)

    print("=" * 50)
    print("Feature Preview")
    print(dataset.iloc[:5, :10])

    print("=" * 50)
    print("Feature Mean By Label")
    print(dataset.groupby("label").mean().iloc[:, :10])

    print("=" * 50)
    print("NaN Count")
    print(dataset.isna().sum().sum())

    print("=" * 50)
    print("Zero Variance Features")
    print((dataset.drop(columns="label").std() == 0).sum())
    machine_learning.train_pipeline(dataset=dataset,model_type='svm', model_name=config.DEFAULT_MODEL)

def train_all(dataset:pd.DataFrame):
    for model_name,model_type in config.MODEL_LIST:
        model=machine_learning.create_model(model_type)
        machine_learning.train_pipeline(dataset,model_type,model_name)

if __name__=="__main__":
    train_main()
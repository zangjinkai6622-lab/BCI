import glob
import pandas as pd
import data_pipeline
import label
import machine_learning
import config

def train_main():
    all_gdf = sorted(glob.glob(str(config.DATA_DIR / "*.gdf")))
    files = [f for f in all_gdf if f.endswith("T.gdf")]
    features_list=[]
    for file in files:
        result=data_pipeline.extract_feature(file)
        if result is None:
            continue 
        feature_df=result
        features_list.append(feature_df)
    dataset=pd.concat(features_list,axis=0,ignore_index=True)
    machine_learning.train_pipeline(dataset=dataset, model_type='lgbm', model_name='lgbm_v1')
def train_all(dataset:pd.DataFrame):
    for model_name,model_type in config.MODEL_LIST:
        model=machine_learning.create_model(model_type)
        machine_learning.train_pipeline(dataset,model_type,model_name)

if __name__=="__main__":
    train_main()
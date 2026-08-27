import sys
from pathlib import Path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import Generator
import config
import torch
import data_pipeline
def _get_raw_feature_dataframe():
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
    return dataset

def _prep_data_from_dataframe(df):
    X_df = df.drop("label", axis=1)          # 去掉 label 列，剩下全是特征
    y_raw = df["label"].values.astype(int)   # 原始标签 {1, 2, 4}
    X_df = X_df.fillna(0.0)
    le = LabelEncoder()
    le.fit(np.array([1, 2, 4]))
    y_encoded = le.transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values).astype(np.float32)
    return X_scaled, y_encoded, le, scaler    

class EEGFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # 取第 idx 个样本和标签
        return self.X[idx], self.y[idx]  
    
def make_dataloaders(batch_size=32,test_ratio=0.2,val_ratio=0.2,seed=42):
    df = _get_raw_feature_dataframe()
    X_scaled, y_encoded, label_encoder, scaler = _prep_data_from_dataframe(df)
    n_features = X_scaled.shape[1]   
    n_classes = 3                      
    dataset = EEGFeatureDataset(X_scaled, y_encoded)
    n_test = int(len(dataset) * test_ratio)
    n_train_val = len(dataset) - n_test
    gen = Generator().manual_seed(seed)
    train_val_set, test_set = random_split(
        dataset,
        [n_train_val, n_test],
        generator=gen
    )
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_val_set,
        [n_train, n_val],
        generator=gen
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader,val_loader, n_features, n_classes, label_encoder

train_loader, test_loader,val_loader, n_features, n_classes, label_encoder = make_dataloaders()
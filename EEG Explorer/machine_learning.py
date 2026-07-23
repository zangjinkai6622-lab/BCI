import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import joblib
import pathlib
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# 流程： 分离dataset的X，y ---> 分别对X,y分出train、test，随机种子42，分层提取，---> 标准化 ---> 训练模型 ---> 预测，根据X_train预测y_pred ---> 评价，根据y_pred和y_test进行准确性和混淆矩阵的评价
def split_xy(dataset:pd.DataFrame):
    # 测试数据集
    X=dataset.drop('label',axis=1)
    # 标签数据集
    y=dataset['label']
    return X,y

def split_dataset(X:pd.DataFrame,y:pd.DataFrame):
    # X,y原始的数据集，test_size测试的比例，random_state随机数种子，让随机操作的结果每次运行都一样。stratify（分层）标签数据集，在取测试用例时分层取样
    # stratify传入一个数组，按照这个数组进行分层，也就是说同label分层，提取测试用例时，每层所占比例相同，都为test_size
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X_train,X_test,y_train,y_test
    

#  签数据集进行标准化
# def standardize(X_train:pd.DataFrame,X_test:pd.DataFrame):
#     scaler=StandardScaler()
#     X_train=scaler.fit_transform(X_train)
#     X_test=scaler.transform(X_test)
#     return X_train,X_test,scaler
# 被pipeline替代
# 目前数据样本远小于维数，所以需要降维，用pca去降维，将多个重复的列 进行合并
# def apply_pca(X_train:pd.DataFrame,X_test:pd.DataFrame):
#     pca=PCA(n_components=0.95) # 保存95%的信息
#     X_train=pca.fit_transform(X_train) #  先计mean，var这些在转化降维
#     X_test=pca.transform(X_test) # 直接降维，预测阶段不能fit，防止数据泄露：测试集的信息，在训练过程中提前泄漏给模型了。
#     return X_train,X_test,pca

# def train_svm(X_train:pd.DataFrame,y_train:pd.DataFrame):
#     model=SVC(kernel='rbf')
#     model.fit(X_train,y_train)
#     return model

def predict(model,X_test:pd.DataFrame):
    y_pred=model.predict(X_test)
    return y_pred

def evaluate_model(y_test:pd.DataFrame,y_pred:pd.DataFrame):
    accuracy=accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred) #混淆矩阵
    report=classification_report(y_test,y_pred)
    return accuracy,matrix,report
def load_model(model_name:str):
    current_dir = pathlib.Path(__file__).parent
    return joblib.load(current_dir /"models" /f"{model_name}.pkl"
    )

def predict_one_sample(feature_df:pd.DataFrame,model_name:str):
    pipeline=load_model(model_name)
    prediction=pipeline.predict(feature_df)
    return prediction

def train_pipeline(dataset:pd.DataFrame,model_name:str):
    X,y=split_xy(dataset)
    model=SVC(kernel='rbf')
    cross_validate_model(model,X,y)
    #  先划分原始数据，再训练
    X_train,X_test,y_train,y_test=split_dataset(X,y)
    pipeline=build_pipeline(model)
    pipeline.fit(X_train,y_train)
    y_pred=pipeline.predict(X_test)
    accuracy, matrix, report = evaluate_model(y_test,y_pred)
    print("="*40)
    print("Evaluation Result")
    print("="*40)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix")
    print(matrix)
    print("\nClassification Report")
    print(report)
    save_model(pipeline,model_name)
    print(f"Model saved: {model_name}")
    return pipeline

def save_model(model,model_name:str):
    current_dir = pathlib.Path(__file__).parent  # data_pipeline.py 所在目录
    model_dir = current_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,model_dir / f"{model_name}.pkl")

def save_evaluation(accuracy:float,matrix:np.array,report:str,file_name:str):
    with open(file_name,"w",encoding="utf8") as f:
        f.write(f"Accuracy:{accuracy}\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(matrix))
        f.write("\n\n")
        f.write(report)
#  交叉验证(Cross Validation)，只划分一次train和test，结果没有普遍性，需要多次不同的划分
def cross_validate_model(model,X:pd.DataFrame,y:pd.DataFrame):
    pipeline=build_pipeline(model)
    scores=cross_val_score(pipeline,X,y,cv=5)# cv：把总数据划分成几份，其中一份作为test，其余的作为train，反复n次
    print("Cross Validation Scores:")
    print(scores)
    print("Mean:", scores.mean())
    print("Std :", scores.std())
    return scores

def build_pipeline(model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", model)
    ])
    return pipeline
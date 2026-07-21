import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import joblib

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
def standardize(X_train:pd.DataFrame,X_test:pd.DataFrame):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    joblib.dump(scaler,'scaler.pkl')
    return X_train,X_test,scaler

# 目前数据样本远小于维数，所以需要降维，用pca去降维，将多个重复的列 进行合并
def apply_pca(X_train:pd.DataFrame,X_test:pd.DataFrame):
    pca=PCA(n_components=0.95) # 保存95%的信息
    X_train=pca.fit_transform(X_train) #  先计mean，var这些在转化降维
    X_test=pca.transform(X_test) # 直接降维，预测阶段不能fit，防止数据泄露：测试集的信息，在训练过程中提前泄漏给模型了。
    joblib.dump(pca,'pca.pkl')
    return X_train,X_test,pca

def train_svm(X_train:pd.DataFrame,y_train:pd.DataFrame):
    model=SVC(kernel='rbf')
    model.fit(X_train,y_train)
    joblib.dump(model,'model.pkl')
    return model

def predict(model,X_test:pd.DataFrame):
    y_pred=model.predict(X_test)
    return y_pred

def evaluate_model(y_test:pd.DataFrame,y_pred:pd.DataFrame):
    accuracy=accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred) #混淆矩阵
    report=classification_report(y_test,y_pred)
    return accuracy,matrix,report
def load_model():
    model=joblib.load('model.pkl')
    scaler=joblib.load('scaler.pkl')
    pca=joblib.load('pca.pkl')
    return model,scaler,pca

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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


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
def build_pipeline(model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", model)
    ])
    return pipeline  

#  不进行单独的交叉验证
#  网格搜索参数是为了找到最合适的参数，先判断模型合不合适，再找出最好的参数，GridSearchCV内部就是CrossValidation，每个参数都试一下，找到最合适的参数
def grid_search_cv(pipeline:Pipeline,X_train:pd.DataFrame,y_train:pd.Series,model_type:str):
    param_grid = get_param_grid(model_type)
    grid=GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)
    grid.fit(X_train,y_train)
    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_
    )

def evaluate_model(y_test:pd.DataFrame,y_pred:np.ndarray):
    accuracy=accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred) #混淆矩阵
    report=classification_report(y_test,y_pred)
    return accuracy,matrix,report


def save_model(model,model_name:str):
    current_dir = pathlib.Path(__file__).parent  # data_pipeline.py 所在目录
    model_dir = current_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,model_dir / f"{model_name}.pkl")

def load_model(model_name:str):
    current_dir = pathlib.Path(__file__).parent
    pipeline=joblib.load(current_dir /"models" / model_name /f"{model_name}.pkl")
    return pipeline
def save_evaluation(accuracy:float,matrix:np.array,report:str,model_name:str):
    current_dir=pathlib.Path(__file__).parent
    result_dir=current_dir/"models"/model_name
    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(
        result_dir/"evaluation.txt",
        "w",
        encoding="utf8"
        ) as f:
            f.write("=" * 60 + "\n")
            f.write(f"模型评估报告: {model_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("【准确率 (Accuracy)】\n")
            f.write(f"{accuracy:.4f}\n\n")

            f.write("【分类报告 (Classification Report)】\n")
            f.write(report + "\n\n")

            f.write("【混淆矩阵 (Confusion Matrix)】\n")
            f.write(str(matrix) + "\n\n")

def train_pipeline(dataset:pd.DataFrame,model_type:str,model_name:str):
    X,y=split_xy(dataset)    #  先划分原始数据，再训练
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model=create_model(model_type)
    pipeline=build_pipeline(model)
    pipeline,best_params, best_score = grid_search_cv(pipeline, X_train, y_train,model_type)
    y_pred=pipeline.predict(X_test)
    accuracy, matrix, report = evaluate_model(y_test,y_pred)    
    save_model(pipeline,model_name)
    save_evaluation(accuracy,matrix,report,f"{model_name}.txt")
    print("=" * 40)
    print("Training Finished")
    print("=" * 40)
    print(f"Model Name : {model_name}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Best Score : {best_score:.4f}")
    print(f"Best Params: {best_params}")
    print("=" * 40)
    return pipeline

def predict_one_sample(feature_df:pd.DataFrame,model_name:str):
    pipeline=load_model(model_name)
    prediction=pipeline.predict(feature_df)
    return prediction

def create_model(model_type:str):
    if model_type == "svm":
        return SVC(kernel="rbf")

    elif model_type == "rf":
        return RandomForestClassifier(random_state=42)

    elif model_type == "lr":
        return LogisticRegression(max_iter=1000)

    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
def get_param_grid(model_type:str):
    if model_type=="svm":
        return {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.1, 0.01, 0.001]
        }
    elif model_type=="rf":
        return {
            "model__n_estimators":[100,200],
            "model__max_depth":[None,10,20]
        }

    elif model_type=="lr":
        return {
            "model__C":[0.1,1,10]
        }

    else:
        raise ValueError(
            f"Unsupported model type:{model_type}"
        )
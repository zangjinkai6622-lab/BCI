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
import logging
logger = logging.getLogger(__name__)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  # =====【修改2-启用KNN】新增KNN导入
import config


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
    print("Train")
    print(y_train.value_counts().sort_index())
    print("Test")
    print(y_test.value_counts().sort_index())
    return X_train,X_test,y_train,y_test

# =====【修改1-移除PCA】为了验证PCA是否把分类信息压掉，暂时去掉PCA，保留仅做 StandardScaler + model
def build_pipeline(model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        # ("pca", PCA(n_components=0.95)),  # =====【修改1-移除PCA】暂时注释掉PCA步骤
        ("model", model)
    ])
    return pipeline  

#  不进行单独的交叉验证
#  网格搜索参数是为了找到最合适的参数，先判断模型合不合适，再找出最好的参数，GridSearchCV内部就是CrossValidation，每个参数都试一下，找到最合适的参数
def grid_search_cv(pipeline:Pipeline,X_train:pd.DataFrame,y_train:pd.Series,model_type:str):
    param_grid = get_param_grid(model_type)
    grid=GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=1)# jobs=-1调用所有进程，但是temp有中文名，所以不能用temp，改为1
    grid.fit(X_train,y_train)
    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_
    )

def evaluate_model(y_test:pd.DataFrame,y_pred:np.ndarray):
    accuracy=accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred) #混淆矩阵
    report=classification_report(y_test,y_pred,zero_division=0) #防止因为某个类别一次都没有被预测出来而警告
    return accuracy,matrix,report


def save_model(model,model_name:str):
    model_dir = config.MODEL_DIR/ model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

def load_model(model_name:str):
    pipeline=joblib.load(config.MODEL_DIR / model_name / "model.pkl")
    return pipeline
def save_evaluation(accuracy:float,matrix:np.array,report:str,model_name:str):
    result_dir=config.EVALUATION_DIR/model_name
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
    best_pipeline,best_params, best_score = grid_search_cv(pipeline, X_train, y_train,model_type)
    # =====【修改1-移除PCA】pca步骤已暂时移除，因此注释掉下面的PCA信息输出
    # pca = best_pipeline.named_steps["pca"]
    # original_dim = X_train.shape[1]
    # reduced_dim = pca.n_components_
    # logger.info(f"PCA: original dim = {original_dim} -> reduced dim = {reduced_dim} (preserved 95% variance)")
    logger.info(f"Feature dim used (no PCA): {X_train.shape[1]}")  # =====【修改1-移除PCA】改为直接输出原始特征维度
    y_pred=best_pipeline.predict(X_test)
    logger.info("Prediction Distribution:")
    logger.info("\n%s", pd.Series(y_pred).value_counts().sort_index())
    accuracy, matrix, report = evaluate_model(y_test, y_pred)    
    print("=" * 50)
    print("Confusion Matrix")
    print(matrix)
    print("=" * 50)
    save_model(best_pipeline, model_name)
    save_evaluation(accuracy, matrix, report, model_name)

    logger.info("Classification Report:")
    logger.info("\n%s", report)
    logger.info("=" * 40)
    logger.info("Training Finished")
    logger.info("=" * 40)
    logger.info(f"Model Name : {model_name}")
    logger.info(f"Accuracy   : {accuracy:.4f}")
    logger.info(f"Best Score : {best_score:.4f}")
    logger.info(f"Best Params: {best_params}")
    logger.info("=" * 40)
    return {
        "model": best_pipeline,
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "best_score": best_score,
        "best_params": best_params,
        "confusion_matrix": matrix,
        "classification_report": report
    }

def predict_one_sample(feature_df, model_name):
    pipeline = load_model(model_name)
    feature_df = feature_df.drop(columns=["label"],errors="ignore")
    prediction = pipeline.predict(feature_df)
    return prediction

def create_model(model_type:str):
    if model_type == "svm":
        return SVC(kernel="rbf")
    elif model_type == "rf":
        return RandomForestClassifier(random_state=42)
    elif model_type == "lr":
        return LogisticRegression(max_iter=1000)
    elif model_type == "knn":  # =====【修改2-启用KNN】取消注释KNN
        return KNeighborsClassifier()  # =====【修改2-启用KNN】默认n_neighbors=5
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
    elif model_type=="knn":  # =====【修改2-启用KNN】新增KNN的参数网格
        return {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],  # 邻居数量，典型范围
            "model__weights": ["uniform", "distance"],    # 投票权重
            "model__metric": ["euclidean", "manhattan"]   # 距离度量
        }

    else:
        raise ValueError(
            f"Unsupported model type:{model_type}"
        )
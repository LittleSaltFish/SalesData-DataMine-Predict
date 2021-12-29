import random
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Tools.Report import MakeReport

from sklearn.utils import shuffle

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

train_index = list(data.columns.values)
train_index.remove("result")

# ---------------
argument = {"train_size": 0.9, "test_size": 0.1}
# ---------------

x = data[train_index]
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)  # 分割数据

R_tree = RandomForestClassifier(n_estimators =24,min_samples_split = 11,min_samples_leaf = 4,max_depth = 17,)
R_tree.fit(x_train, y_train)

ypred = R_tree.predict_proba(x_test)
# 输出二维数组，0维是0概率，1维是1概率
# y_pred = R_tree.predict(x_test)

# 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
y_pred = (ypred[:, 1] >= 0.5) * 1

print(ypred[:, 1])

(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("RandomForest", y_test, y_pred, argument)

joblib.dump(R_tree, f"./Predict/Model/Randomforest/{F1_Score_value}.pkl")

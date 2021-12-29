from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import random
import pandas as pd
import pydot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from Tools.Report import MakeReport

from sklearn.utils import shuffle

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

train_index = list(data.columns.values)
train_index.remove("result")
# train_index=random.choices(train_index,k=7)

data.fillna(0, inplace=True)

# ---------------
argument = {
    "max_depth": 9,
    "train_size": 0.9,
    "test_size": 0.1,
}

# ---------------

x = data[train_index]
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)  # 分割数据

# 欠采样修正
# rus = RandomUnderSampler(random_state=0)
# x_train, y_train = rus.fit_resample(x_train, y_train)

smo = SMOTE(sampling_strategy=0.05)
x_train, y_train = smo.fit_resample(x_train, y_train)

dtc = DTC(criterion="entropy", max_depth=argument["max_depth"])  # 基于信息熵
dtc.fit(x_train, y_train)

ans = dtc.predict(x_test)
score = str(dtc.score(x_test, y_test))[:5]
report = classification_report(y_test, ans)
print("准确率：\n", score)
print("效果：\n", report)

with open(
    f"Predict/Log/Dot/{argument['max_depth']}-{argument['train_size']}-{argument['test_size']}-{score}.dot",
    "w",
) as f:
    f = export_graphviz(dtc, feature_names=x_train.columns, out_file=f)
(graph,) = pydot.graph_from_dot_file(
    f"Predict/Log/Dot/{argument['max_depth']}-{argument['train_size']}-{argument['test_size']}-{score}.dot"
)
graph.write_png(
    f"Predict/Log/Png/{argument['max_depth']}-{argument['train_size']}-{argument['test_size']}-{score}.png"
)

(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("Tree", y_test, ans, argument)

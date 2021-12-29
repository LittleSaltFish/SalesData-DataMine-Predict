import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Tools.Report import MakeReport

from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

train_index = list(data.columns.values)
train_index.remove("result")

# ---------------
argument = {"train_size": 0.9, "test_size": 0.1}
param_grid = {
    "n_estimators": np.arange(1, 200),
    "max_depth": np.arange(1, 20),
    "min_samples_leaf": np.arange(2, 20),
    "min_samples_split": np.arange(2, 20),
    # "min_samples_split": np.arange(2, 20, step=2),
}
# ---------------

x = data[train_index]
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)  # 分割数据

R_tree = RandomForestClassifier()
random_cv = RandomizedSearchCV(R_tree, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=5,scoring='f1',verbose=2)
random_cv.fit(x_train, y_train)

# ---------------
best = random_cv.best_params_
argument.update(best)
print("Best params:\n")
print(best)

R_tree = RandomForestClassifier(
    n_estimators=best["n_estimators"],
    min_samples_split=best["min_samples_split"],
    min_samples_leaf=best["min_samples_leaf"],
    max_depth=best["max_depth"],
)
R_tree.fit(x_train, y_train)

ypred = R_tree.predict_proba(x_test)
# 输出二位数组，0维是0概率，1维是1概率
y_pred = R_tree.predict(x_test)

print(ypred[:, 1])

(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("RandomForest", y_test, y_pred, argument, ypred[:, 1])

with open(f"./Predict/Model/RF-best.txt", "a+") as f:
    f.write(str(best)+"\n")
    f.write(str(Report))

# with open(f"Predict/Log/Dot/{max_depth}-{train_size}-{test_size}-{score}.dot", "w") as f:
#     f = export_graphviz(R_tree, feature_names=x_train.columns, out_file=f)
# (graph,) = pydot.graph_from_dot_file(f"Predict/Log/Dot/{max_depth}-{train_size}-{test_size}-{score}.dot")
# graph.write_png(f"Predict/Log/Png/{max_depth}-{train_size}-{test_size}-{score}.png")

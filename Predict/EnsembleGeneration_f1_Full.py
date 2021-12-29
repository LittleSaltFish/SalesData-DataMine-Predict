import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import xgboost as XGBR
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from Tools.Report import MakeReport
from sklearn.metrics import f1_score

from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

# ==========================================================
params = {
    "booster": "gbtree",  # 选择每次迭代的模型
    "objective": "binary:logistic",  # 定义学习任务及相应的学习目标
    "eval_metric": "auc",  # 对于有效数据的度量方法。
    "max_depth": 10,  # 最大深度
    "eta": 0.025,  # 学习率
    "seed": 0,  # 随机数种子
}
argument = {"train_size": 0.9, "test_size": 0.1, "n_splits": 3, "num_boost_round": 10}
argument.update(params)
# ==========================================================


def f1_macro(preds, dtrain):
    y_train = dtrain.get_label()  # 'numpy.ndarray'
    y_pred = [np.argmax(d) for d in preds]
    return "f1_macro", metrics.f1_score(y_train, y_pred, average="binary")


# ==========================================================

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

x = data.drop(["result"], axis=1)
y = data["result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)
print("x_train", np.shape(x_train))
print("y_train", np.shape(y_train))
print("x_test", np.shape(x_test))
print("y_test", np.shape(y_test))

y_train.to_csv(
    "./Data/EnsembleGeneration/train.csv", encoding="utf-8-sig", index=False, sep=","
)
y_test.to_csv(
    "./Data/EnsembleGeneration/test.csv", encoding="utf-8-sig", index=False, sep=","
)

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

# For XGB
x_dtest = XGBR.DMatrix(x_test)

ntrain = x_train.shape[0]
ntest = x_test.shape[0]

oof_train_XGB = np.zeros((ntrain))
oof_test_XGB = np.zeros((ntest))
oof_test_skf_XGB = np.empty((5, ntest))

oof_train_RF = np.zeros((ntrain))
oof_test_RF = np.zeros((ntest))
oof_test_skf_RF = np.empty((5, ntest))

# ==========================================================

ModelList = []

kf = KFold(n_splits=argument["n_splits"])
for i, (train_index, test_index) in enumerate(kf.split(x_train)):

    print("train_index", np.shape(train_index))
    print("test_index", np.shape(test_index))

    # XGB
    print(f"XGB-{i}")
    kf_x_train = x_train[train_index]
    kf_y_train = y_train[train_index]
    kf_dtrain = XGBR.DMatrix(kf_x_train, label=kf_y_train)
    print(np.shape(kf_x_train))
    print(np.shape(kf_y_train))

    kf_x_test = x_train[test_index]
    kf_x_dtest = XGBR.DMatrix(kf_x_test)
    print(np.shape(kf_x_test))

    watchlist = [(kf_dtrain, "train")]
    bst = XGBR.train(
        params,
        kf_dtrain,
        num_boost_round=argument["num_boost_round"],
        evals=watchlist,
        feval=f1_macro,
    )
    # 输出概率
    oof_train_XGB[test_index] = bst.predict(kf_x_dtest)
    oof_test_skf_XGB[i, :] = bst.predict(x_dtest)
    y_pred = (bst.predict(x_dtest) >= 0.5) * 1
    (
        AUC_value,
        ACC_value,
        Recall_value,
        F1_Score_value,
        Precesion_value,
        ConfusionMatrix,
        Report,
    ) = MakeReport("XGBoost_f1", y_test, y_pred, {})

    # RF
    print(f"RF-{i}")
    R_tree = RandomForestClassifier(
    )
    R_tree.fit(x_train[train_index], y_train[train_index])
    oof_train_RF[test_index] = R_tree.predict_proba(kf_x_test)[:, 1]
    oof_test_skf_RF[i, :] = R_tree.predict_proba(x_test)[:, 1]
    (
        AUC_value,
        ACC_value,
        Recall_value,
        F1_Score_value,
        Precesion_value,
        ConfusionMatrix,
        Report,
    ) = MakeReport("RandomForest", y_test, R_tree.predict(x_test), {})

    ModelList.append((bst, R_tree))

# ==========================================================
# 第二层模型

oof_x_train = np.zeros((ntrain, 2))
oof_x_test = np.zeros((ntest, 2))

oof_x_train[:, 0] = oof_train_XGB
oof_x_train[:, 1] = oof_train_RF
oof_x_test[:, 0] = oof_test_skf_XGB.mean(axis=0)
oof_x_test[:, 1] = oof_test_skf_RF.mean(axis=0)


# 欠采样优于smote
rus = RandomUnderSampler(random_state=0)
oof_x_train, oof_y_train = rus.fit_resample(oof_x_train, y_train)

clf = LogisticRegression(penalty="l2", C=0.3, max_iter=400, tol=1e-4, solver="lbfgs")
clf.fit(oof_x_train, oof_y_train)

print(oof_x_test)

ans = clf.predict(oof_x_test)
(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("MixLR", y_test, ans, argument)

# ==========================================================
# 保存模型

path = f"./Predict/Model/EnsembleGeneration/{F1_Score_value}"
os.mkdir(path)
joblib.dump(clf, f"{path}/LR.pkl")
for i, (j, k) in enumerate(ModelList):
    joblib.dump(j, f"{path}/XGB-{i}.pkl")
    joblib.dump(k, f"{path}/RF-{i}.pkl")


path = f"./Data/EnsembleGeneration/{F1_Score_value}"
os.mkdir(path)
np.savetxt(
    f"./Data/EnsembleGeneration/{F1_Score_value}/oof_test_XGB.csv",
    oof_test_XGB,
    encoding="utf-8-sig",
    delimiter=",",
    fmt="%.5f",
    header="XGB",
)
np.savetxt(
    f"./Data/EnsembleGeneration/{F1_Score_value}/oof_test_RF.csv",
    oof_test_RF,
    encoding="utf-8-sig",
    delimiter=",",
    fmt="%.5f",
    header="RF",
)
np.savetxt(
    f"./Data/EnsembleGeneration/{F1_Score_value}/oof_train_XGB.csv",
    oof_train_XGB,
    encoding="utf-8-sig",
    delimiter=",",
    fmt="%.5f",
    header="XGB",
)
np.savetxt(
    f"./Data/EnsembleGeneration/{F1_Score_value}/oof_train_RF.csv",
    oof_train_RF,
    encoding="utf-8-sig",
    delimiter=",",
    fmt="%.5f",
    header="RF",
)

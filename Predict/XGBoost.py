import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import time

from Tools.Report import MakeReport
from Tools.XGBoost_Draw import DrawTree

import xgboost as XGBR
from sklearn import metrics

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, SMOTEN

from sklearn.metrics import f1_score

# ==========================================================

params = {
    "booster": "gbtree",  # 选择每次迭代的模型
    "objective": "binary:logistic",  # 定义学习任务及相应的学习目标
    "eval_metric": "auc",  # 对于有效数据的度量方法。
    "max_depth": 5,  # 最大深度
    # "lambda": 10,  # L2正则化的权重
    # "subsample": 0.75,
    # 子样本数目。是否只使用部分的样本进行训练，这可以避免过拟合化。默认为1，即全部用作训练。
    # "colsample_bytree": 0.75,  # 每棵树的列数（特征数）
    # "min_child_weight": 2,
    "eta": 0.005,  # 学习率
    "seed": 0,  # 随机数种子
    # ====================================
    # "nthread": 8,  #  并发线程数
    # "silent": 1, # 0表示输出运行信息，1表示采取静默模式
    # "scale_pos_weight": weight,  # 用于不均衡分类
}

argument = {
    "train_size": 0.9,
    "test_size": 0.1,
    "sampling_strategy": 0.05,
    "num_boost_round": 7,
}
argument.update(params)

# ==========================================================

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

x = data.drop(["result"], axis=1)
y = data["result"]

# 分割数据
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)
# ==========================================================

# 欠采样修正
# rus = RandomUnderSampler(random_state=0)
# x_train, y_train = rus.fit_resample(x_train, y_train)

# 过采样修正
smo = SMOTE(sampling_strategy=argument["sampling_strategy"])
x_train, y_train = smo.fit_resample(x_train, y_train)

# smo=ADASYN()
# x_train, y_train = smo.fit_resample(x_train, y_train)

# 联合修正
# smote_tomek = SMOTETomek()
# x_train, y_train = smote_tomek.fit_resample(x_train, y_train)

# smote_enn = SMOTEENN()
# x_train, y_train = smote_enn.fit_resample(x_train, y_train)
# ==========================================================

dtrain = XGBR.DMatrix(x_train, label=y_train)
dtest = XGBR.DMatrix(x_test)


watchlist = [(dtrain, "train")]
bst = XGBR.train(
    params, dtrain, num_boost_round=argument["num_boost_round"], evals=watchlist
)
# 输出概率
ypred = bst.predict(dtest)

# 设置阈值, 输出一些评价指标，选择概率大于0.5的为1，其他为0类
y_pred = (ypred >= 0.5) * 1
(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("XGBoost", y_test, y_pred, argument)
# DrawTree(bst,f1)

# 保存模型
bst.save_model(f"./Predict/Model/XGBoost/test-{F1_Score_value}.model")
# 导出模型到文件
bst.dump_model(f"./Predict/Model/XGBoost/dump-{F1_Score_value}.raw.txt")
# 导出模型和特征映射
# bst.dump_model('./Predict/Model/dump.raw.txt','./Predict/Model/featmap.txt')

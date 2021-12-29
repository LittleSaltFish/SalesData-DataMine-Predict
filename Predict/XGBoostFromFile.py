import pandas as pd
from Tools.Report import MakeReport
import xgboost as XGBR
from Tools.XGBoost_Draw import DrawTree

modelame = "test-0.78941"
dataname = "Data4Predict4XGB"

data = pd.read_csv(f"./Data/{dataname}.csv")

x = data.drop(["result"], axis=1)
y = data["result"]
x = XGBR.DMatrix(x)
bst = XGBR.Booster(model_file=f"./Predict/Model/XGBoost/{modelame}.model")
ypred = bst.predict(x)

# 设置阈值, 输出一些评价指标，选择概率大于阈值的为1，其他为0类
y_pred = (ypred >= 0.4896) * 1

(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("XGBoost", y, y_pred, {}, ypred)

# DrawTree(bst,F1_Score_value)

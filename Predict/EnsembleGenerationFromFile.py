import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as XGBR

from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.utils import shuffle

filename="0.75217-0.77966"
dataname="Data4Predict"
fullpath=f"./Predict/Model/{filename}"


XGB_0 = XGBR.Booster(model_file=f"{fullpath}/XGB_0.model")
XGB_1 = XGBR.Booster(model_file=f"{fullpath}/XGB_1.model")
XGB_2 = XGBR.Booster(model_file=f"{fullpath}/XGB_2.model")

RF_0 = RFR.Booster(model_file=f"{fullpath}/RF_0.model")
RF_1 = RFR.Booster(model_file=f"{fullpath}/RF_1.model")
RF_2 = RFR.Booster(model_file=f"{fullpath}/RF_2.model")

LR = RFR.Booster(model_file=f"{fullpath}/LR.model")


data = pd.read_csv(f"./Data/{dataname}.csv")
data = shuffle(data)

x = data.drop(["result"], axis=1)
y = data["result"]


smo = SMOTE(sampling_strategy=0.05)
x0, y = smo.fit_resample(x, y)

xd = XGBR.DMatrix(x0)

nx=x.shape()[0]

x1_XGB=np.zeros((nx,3))
x1_RF=np.zeros((nx,3))
x2=np.zeros((nx,2))

x1_XGB[:,0] = XGB_0.predict(x0)
x1_XGB[:,1] = XGB_1.predict(x0)
x1_XGB[:,2] = XGB_2.predict(x0)

x1_RF[:,0] = RF_0.predict(x0)
x1_RF[:,1] = RF_1.predict(x0)
x1_RF[:,2] = RF_2.predict(x0)

x2[:, 0] = x1_XGB.mean(axis=0)
x2[:, 1] = x1_RF.mean(axis=0)

rus = RandomUnderSampler(random_state=0)
x2, y = rus.fit_resample(x2, y)

ans = LR.predict(x2)

ans2 = LR.predict_proba(x2)[:,1]
ans2 = (ans2 >= 0.2) * 1

report = classification_report(y, ans)
print("效果：\n", report)
report2 = classification_report(y, ans2)
print("效果：\n", report2)

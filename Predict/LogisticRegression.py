import random
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, SMOTEN
from imblearn.under_sampling import RandomUnderSampler
from Tools.Report import MakeReport

import time

from sklearn.utils import shuffle


def TimeTransfer(x):
    timeArray = time.strptime(x, "%Y/%m/%d %H:%M")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)


# ---------------
argument = {"train_size": 0.9, "test_size": 0.1}
# ---------------

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

x = data.drop(["result"], axis=1)
y = data["result"]

# 分割数据
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)

rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_resample(x_train, y_train)

clf = LogisticRegression()
clf.fit(x_train, y_train)

ans = clf.predict(x_test)
(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report,
) = MakeReport("LR", y_test, ans, {})

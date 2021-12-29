import random
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import time

from sklearn.utils import shuffle

filename="0.72750"

train_XGB = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/oof_train_XGB.csv")
train_RF = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/oof_train_RF.csv")
# train_LR = pd.read_csv(f"./Data/EnsembleGeneration/{filename}oof_train_LR.csv")

test_XGB = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/oof_test_XGB.csv")
test_RF = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/oof_test_RF.csv")
# test_LR = pd.read_csv(f"./Data/EnsembleGeneration/{filename}oof_test_LR.csv")

y_train = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/train.csv")
y_test = pd.read_csv(f"./Data/EnsembleGeneration/{filename}/test.csv")

x_train = pd.concat([train_XGB,train_RF],axis=1)
x_test=pd.concat([test_XGB,test_RF],axis=1)
# x_train = pd.concat([train_XGB,train_RF,train_LR],axis=1)
# x_test=pd.concat([test_XGB,test_RF,test_LR],axis=1)

rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_resample(x_train, y_train)

clf = LogisticRegression()
clf.fit(x_train, y_train)

ans = clf.predict(x_test)
ans2 = clf.predict_proba(x_test)[:,1]
ans2 = (ans2 >= 0.5) * 1
report = classification_report(y_test, ans)
print("效果：\n", report)

# report2 = classification_report(y_test, ans2)
# print("效果：\n", report2)

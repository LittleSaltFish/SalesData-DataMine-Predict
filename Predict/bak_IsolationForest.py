from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.model_selection import train_test_split
from Tools.Report import MakeReport
from sklearn.metrics import classification_report

from sklearn.utils import shuffle


argument = {"train_size": 0.1, "test_size": 0.9}


data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

x = data.drop(["result"], axis=1)
y = data["result"]

# 分割数据
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=argument["train_size"], test_size=argument["test_size"]
)

# fit the mode
clf = IsolationForest()
clf.fit((x_train))

ans = clf.predict(x_test)
ans[ans == -1] = 0
# score = str(clf.score(x_test, y_test))[:5]
report = classification_report(y_test, ans)

print(report)

(
    AUC_value,
    ACC_value,
    Recall_value,
    F1_Score_value,
    Precesion_value,
    ConfusionMatrix,
    Report
) = MakeReport("OneClassSVM", y_test, ans, argument)

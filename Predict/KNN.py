import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from Tools.Report import MakeReport

from sklearn.utils import shuffle

data = pd.read_csv("./Data/Data4Predict.csv")
data = shuffle(data)

x = data.loc[
    :100,
    [
        "coupon",
        "distance_day",
        "login_diff_time",
        "course_order_num",
        "coupon_visit",
        "study_num",
        "first_order_time",
        "learn_num",
        "camp_num",
    ],
]

knn = KMeans(n_clusters=3)
knn.fit_predict(x)
print(knn.labels_)


x = pd.concat([x,knn.labels_],axis=1)
x.to_csv(
    "./knn.csv", encoding="utf-8-sig", index=False, sep=","
)

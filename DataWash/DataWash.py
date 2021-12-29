import pandas as pd
import numpy as np
import time

from sklearn.utils import shuffle


def TimeTransfer(x):
    if not pd.isna(x):
        timeArray = time.strptime(x, "%Y/%m/%d %H:%M")
        timeStamp = int(time.mktime(timeArray))
        return timeStamp
    else:
        return np.NaN


data = pd.read_csv("./Data/data_merge.csv")
data = shuffle(data)
data["city_num"].fillna("未知", inplace=True)
data.dropna(inplace=True)

# city转one-hot
city_num = pd.get_dummies(data["city_num"])
data = data.join(city_num)
data.drop(["city_num"], inplace=True, axis=1)
data.drop(["user_id"], inplace=True, axis=1)
data.drop(["app_num"], inplace=True, axis=1)
data.rename(columns={"error": "错误"}, inplace=True)


# 时间转时间戳
data["first_order_time"] = data["first_order_time"].apply(TimeTransfer)

# pd.set_option('display.max_columns', None)
# print(data.isnull().any)

data.to_csv("./Data/Data4Predict.csv", encoding="utf-8-sig", index=False, sep=",")

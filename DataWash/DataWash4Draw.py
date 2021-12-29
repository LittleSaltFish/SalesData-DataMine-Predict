import pandas as pd

from sklearn.utils import shuffle

# def timecheck(x,y):
#     if x*24<y:
#         r
#显示所有列
pd.set_option('display.max_columns', None)

#显示所有行
pd.set_option('display.max_rows', None)

data = pd.read_csv("./Data/data_merge.csv")
data = shuffle(data)
data["city_num"].fillna("未知", inplace=True)
data.dropna(inplace=True)

# # city转one-hot
# data.rename(columns={"error": "错误"}, inplace=True)

print(data.duplicated().sum())

# data.to_csv("./Data/Data4Draw.csv", encoding="utf-8-sig", index=False, sep=",")

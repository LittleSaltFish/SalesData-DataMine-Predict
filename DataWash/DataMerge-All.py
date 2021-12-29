import pandas_profiling as pp
import pandas as pd
import os

data_all = []
data_merge = pd.DataFrame()

for filename in os.listdir("./Data/RawData"):
    head, tail = filename.split(".")
    print(filename)
    data = pd.read_csv(f"./Data/RawData/{filename}")
    data_all.append(data)

for i in data_all:
    if data_merge.empty:
        data_merge = i
    else:
        data_merge = pd.merge(data_merge, i, on="user_id", how="outer")

data_merge["result"] = data_merge["result"].apply(lambda x: 1 if x == 1 else 0)

data_merge.to_csv(
    "./Data/data_merge.csv", encoding="utf-8-sig", index=False, sep=","
)

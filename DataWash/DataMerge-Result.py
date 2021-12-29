import pandas_profiling as pp
import pandas as pd
import os

data_merge = pd.DataFrame()

result = pd.read_csv(f"./Data/RawData/result.csv")

for filename in os.listdir("./Data/RawData"):
    if filename != "result.csv":
        print(filename)
        data = pd.read_csv(f"./Data/RawData/{filename}")
        data_merge = pd.merge(data, result, on="user_id", how="left")

        data_merge["result"] = data_merge["result"].apply(lambda x: 1 if x == 1 else 0)
        data_merge.to_csv(
            f"./Data/WithResult/{filename}",
            encoding="utf-8-sig",
            index=False,
            sep=",",
        )

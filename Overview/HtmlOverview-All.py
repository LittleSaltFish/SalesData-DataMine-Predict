import pandas_profiling as pp
import pandas as pd
import os

# 绘制各表预览
for filename in os.listdir("./Data/RawData"):
    head,tail=filename.split(".")
    print(filename)
    data = pd.read_csv(f"./Data/RawData/{filename}")
    report = pp.ProfileReport(data,title=filename)
    report.to_file(f"./Overview/OutPut/{head}.html")

# 绘制总表预览
data = pd.read_csv(f"./Data/data_merge.csv")
report = pp.ProfileReport(data,title="data_merge")
report.to_file(f"./Overview/OutPut/Html/data_merge.html")

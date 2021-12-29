import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import os

# 列出已绘制图表
done=os.listdir("./Overview/OutPut/Pic/Scatter")

# 循环绘制全局全相关性图
data = pd.read_csv(f"./Data/data_merge.csv")
data = data.sample(frac=0.01, replace=False, random_state=1)


index=data.columns.values
l_index=len(index)

for i in trange(l_index):
    for j in range(i,l_index):
        fig = plt.figure(dpi=500, figsize=(12, 12))
        if f"{index[i]}-{index[j]}.jpg" in done:
            print("skip")
            continue
        print(index[i], index[j])
        sns.set(style="dark", font="SimHei")
        # plt.figure(figsize=(15, 15))
        pic = sns.scatterplot(
            x=index[i],
            y=index[j],
            data=data,
            hue="result",
        )
        pic.set_title(f"{index[i]}-{index[j]}")
        fig.savefig(f"./Overview/OutPut/Pic/Scatter/{index[i]}-{index[j]}.jpg", bbox_inches="tight")
        plt.close()
        plt.clf()
        plt.cla()

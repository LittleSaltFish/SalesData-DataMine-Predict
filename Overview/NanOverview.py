import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

filename="Data4Predict"

fig = plt.figure(dpi=500, figsize=(15, 12))
sns.set(style="dark", font="SimHei")
plt.rcParams['axes.unicode_minus']=False

# 绘制缺失数值概览图
# data = pd.read_csv(f"./Data/{filename}.csv")
# colours = ["#000099", "#ffff00"]
# # specify the colours - yellow is missing. blue is not missing.
# pic=sns.heatmap(data.isnull(), cmap=sns.color_palette(colours))
# fig.savefig(f"./Overview/OutPut/Pic/NanOverview-{filename}.jpg", bbox_inches='tight')

for filename in os.listdir("./Data/RawData"):
    head, tail = filename.split(".")
    print("start plot:",head)
    data = pd.read_csv(f"./Data/RawData/{head}.csv")
    colours = ["#000099", "#ffff00"]
    # specify the colours - yellow is missing. blue is not missing.
    pic=sns.heatmap(data.isnull(), cmap=sns.color_palette(colours))
    fig.savefig(f"./Overview/OutPut/Pic/NanOverview-{head}.jpg", bbox_inches='tight')
    plt.clf()
    plt.cla()




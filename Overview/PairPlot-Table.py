import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 绘制表级相关性图
plt.gcf().set_size_inches(1000,1000)

for filename in os.listdir("./Data/WithResult"):
    head, tail = filename.split(".")
    print(filename)
    data = pd.read_csv(f"./Data/WithResult/{filename}")
    sns.pairplot(
        data,
        hue="result",
        # kind="reg",
        diag_kind="kde",
    )
    plt.savefig(f"./Overview/OutPut/Pic/PairPlot-{head}.jpg", bbox_inches="tight")
    plt.cla()

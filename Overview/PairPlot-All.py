import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制全局全相关性图
plt.figure(dpi=100,figsize=(1000,1000))
plt.gcf().set_size_inches(1000,1000)
data = pd.read_csv(f"./Data/data_merge.csv")
sns.pairplot(
    data,
    hue="result",
    # kind="reg",
    diag_kind="kde",
)
plt.savefig("./Overview/OutPut/Pic/PairPlot-All.jpg", bbox_inches="tight")

import pandas as pd
from pandasgui import show
data = pd.read_csv(f"./Data/data_merge.csv")
show(data)

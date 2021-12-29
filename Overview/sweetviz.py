import pandas as pd
import sweetviz as sv

my_dataframe = pd.read_csv(f"./Data/data_merge.csv")

my_report = sv.analyze(my_dataframe)
my_report.show_html()

import pandas_profiling as pp
import pandas as pd

filename="result.csv"
head,tail=filename.split(".")

# data_sample=data.sample(frac=0.01, replace=False, random_state=1)

data = pd.read_csv(f"./Data/RawData/{filename}")
report = pp.ProfileReport(data,title=head)
report.to_file(f"./Overview/OutPut/Html/{head}.html")

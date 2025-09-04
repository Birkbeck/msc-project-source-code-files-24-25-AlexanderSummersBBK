import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path.cwd()
data = pd.DataFrame(pd.read_csv(path / "main_table.csv"))

dates = data["Date"]
dates_id = {}
for i in range(len(dates)-1):
    if dates[i] not in dates_id:
        dates_id[dates[i]] = data[data["Date"] == dates[i]]["ID"].tolist()
print(dates_id)
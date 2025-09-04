import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Defines the path to the data
path = Path.cwd()
data = pd.DataFrame(pd.read_csv(path / "main_table.csv"))

# Creates a library of dates with associated IDs, that are stored in an array
dates = data["Date"]
dates_id = {}
for i in range(len(dates)-1):
    if dates[i] not in dates_id:
        dates_id[dates[i]] = data[data["Date"] == dates[i]]["ID"].tolist()

# Creates a calendar within a library
start, end = datetime.strptime("01/01/2016", "%d/%m/%Y"), datetime.strptime("28/02/2025", "%d/%m/%Y")
calendar = {}
while start <= end:
    calendar[start.strftime("%d/%m/%Y")] = []
    start += timedelta(days=1)
print(calendar)


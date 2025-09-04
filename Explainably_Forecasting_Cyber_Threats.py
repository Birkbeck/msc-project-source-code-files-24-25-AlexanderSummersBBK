import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import numpy as np


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
    calendar[start.strftime("%d/%m/%Y")] = [[]]
    start += timedelta(days=1)

# Combines dates_id and calendar by cycling through the dates in calendar.
# Once a date in calendar has a coressponding array of IDs, each of the IDs 
# will be implemented into the arrays within the calendar.
data_without_dates = data.drop(columns=["ID", "Date"])
for each in calendar:
    if each in dates_id:
        for thy in dates_id[each]:
            calendar[each][0].append(data_without_dates[data["ID"] == thy])

# Adding a number of attacks to each date
for each in calendar:
    calendar[each].append(len(calendar[each][0]))


print(calendar["01/01/2016"])


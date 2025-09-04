import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb

# Defines the path to the data
path = Path.cwd()
data = pd.DataFrame(pd.read_csv(path / "main_table.csv"))

# Creates a library of dates with associated IDs, that are stored in an array
dates = data["Date"]
dates_id = {}
for i in range(len(dates)):
    if dates[i] not in dates_id:
        dates_id[dates[i]] = data[data["Date"] == dates[i]]["ID"].tolist()

# Creates a calendar within a library
start, end = datetime.strptime("01/01/2016", "%d/%m/%Y"), datetime.strptime("28/02/2025", "%d/%m/%Y")
calendar = {}
while start <= end:
    calendar[start.strftime("%d/%m/%Y")] = []
    start += timedelta(days=1)

# Combines dates_id and calendar by cycling through the dates in calendar.
# Once a date in calendar has a coressponding array of IDs, each of the IDs 
# will be implemented into the arrays within the calendar.
data_without_dates = data.drop(columns=["Date"])
for each in calendar:
    if each in dates_id:
        for thy in dates_id[each]:
            calendar[each].append(data_without_dates[data["ID"] == thy])

# Adding a number of attacks to each date
n_of_attacks = []
for each in calendar:
    n_of_attacks.append(len(calendar[each]))

# Using back testing and xgboost
def backtest(X, y, model, splits = 3):
    ts = TimeSeriesSplit(n_splits = splits)
    values = []
    for train, test in ts.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train , y_test = y.iloc[train], y.iloc[test]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        values.append([y_test, predictions])
    return values
print(len(calendar), len(n_of_attacks))
X = calendar
y = pd.DataFrame(n_of_attacks)
print(y)
model = xgb.XGBRegressor()
values = backtest(X, y, model)
print(values[0])

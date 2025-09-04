import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
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
    if len(calendar[each]) > 0:
        for i in range(len(calendar[each])):
            n_of_attacks.append(len(calendar[each]))
    else:
        n_of_attacks.append(len(calendar[each]))

# Using back testing and xgboost
def continuous_backtest(X, y, training_n = 8400, testing_n = 20, max_folds = 150):
    values = []
    for each in X.select_dtypes(include = 'object').columns:
        X[each] = X[each].astype('category')
    length_of_data = len(data)
    fold_n = 0
    train_final = training_n
    while (train_final + testing_n) <= length_of_data and fold_n < max_folds:
        test_initial = train_final
        test_final = test_initial + testing_n
        X_train = X.iloc[: train_final]
        X_test = X.iloc[test_initial : test_final]
        y_train = y.iloc[: train_final]
        y_test = y.iloc[test_initial : test_final]
        D_train = xgb.DMatrix(X_train, label = y_train, enable_categorical = True)
        D_test = xgb.DMatrix(X_test, label = y_test, enable_categorical = True)
        parameters = {'objective': 'reg:absoluteerror',
                      'enable_categorical': True,
                      'tree_method': 'hist',
                      'max_depth': 5,
                      'learning_rate': 0.05,
                      'eval_metric': 'mae'}
        model = xgb.train(parameters, D_train, num_boost_round = 200, early_stopping_rounds = 25)
        predictions = model.predict(D_test)
        mae = mean_absolute_error(y_test, predictions)
        values.append({'mae': mae,
                       'Predictions': predictions,
                       'y_test': y_test.values})
        train_final += testing_n
        fold_n += 1
    return values


X = data.drop(columns = ["ID"])
y = pd.DataFrame(n_of_attacks)

values = continuous_backtest(X, y)
print(values)

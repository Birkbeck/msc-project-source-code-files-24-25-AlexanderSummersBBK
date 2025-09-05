import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import numpy as np
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
import shap
import numpy as np
import matplotlib.pyplot as plt

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

# Using back testing and lightGBM with SHAP
def model(a, b):
    X = a.drop(columns = ["ID"])
    features = [column for column in X.columns]
    for each in features:
        X[each] = X[each].astype('category')
    y = pd.DataFrame(b)
    tss = TimeSeriesSplit(n_splits=4)
    modelling, predictions, maes, shap_values, Xs = [], [], [], [], []
    for initial in range(len(a) - int(len(a)*0.7) - int(len(a)*0.2) + 1):
        i, j = list(range(initial, initial + int(len(a)*0.7))), list(range(initial + int(len(a)*0.7), initial + int(len(a)*0.7) + int(len(a)*0.2)))
        X_train = X.iloc[i]
        X_test = X.iloc[j]
        y_train = y.iloc[i]
        y_test = y.iloc[j]
        y_train, y_test = y_train.squeeze(), y_test.squeeze()
        model = LGBMRegressor(objective = 'regression', n_estimators = 1000)
        model.fit(X_train, y_train, eval_set = [(X_test, y_test)], callbacks = [early_stopping(25), log_evaluation(10)])
        y_prediction = model.predict(X_test)
        predictions.append(y_prediction)
        mae = mean_absolute_error(y_test, y_prediction)
        maes.append(mae)
        modelling.append(model)
        explainer = shap.TreeExplainer(model)
        shap_values.append(explainer.shap_values(X_test))
        Xs.append(X_test)


    '''main_shap = np.vstack(shap_values)
    main_X = pd.concat(Xs, axis = 0)
    shap.summary_plot(main_shap, features = main_X, plot_type = "dot", max_display = 10)
    plt.show()'''
    return modelling
main_model = model(data, n_of_attacks)

# Predicting next 3 Years of Cyber Attacks
n_days = 365*3
X_for = data.drop(columns = ["ID"]).sample(n = n_days, replace = True).reset_index(drop = True)
days = pd.date_range(start = datetime.strptime("01/03/2025", "%d/%m/%Y"), periods = n_days, freq = "D")
X_for.index = days
features_for = [column for column in X_for.columns]
for each in features_for:
    X_for[each] = X_for[each].astype('category')


# Cycle through each model, then access on real data between 01-01-2025 and 28-02-2025 to choose best model to then make forecast with.
forecast = main_model[-1].predict(X_for)

# Create the associated SHAP value graph.

# Final Projection
def plot(dates = days, optimal = forecast):
    plt.plot(dates, optimal)
    plt.xlabel("Date")
    plt.ylabel("Predicted Number of Attacks Overall")
    plt.show()
plot()
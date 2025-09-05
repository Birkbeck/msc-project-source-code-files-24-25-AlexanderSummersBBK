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
X = data.drop(columns = ["ID"])
features = [column for column in X.columns]
for each in features:
    X[each] = X[each].astype('category')
y = pd.DataFrame(n_of_attacks)
tss = TimeSeriesSplit(n_splits=4)
modelling, predictions, maes, shap_values, Xs = [], [], [], [], []
for initial in range(len(data) - int(len(data)*0.7) - int(len(data)*0.2) + 1):
    i, j = list(range(initial, initial + int(len(data)*0.7))), list(range(initial + int(len(data)*0.7), initial + int(len(data)*0.7) + int(len(data)*0.2)))
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


main_shap = np.vstack(shap_values)
main_X = pd.concat(Xs, axis = 0)
shap.summary_plot(main_shap, features = main_X, plot_type = "dot", max_display = 10)
plt.show()

'''forecast_dates = pd.date_range(start = end + timedelta(days = 1), periods = 365*3)
forecast = pd.DataFrame(index = forecast_dates, columns = X.columns)
X_forecast = forecast[X.columns]
for each in X.select_dtypes(include = 'object').columns:
    X[each] = X[each].astype('category')
D_train_main = xgb.DMatrix(X, label = y.loc[:12114], enable_categorical = True)
parameters = {'objective': 'reg:absoluteerror',
              'enable_categorical': True,
                'tree_method': 'hist',
                'max_depth': 5,
                'learning_rate': 0.05,
                'eval_metric': 'mae'}

for each in X_forecast.select_dtypes(include = 'object').columns:
    X_forecast[each] = X_forecast[each].astype('category')
shap_values = explainer.shap_values(X_forecast)'''



# Final Projection
'''def plot(dates = forecast_dates, optimal = optimal_forecast):
    plt.plot(forecast_dates, optimal_forecast)
    plt.xlabel("Date")
    plt.ylabel("Predicted Number of Attacks Overall")
    plt.show()'''
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics

import json
from pprint import pprint
import mlflow
import mlflow.sklearn

base_path = '/media/armando/Neptuno/git/mlflow_demo/'
income = pd.read_csv(base_path + '02_sklearn_demo/California_Houses.csv')

columns = ['Median_House_Value', 'Median_Income', 'Median_Age', 'Tot_Rooms',
           'Tot_Bedrooms', 'Population']
income = income[columns]

print(income.head(3))
print(income.columns)

columns = income.columns.tolist()
columns.remove('Median_House_Value')

X = income[columns]
y = income['Median_House_Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('\n\n')

with mlflow.start_run() as run:
    mlflow.set_tag("model_sklearn", "1.0.0")

    params = {'random_state': 100, 'n_estimators': 150, 'max_depth':4}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(run.info.run_id)

    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)


    #LOG PARAMS
    mlflow.log_params(params)

    #LOG METRICS
    mlflow.log_metric("metric_mae", mae)
    mlflow.log_metric("metric_mse", mse)

    #LOG ARTIFACTS
    mlflow.log_artifact("California_Houses.csv")

    #LOG MODEL
    mlflow.sklearn.log_model(
        artifact_path="sklearn_model",
        sk_model=model)
     


print(model.feature_importances_)
print('Data modeled successfully') 



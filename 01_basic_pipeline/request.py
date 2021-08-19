import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
#from mlflow.models.signature import infer_signature

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(URL, header=None, names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach', 'exang','oldpeak','slope','ca','thal','num'])
df['target']=np.where(df['num'] > 0,1,0)
print(df.head()) 


train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


import requests
host = 'localhost'
port = '5000'

url = f'http://{host}:{port}/invocations'

headers = {'Content-Type': 'application/json',}
# test contains our data from the original train/valid/test split

http_data = test.to_json(orient='split')
r = requests.post(url=url, headers=headers, data=http_data)

print(f'Predictions: {r.text}') 

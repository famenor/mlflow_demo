from fuzzywuzzy import fuzz

import pandas as pd
import numpy as np

import mlflow.pyfunc

model_path = 'model_word_model'

str_1 = ['alibaba', 'guitarra', 'zopilote', 'xxxxxxx']
str_2 = ['alibab', 'guerra', 'clandestuno', 'aaaa']
input = pd.DataFrame({'str_1': str_1, 'str_2': str_2}) 
print(input)

loaded_model = mlflow.pyfunc.load_model(model_path)
output = loaded_model.predict(input)
print(output)



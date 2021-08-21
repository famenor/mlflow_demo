from fuzzywuzzy import fuzz

import pandas as pd
import numpy as np

import mlflow.pyfunc

class ModelWord(mlflow.pyfunc.PythonModel):

	def __init__(self, int_1):
		self.int_1 = int_1

	def predict(self, context, model_input):
		results = []
		for i, row in model_input.iterrows():
			match = fuzz.ratio(row['str_1'], row['str_2'])
			output = match + 1000 * self.int_1
			results.append(output)
			print('Match between', row['str_1'], row['str_2'], match)

		return np.array(results)

model_path = 'model_word_model'


with mlflow.start_run(run_name='exec_01') as run:
	params = {'multiplyer': 7}
	word_model = ModelWord(params['multiplyer'])

	mlflow.log_params(params)
	mlflow.pyfunc.log_model(python_model=word_model,
						    artifact_path='model')

print('Logged Successfully')
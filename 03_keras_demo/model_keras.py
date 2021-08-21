import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys

import mlflow

if len(sys.argv) <= 1:
	print('Param error')
	exit()

#mlflow run . -P param_epochs=3 -P param_drop=0.25
param_epochs = int(sys.argv[1]) #if len(sys.argv) > 1 else 0.5
param_drop = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

print('Parametros:')
print('Epochs:', param_epochs)
print('Drop:', param_drop)


EPOCHS=param_epochs
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10
N_HIDDEN=128
VALIDATION_SPLIT=0.2
DROPOUT=param_drop

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED) 
X_test = X_test.reshape(10000, RESHAPED) 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name='dense_layer', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN, name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES, name='dense_layer_3', activation='softmax'))
model.summary()

#model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

with mlflow.start_run() as run:
    mlflow.set_tag("model_keras", "1.0.0")

    #params = {'EPOCHS': param_epochs, 'DROP': param_drop}

    #LOG PARAMS
    #mlflow.log_params(params)

    #LOG METRICS
    mlflow.log_metric("metric_acc", test_acc)

    #LOG MODEL
    #mlflow.sklearn.log_model(
    #    artifact_path="sklearn_model",
    #    sk_model=model)
     
print('Success') 

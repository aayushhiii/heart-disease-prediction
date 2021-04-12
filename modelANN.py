# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:28:32 2020

@author: Aashi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import tensorflow as tf
import keras


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

from keras.models import model_from_json
import os
from sklearn.model_selection import train_test_split
from keras import models, layers
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import BatchNormalization

dataset = pd.read_csv(r'C:\\Users\\Aashi\\ADHF-Barkhabangur2.csv')

new=dataset.drop(['S.0','Diagnosis'], axis=1)


X = new.iloc[:, :6]


y = new.iloc[:, -1]

#Splitting Training and Test Set

y = new['STATUS']
X = new.drop(['STATUS'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 23)
#Since we have a very small dataset, we will train our model with all availabe data.
#Since we have a very small dataset, we will train our model with all availabe data.
#len(X[0])
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state= 10)
len(X_train)
len(X_test)
X_train
#Initialising ANN
modelANN = Sequential()
modelANN.add(Dense(32, input_shape=(6,))),
modelANN.add(Dense(20,activation="relu")),
modelANN.add(Dense(2,activation='softmax'))


modelANN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])



history = modelANN.fit(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))


prediction_features=modelANN.predict(X_test)
performance=modelANN.evaluate(X_test,y_test)
print(performance)


history_dict = history.history
history_dict.keys()

# Checking Overfit
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Save the model
# serialize model to JSON
modelANN_json = modelANN.to_json()
with open("modelANN.json", "w") as json_file:
    json_file.write(modelANN_json)
# serialize weights to HDF5
modelANN.save_weights("modelANN.h5")
print("Saved model to disk")

# load json and create model
json_file = open('modelANN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelANN.h5")
print("Loaded model from disk")


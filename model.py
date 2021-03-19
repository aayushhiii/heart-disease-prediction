# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:28:35 2020

@author: Aashi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split

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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print("Accuracy of Model",regressor.score(X_test,y_test)*100,"%")

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:28:35 2020

@author: Aashi
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

dataset = pd.read_csv(r'F:\\ADHF-Barkhabangur2.csv')
new=dataset.drop(['S.0','Diagnosis'], axis=1)
X = new.iloc[:, :6]


y = new.iloc[:, -1]
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
y = new['STATUS']
X = new.drop(['STATUS'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.externals import joblib
joblib.dump(clf, 'modelNB.pkl')

modelNB = open('modelNB.pkl','rb')
clf = joblib.load(modelNB)

print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

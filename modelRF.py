# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:40:56 2020

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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

classifierRF=RandomForestClassifier()
classifierRF.fit(X_train,y_train)
classifierRF.score(X_test, y_test)
print('RandomForest LogLoss {score}'.format(score=log_loss(y_test, classifierRF.predict_proba(X_test))))
#clfs.append(classifierRF)
# save best model to current working directory
joblib.dump(classifierRF, 'modelRF.pkl')
# load from file and predict using the best configs found in the CV step
model_classifierRF = joblib.load('modelRF.pkl' )
# get predictions from best model above
y_preds = model_classifierRF.predict(X_test)
print('Random Forest accuracy score:',accuracy_score(y_test, y_preds))
print('\n')
import pylab as plt
labels=[0,1]
cmx=confusion_matrix(y_test,y_preds, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')
print(classification_report(y_test, y_preds))

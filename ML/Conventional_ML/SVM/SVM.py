#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import metrics

### Choose dataset ###
dataset = 'DFT'

data = pd.read_csv(f'data/data_{dataset}.csv')
y = pd.DataFrame(data['Yield'],columns=['Yield'])
print(y)
X = data.drop(columns=['Name', 'ID', 'Yield'])
print(X)


r2_train = []
r2_test = []
mae_train = []
mae_test = []
for i in range(0,100):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
    a_X_train = (X_train - X_train.mean()) / X_train.std()
    a_X_test = (X_test - X_train.mean()) / X_train.std()
    param = {'C':[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
             'gamma':[1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 50, 100, 500, 1000],
             'epsilon':[1e-05, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]}
    reg = GridSearchCV(SVR(kernel='rbf'), param_grid=param, cv=10, n_jobs=16)
    reg.fit(a_X_train,y_train['Yield'])
    best = reg.best_estimator_
    print(reg.best_estimator_)
    y_pred1 = best.predict(a_X_train)
    y_pred2 = best.predict(a_X_test)
    r2_train.append(metrics.r2_score(y_train, y_pred1))
    r2_test.append(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(y_train, y_pred1))
    mae_test.append(metrics.mean_absolute_error(y_test, y_pred2))
    

r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test], axis=1, join='inner')
result.to_csv(f'result/result_{dataset}.csv')

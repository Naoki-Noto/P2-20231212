#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

### Choose dataset ###
dataset = 'DFT'

### Choose test size ###
size = 0.6

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=i)
    param = {"n_estimators": [100, 1000, 5000], "max_depth": [3, 4, 5, 6]}
    reg = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param, cv=10, n_jobs=16)
    reg.fit(X_train,y_train['Yield'])
    best = reg.best_estimator_
    print(reg.best_estimator_)
    y_pred1 = best.predict(X_train)
    y_pred2 = best.predict(X_test)
    r2_train.append(metrics.r2_score(y_train, y_pred1))
    r2_test.append(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(y_train, y_pred1))
    mae_test.append(metrics.mean_absolute_error(y_test, y_pred2))
    

r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test], axis=1, join='inner')
result.to_csv(f'result/result_test{size}_{dataset}.csv')


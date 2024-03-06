#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv('data/data_DFT.csv')
y = pd.DataFrame(data['Yield'],columns=['Yield'])
X = data.drop(columns=['Name', 'ID', 'Yield'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=75)
print(X_train)
param = {"n_estimators": [1000, 3000, 5000], "max_depth": [3, 4, 5, 6]}
reg = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param, cv=10, n_jobs=16)
reg.fit(X_train,y_train['Yield'])
best = reg.best_estimator_
print(reg.best_estimator_)
y_pred = best.predict(X_test)

print('r^2 for test data:', metrics.r2_score(y_test, y_pred))
print('MAE for test data:', metrics.mean_absolute_error(y_test, y_pred))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(0,100),range(0,100), c = "black")
plt.scatter(y_test, y_pred)
plt.xlim(0,100)
plt.ylim(0,100)
ax.set_aspect('equal', adjustable='box')
plt.xlabel("experiment")
plt.ylabel("prediction")
plt.show()

y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
y_test = np.array(y_test)
y_test = pd.DataFrame(y_test, columns=['y_exp'])
y = pd.concat([y_pred, y_test], axis=1, join='inner')
print(y)
y.to_csv('result/RF.csv', index=False)



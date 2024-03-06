#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:17:38 2021

@author: notonaoki
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn import metrics

### Choose dataset ###
dataset = 'DFT'

data = pd.read_csv('data/data_{}.csv'.format(dataset))
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
    model = LassoCV(alphas=np.linspace(0, 10, num=101), cv=10, max_iter=1000, n_jobs=16)
    model.fit(a_X_train, y_train)
    print("Best parameters:", model.alpha_)
    y_pred1 = model.predict(a_X_train)
    y_pred2 = model.predict(a_X_test)
    r2_train.append(metrics.r2_score(y_train, y_pred1))
    r2_test.append(metrics.r2_score(y_test, y_pred2))
    mae_train.append(metrics.mean_absolute_error(y_train, y_pred1))
    mae_test.append(metrics.mean_absolute_error(y_test, y_pred2))
    

r2_train = pd.DataFrame(data=r2_train, columns=['r2_train'])
r2_test = pd.DataFrame(data=r2_test, columns=['r2_test'])
mae_train = pd.DataFrame(data=mae_train, columns=['mae_train'])
mae_test = pd.DataFrame(data=mae_test, columns=['mae_test'])
result = pd.concat([r2_train, r2_test, mae_train, mae_test], axis=1, join='inner')
result.to_csv('result/result_{}.csv'.format(dataset))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:51:13 2021

@author: notonaoki
"""


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import shap


data_train = pd.read_csv('data/data_train.csv')
data_train = data_train.drop(columns=['Name', 'ID'])
data_test = pd.read_csv('data/data_test.csv')
ID_test = data_test[['Name', 'ID']]
X_test = data_test.drop(columns=['Name', 'ID'])

y_train = pd.DataFrame(data_train['Yield'],columns=['Yield'])
X_train = data_train.drop(columns=['Yield'])
print(X_train)

loo = LeaveOneOut()
param = {"n_estimators": [1000, 3000, 5000], "max_depth": [2, 3, 4]}
reg = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param,
                   cv=loo, scoring='neg_mean_absolute_error', n_jobs=16)
reg.fit(X_train,y_train['Yield'])
best = reg.best_estimator_
print(reg.best_estimator_)
y_pred = best.predict(X_test)

y_pred = pd.DataFrame(y_pred, columns=['Pred_yield'])
prediction = pd.concat([ID_test, y_pred], axis=1, join='inner')
prediction = prediction.sort_values('Pred_yield', ascending=False)
print(prediction)
prediction.to_csv('result/prediction_iso_RF.csv')

importance = pd.DataFrame(best.feature_importances_, index=X_train.columns, columns=['importance'])
importance = importance.sort_values('importance', ascending=True)
print(importance)
importance.to_csv('result/importance_iso_RF.csv')

fig = plt.figure()
explainer = shap.TreeExplainer(model=best)
shap_values = explainer.shap_values(X=X_train)
shap.summary_plot(shap_values, X_train)
plt.show()
fig.savefig('result/shap.pdf')



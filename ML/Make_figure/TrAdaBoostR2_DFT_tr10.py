# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import HistGradientBoostingRegressor as hgb
from sklearn import metrics
import matplotlib.pyplot as plt


data_s = pd.read_csv('data/data_s_S1_tr10.csv')
data_s = data_s.drop(columns=['Name', 'ID'])
data_t = pd.read_csv('data/data_t_tr10.csv')
data_t = data_t.drop(columns=['Name', 'ID'])

ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
Xs = data_s.drop(columns=['Yield'])
yt = pd.DataFrame(data_t['Yield'],columns=['Yield'])
Xt = data_t.drop(columns=['Yield'])

Xt_lab, Xt_unlab, yt_lab, yt_unlab = train_test_split(Xt, yt, test_size=0.9, random_state=89)
print(Xt_lab)
model = TrAdaBoostR2(hgb(), n_estimators=31, Xt=Xt_lab, yt=yt_lab, random_state=0)
model.fit(Xs, ys)
y_pred = model.predict(Xt_unlab)

print('r^2 for test data:', metrics.r2_score(yt_unlab, y_pred))
print('MAE for test data:', metrics.mean_absolute_error(yt_unlab, y_pred))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(0,100),range(0,100), c = "black")
plt.scatter(yt_unlab, y_pred)
plt.xlim(0,100)
plt.ylim(0,100)
ax.set_aspect('equal', adjustable='box')
plt.xlabel("experiment")
plt.ylabel("prediction")
plt.show()

y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
yt_unlab = np.array(yt_unlab)
yt_unlab = pd.DataFrame(yt_unlab, columns=['y_exp'])
y = pd.concat([y_pred, yt_unlab], axis=1, join='inner')
print(y)
y.to_csv('result/TrAB_tr10.csv', index=False)

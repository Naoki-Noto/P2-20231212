# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from adapt.feature_based import FA
from sklearn import metrics
from lightgbm import LGBMRegressor as lgbm


es, dataset, dataset2 = lgbm, 'All', 't'

data_s = pd.read_csv(f'data/data_s_{dataset}.csv')
data_s = data_s.drop(columns=['Name', 'ID'])
data_t = pd.read_csv(f'data/data_{dataset2}.csv')
data_t = data_t.drop(columns=['Name', 'ID'])
print(data_s)
print(data_t)

ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
Xs = data_s.drop(columns=['Yield'])
print(Xs)
yt = pd.DataFrame(data_t['Yield'],columns=['Yield'])
Xt = data_t.drop(columns=['Yield'])
print(Xt)


r2_all = []
r2_lab = []
r2_unlab = []
mae_all = []
mae_lab = []
mae_unlab = []
unsupervised = []
for i in range(0,100):
    Xt_lab, Xt_unlab, yt_lab, yt_unlab = train_test_split(Xt, yt, test_size=0.5, random_state=i)
    #print(Xt_lab)
    model = FA(es(), Xt=Xt_lab, yt=yt_lab, random_state=0, verbose=0)
    model.fit(Xs, ys)
    y_pred = model.predict(Xt)
    y_pred1 = model.predict(Xt_lab)
    y_pred2 = model.predict(Xt_unlab)
    r2_all.append(metrics.r2_score(yt, y_pred))
    r2_lab.append(metrics.r2_score(yt_lab, y_pred1))
    r2_unlab.append(metrics.r2_score(yt_unlab, y_pred2))
    mae_all.append(metrics.mean_absolute_error(yt, y_pred))
    mae_lab.append(metrics.mean_absolute_error(yt_lab, y_pred1))
    mae_unlab.append(metrics.mean_absolute_error(yt_unlab, y_pred2))
    unsupervised.append(model.unsupervised_score(Xs, Xt))

r2_all = pd.DataFrame(data=r2_all, columns=['r2_all'])
r2_lab = pd.DataFrame(data=r2_lab, columns=['r2_lab'])
r2_unlab = pd.DataFrame(data=r2_unlab, columns=['r2_unlab'])
mae_all = pd.DataFrame(data=mae_all, columns=['mae_all'])
mae_lab = pd.DataFrame(data=mae_lab, columns=['mae_lab'])
mae_unlab = pd.DataFrame(data=mae_unlab, columns=['mae_unlab'])
result = pd.concat([r2_all, mae_all, r2_lab, mae_lab, r2_unlab, mae_unlab], axis=1, join='inner')
result.to_csv('result/result_FA.csv')

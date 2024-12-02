# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from adapt.instance_based import TrAdaBoostR2
from lightgbm import LGBMRegressor as lgbm
from sklearn.ensemble import HistGradientBoostingRegressor as hgb
from sklearn import metrics

### Choose test size & enter n_estimators ###
#size, ne, es = 0.5, 5, lgbm
size, ne, es = 0.5, 7, lgbm
#size, ne, es = 0.6, 7, lgbm
#size, ne, es = 0.7, 8, lgbm
#size, ne, es = 0.8, 8, lgbm
#size, ne, es = 0.9, 12, hgb

### Choose source & target domains ###
#dataset, dataset2 = 'S1', 't'
#dataset, dataset2 = 'S2', 't'
#dataset, dataset2 = 'S3', 't'
#dataset, dataset2 = 'S4', 't'
#dataset, dataset2 = 'S5', 't'
#dataset, dataset2 = 'All', 't'
dataset, dataset2 = 'S1FE', 'tFE'


data_s = pd.read_csv(f'data/data_s_{dataset}.csv')
data_s = data_s.drop(columns=['Name', 'ID'])
data_t = pd.read_csv(f'data/data_{dataset2}.csv')
data_t = data_t.drop(columns=['Name', 'ID'])
print(data_s)
print(data_t)

ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
Xs = data_s.drop(columns=['Yield',
                          #'HOMO', 'E_S1', 'E_T1', 'dEST', 'dDM,
                          #'P(E_S1) * P(E_T1)', 'P(dDM) * P(E_S1)', 'P(dDM) * P(HOMO)',
                          #'P(E_S1) + P(HOMO)', 'P(dEST) - P(E_S1)', 'P(dEST) / P(HOMO)',
                          ])
print(Xs)
yt = pd.DataFrame(data_t['Yield'],columns=['Yield'])
Xt = data_t.drop(columns=['Yield',
                          #'HOMO', 'E_S1', 'E_T1', 'dEST', 'dDM,
                          #'P(E_S1) * P(E_T1)', 'P(dDM) * P(E_S1)', 'P(dDM) * P(HOMO)',
                          #'P(E_S1) + P(HOMO)', 'P(dEST) - P(E_S1)', 'P(dEST) / P(HOMO)',
                          'Reaction_CO_1.5h',
                          'Reaction_CO_7.5h',
                          'Reaction_CO_biphenyl',
                          'Reaction_CO_ortho',
                          #'Reaction_CO_Cl',
                          #'Reaction_CS',
                          #'Reaction_CN',
                          ])
print(Xt)


r2_all = []
r2_lab = []
r2_unlab = []
mae_all = []
mae_lab = []
mae_unlab = []
unsupervised = []
for i in range(0,100):
    print(i)
    Xt_lab, Xt_unlab, yt_lab, yt_unlab = train_test_split(Xt, yt, test_size=size, random_state=i)
    #print(Xt_lab)
    model = TrAdaBoostR2(es(), n_estimators=ne, Xt=Xt_lab, yt=yt_lab, random_state=0)
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
unsupervised = pd.DataFrame(data=unsupervised, columns=['unsupervised'])
result = pd.concat([r2_all, mae_all, r2_lab, mae_lab, r2_unlab, mae_unlab, unsupervised], axis=1, join='inner')
result.to_csv(f'result/result_unlab{size}_{dataset}&{dataset2}_ne{ne}.csv')

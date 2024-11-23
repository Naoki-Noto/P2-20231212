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
#size, ne, es = 0.5, 9, lgbm
#size, ne, es = 0.6, 10, lgbm
#size, ne, es = 0.7, 14, lgbm
#size, ne, es = 0.8, 17, lgbm
size, ne, es = 0.9, 24, lgbm

### Choose source & target domains ###
#dataset, dataset2 = 'S1', 't'
#dataset, dataset2 = 'S1FE', 'tFE'
dataset, dataset2 = 'S1FE2', 'tFE2'

def TrAB(size, ne, es, dataset, dataset2):
    data_s = pd.read_csv(f'data/data_s_{dataset}.csv')
    data_s = data_s.drop(columns=['Name', 'ID'])
    data_t = pd.read_csv(f'data/data_{dataset2}.csv')
    data_t = data_t.drop(columns=['Name', 'ID'])

    ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
    Xs = data_s.drop(columns=['Yield'])
    yt = pd.DataFrame(data_t['Yield'],columns=['Yield'])
    Xt = data_t.drop(columns=['Yield'])

    r2_unlab = []
    for i in range(0,100):
        Xt_lab, Xt_unlab, yt_lab, yt_unlab = train_test_split(Xt, yt, test_size=size, random_state=i)
        model = TrAdaBoostR2(es(), n_estimators=ne, Xt=Xt_lab, yt=yt_lab, random_state=0, verbose=0)
        model.fit(Xs, ys)
        y_pred = model.predict(Xt_unlab)
        r2_unlab.append(metrics.r2_score(yt_unlab, y_pred))
    
    result = pd.DataFrame(data=r2_unlab, columns=['r2_unlab'])
    result.to_csv(f'result/result_unlab{size}_{dataset}&{dataset2}_ne{ne}.csv')
    return r2_unlab
    
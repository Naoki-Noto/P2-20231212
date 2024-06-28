# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd
from adapt.instance_based import TrAdaBoostR2
from sklearn.ensemble import HistGradientBoostingRegressor as hgb

### Choose source domain ###
source = 'S1'
#source = 'S6'

data_s = pd.read_csv(f'data/data_s_{source}.csv')
data_s = data_s.drop(columns=['Name', 'ID'])

data_t_lab = pd.read_csv('data/data_t_lab_iso.csv')
data_t_lab = data_t_lab.drop(columns=['Name', 'ID'])
#print(data_t_lab)
data_t_unlab = pd.read_csv('data/data_t_unlab_iso.csv')
ID_t_unlab = data_t_unlab[['Name', 'ID']]
data_t_unlab = data_t_unlab.drop(columns=['Name', 'ID'])
#print(data_t_unlab)

ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
Xs = data_s.drop(columns=['Yield'])
#print(Xs)
yt_lab = pd.DataFrame(data_t_lab['Yield'],columns=['Yield'])
Xt_lab = data_t_lab.drop(columns=['Yield',
                                  #'Reaction_CO_Cl',
                                  #'Reaction_CS',
                                  #'Reaction_CN',
                                  'Reaction_2+2',
                                  ])
Xt_unlab = data_t_unlab.drop(columns=[
                                      #'Reaction_CO_Cl',
                                      #'Reaction_CS',
                                      #'Reaction_CN',
                                      'Reaction_2+2',
                                      ])


model = TrAdaBoostR2(hgb(), n_estimators=23, Xt=Xt_lab, yt=yt_lab, random_state=0, verbose=0)
model.fit(Xs, ys)
y_pred = model.predict(Xt_unlab)
y_pred = pd.DataFrame(y_pred, columns=['Pred_yield'])
prediction = pd.concat([ID_t_unlab, y_pred], axis=1, join='inner')
prediction = prediction.sort_values('Pred_yield', ascending=False)
print(prediction.head())
prediction.to_csv(f'result/prediction_iso_{source}.csv')

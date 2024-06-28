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

data_s = pd.read_csv('data/data_s_S1FE.csv')
data_s = data_s.drop(columns=['Name', 'ID'])
data_t = pd.read_csv('data/data_tFE.csv')
data_t_lab, data_t_unlab = train_test_split(data_t, test_size=0.9, random_state=89)
ID_unlab = data_t_unlab[['Name', 'ID']]
data_t = data_t.drop(columns=['Name', 'ID'])
#print(data_t_lab)

ys = pd.DataFrame(data_s['Yield'],columns=['Yield'])
Xs = data_s.drop(columns=['Yield',
                          #'HOMO', 'E_S1', 'f_S1', 'E_T1', 'dDM', 'P(dDM) * P(E_S1)', 'P(dDM) * P(dEST)', 'P(dEST) * P(f_S1)', 
                          #'P(dDM) + P(dEST)', 'P(dEST) + P(E_S1)', 'P(dDM) - P(f_S1)', 'P(f_S1) - P(HOMO)',
                          ])
#print(Xs)
yt = pd.DataFrame(data_t['Yield'],columns=['Yield'])
Xt = data_t.drop(columns=['Yield',
                          #'HOMO', 'E_S1', 'f_S1', 'E_T1', 'dDM', 'P(dDM) * P(E_S1)', 'P(dDM) * P(dEST)', 'P(dEST) * P(f_S1)', 
                          #'P(dDM) + P(dEST)', 'P(dEST) + P(E_S1)', 'P(dDM) - P(f_S1)', 'P(f_S1) - P(HOMO)',
                          'Reaction_CO_1.5h', 'Reaction_CO_7.5h', 'Reaction_CO_biphenyl', 'Reaction_CO_ortho',
                          #'Reaction_CO_Cl', 'Reaction_CS', 'Reaction_CN',
                          ])
#print(Xt)

Xt_lab, Xt_unlab, yt_lab, yt_unlab = train_test_split(Xt, yt, test_size=0.9, random_state=89)
#print(Xt_lab)

model = TrAdaBoostR2(hgb(), n_estimators=23, Xt=Xt_lab, yt=yt_lab, random_state=0, verbose=0)
model.fit(Xs, ys)
y_pred = model.predict(Xt_unlab)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(-10,100),range(-10,100), c = "black")
plt.scatter(yt_unlab, y_pred)
plt.xlim(0,100)
plt.ylim(-10,100)
ax.set_aspect('equal', adjustable='box')
plt.xlabel("experiment")
plt.ylabel("prediction")
plt.show()
fig.savefig("result/TrAB_0.9.pdf")

r2 = metrics.r2_score(yt_unlab, y_pred)
mae = metrics.mean_absolute_error(yt_unlab, y_pred)
uns = model.unsupervised_score(Xs, Xt)

print("r2:", r2)
print("mae:", mae)
print("unsupervised score:", uns)

yt_unlab = np.array(yt_unlab)
yt_unlab = pd.DataFrame(yt_unlab, columns=['Yield'])
y_pred = pd.DataFrame(y_pred, columns=["Pred_Yield"])
ID_unlab = np.array(ID_unlab)
ID_unlab = pd.DataFrame(ID_unlab, columns=["Name", "ID"])
yield_data = pd.concat([ID_unlab, yt_unlab, y_pred], axis=1, join='inner')
print(yield_data)
yield_data.to_csv('result/yield_data.csv')


'P(E_S1) * P(E_T1)',
'P(E_S1) * P(HOMO)',
'P(E_S1) * P(f_S1)',
'P(E_T1) * P(HOMO)',
'P(E_T1) * P(f_S1)',
'P(dDM) * P(E_S1)',
'P(dDM) * P(E_T1)',
'P(dDM) * P(HOMO)',
'P(dDM) * P(dEST)',
'P(dDM) * P(f_S1)',
'P(dEST) * P(E_S1)',
'P(dEST) * P(E_T1)',
'P(dEST) * P(HOMO)',
'P(dEST) * P(f_S1)',
'P(f_S1) * P(HOMO)',
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('result/prediction_iso_S1_top5.csv')
data = data.drop(columns=['Unnamed: 0', 'Name', 'ID'])
pred = data.drop(columns=['Exp_yield'])
exp = data.drop(columns=['Pred_yield'])
print(pred)
print(exp)

print(mean_absolute_error(pred, exp))

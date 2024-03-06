# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:56:28 2023

@author: noton
"""

import pandas as pd

data = pd.read_csv('result/result_unlab0.5_S1FE_tFE_ne11.csv')
data = data.drop(columns=['Unnamed: 0', 'r2_all', 'mae_all', 'r2_lab', 'mae_lab',
                          #'unsupervised',
                          ])
#print(data)

data_max = data.max()
data_min = data.min()
data_mean = data.mean()
data_std = data.std(ddof=0)

print('max:')
print(data_max)
print('min:')
print(data_min)
print('mean:')
print(data_mean)
print('std:')
print(data_std)

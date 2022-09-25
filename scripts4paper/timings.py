#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sibirbil
"""
# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from locreg import LocalRegression
from magging import Magging
from less import LESSRegressor
import datasetsr as DS
import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.color_palette('pastel')

fig, axs = plt.subplots(1, 2, figsize=(10,4))

# problem = DS.energy
problem = DS.superconduct

df = np.array(problem('./datasetsR/'))
X = df[:, 0:-1]
y = df[:, -1]

X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(np.expand_dims(y, -1))
y = np.reshape(y, (len(y),))


LessTimes = []
grids = np.linspace(0.01, 0.40, 50)
# grids[5] is the default fraction for LESS
for frac in grids:
    est = LESSRegressor(scaling=False, global_estimator=lambda: LinearRegression(), frac=frac)
    starttime = time.time()
    est.fit(X, y)
    endtime = time.time()
    LessTimes.append(endtime-starttime)

dftimes = pd.DataFrame({'Percentage of Samples':100*grids, 
                   'Times (sec.)': np.array(LessTimes)})

f1 = sns.scatterplot(x='Percentage of Samples', y='Times (sec.)',
                data=dftimes, color='red', alpha=0.6, ax=axs[0],  s=100)

f1.set_xlabel('Percentage of Samples')
f1.set_ylabel('Time (sec.)')
f1.set_title('LESS')

AllTimes = {}
est = RandomForestRegressor()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['RF'] = [endtime-starttime]

est = AdaBoostRegressor()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['ADA'] = [endtime-starttime]

est = GradientBoostingRegressor()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['GB'] = [endtime-starttime]

est = KNeighborsRegressor()
starttime = time.time()
est.fit(X, y)
est.predict(X) # Since it is a lazy learner
endtime = time.time()
AllTimes['KNN'] = [endtime-starttime]

est = DecisionTreeRegressor()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['DT'] = [endtime-starttime]

est = SVR()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['SVR'] = [endtime-starttime]

est = MLPRegressor(max_iter=5000)
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['MLP'] = [endtime-starttime]

est = LinearRegression()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['LR'] = [endtime-starttime]

est = LocalRegression()
starttime = time.time()
est.fit(X, y)
est.predict(X) # Since it is a lazy learner
endtime = time.time()
AllTimes['LocR'] = [endtime-starttime]

est = Magging()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['MAG'] = [endtime-starttime]

est = KernelRidge(kernel='rbf')
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['KR'] = [endtime-starttime]

est = GaussianProcessRegressor()
starttime = time.time()
est.fit(X, y)
endtime = time.time()
AllTimes['GPR'] = [endtime-starttime]

dfalltimes = pd.DataFrame(AllTimes)

f2 = sns.barplot(data=dfalltimes, ax=axs[1])
f2.set_title('Other Methods')
f2.axhline(LessTimes[5], color='red', alpha=0.6, 
           linestyle='--', linewidth=3.0)
f2.set_yscale('log')
f2.set_ylabel('Time (log(sec.))')

plt.tight_layout()
plt.show()
# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sibirbil
"""
from typing import List, Callable
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from less import LESSRegressor, rbf
import datasetsr as DS

numOutCV: int = 5
numInCV: int = 4

randomState = 16252329

fname: str = './results/less_ablation_'

problems: List[Callable[[str], DataFrame]] = [DS.abalone, DS.airfoil, DS.housing, 
                                              DS.cadata, DS.ccpp, DS.energy,
                                              DS.cpusmallscale, DS.superconduct]

def solve_problem(problem):
    
    pname = problem.__name__.upper()

    # Server
    # df = np.array(problem('/scratch/datasets/'))
    # Local
    df = np.array(problem('./datasetsR/')) 
    
    X = df[:, 0:-1]
    y = df[:, -1]
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    y = np.reshape(y, (len(y),))
    
    # Initializing Regressors
    LESS_NoDis_NoGF=LESSRegressor(scaling=False, random_state = randomState, global_estimator=None, distance_function=lambda data, center: rbf(data, center, coeff=0.0))
    LESS_NoDis_GF=LESSRegressor(scaling=False, random_state = randomState, distance_function=lambda data, center: rbf(data, center, coeff=0.0), global_estimator=lambda: LinearRegression())
    LESS_Dis_NoGF=LESSRegressor(scaling=False, random_state = randomState, global_estimator=None)
    LESS_default=LESSRegressor(scaling=False, random_state = randomState, global_estimator=lambda: LinearRegression())
    
    # Setting up the parameter grids
    LESS_pgrid = {'frac': [0.01, 0.05, 0.10, 0.20]}
    
    scores = {'LESS_NoDis_NoGF': [], 
              'LESS_NoDis_GF': [], 
              'LESS_Dis_NoGF': [],
              'LESS_default': []}
    
    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=randomState)

    foldnum = 0
    for train_index, test_index in skf.split(X, y):
        foldnum += 1
        print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        SC =  StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.transform(X_test)

        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=randomState)
        
        for pgrid, est, name in zip((LESS_pgrid, LESS_pgrid, LESS_pgrid, LESS_pgrid),
                                    (LESS_NoDis_NoGF, LESS_NoDis_GF, LESS_Dis_NoGF, LESS_default),
                                    ('LESS_NoDis_NoGF', 'LESS_NoDis_GF', 'LESS_Dis_NoGF', 'LESS_default')):
            
            # print('Problem: {0} \t Method: {1}'.format(pname, name))
            gcv = GridSearchCV(estimator=est, param_grid=pgrid,
                                n_jobs=4, cv=inner_cv, verbose=0, refit=True)
            gcv_fit = gcv.fit(X_train, y_train)
            
            # Evaluate with the best estimator
            gcv_pred = gcv_fit.best_estimator_.predict(X_test)
            scores[name].append(mean_squared_error(gcv_pred, y_test))
    
    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Method: Average & Std. Dev.\n', file=f)
                
        for method in scores.keys():
            txt = '{0}:  \t{1:.4f} \t {2:.4f}'.format(method, np.mean(scores[method]), np.std(scores[method]))
            print(txt, file=f)
            
        print('<---\n', file=f)


####################
# Solve all problems
for problem in problems:
    solve_problem(problem)
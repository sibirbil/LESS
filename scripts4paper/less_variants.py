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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from less.less import LESSRegressor, rbf
import datasetsr as DS

numOutCV: int = 5
numInCV: int = 4

randomState = 16252329

fname: str = './results/less_variants_'

problems: List[Callable[[str], DataFrame]] = [DS.abalone, DS.airfoil, DS.housing, 
                                              DS.cadata, DS.ccpp, DS.energy,
                                              DS.cpusmallscale, DS.superconduct]

def solve_problem(problem):

    pname = problem.__name__.upper()

    #Server
    df = np.array(problem('/scratch/datasets/'))
    # Local
    # df = np.array(problem('./datasetsR/'))
    
    X = df[:, 0:-1]
    y = df[:, -1]
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    y = np.reshape(y, (len(y),))
    
    # Initializing Regressors
    LESS_NoVal=LESSRegressor(random_state = randomState)
    LESS_Val=LESSRegressor(random_state = randomState, val_size=0.3)
    LESS_C_NoVal=LESSRegressor(random_state = randomState)
    LESS_C_Val=LESSRegressor(random_state = randomState, val_size=0.3)
    
    # Setting up the parameter grids
    LESS_pgrid = {'frac': [0.01, 0.05, 0.10, 0.20],
                  'distance_function': [None,
                                    lambda data, center: rbf(data, center, coeff=0.01),
                                    lambda data, center: rbf(data, center, coeff=0.10)],
                  'local_estimator':  [LinearRegression,
                                       DecisionTreeRegressor],
                  'global_estimator': [LinearRegression,
                                       RandomForestRegressor]}
    
    LESSC_pgrid = {'cluster_method': [lambda : KMeans(n_clusters=100),
                                     lambda : KMeans(n_clusters=20),
                                     lambda : KMeans(n_clusters=10),
                                     lambda : KMeans(n_clusters=5)],
                  'distance_function': [None,
                                    lambda data, center: rbf(data, center, coeff=0.01),
                                    lambda data, center: rbf(data, center, coeff=0.10)],
                  'local_estimator':  [LinearRegression,
                                       DecisionTreeRegressor],
                  'global_estimator': [LinearRegression,
                                       RandomForestRegressor]}
    
    scores = {'LESS_NoVal': [], 
              'LESS_Val': [], 
              'LESS_C_NoVal': [],
              'LESS_C_Val': []}
    
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
        
        for pgrid, est, name in zip((LESS_pgrid, LESS_pgrid, LESSC_pgrid, LESSC_pgrid),
                                    (LESS_NoVal, LESS_Val, LESS_C_NoVal, LESS_C_Val),
                                    ('LESS_NoVal', 'LESS_Val', 'LESS_C_NoVal', 'LESS_C_Val')):
            
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
# Solve all problems
for problem in problems:
    solve_problem(problem)

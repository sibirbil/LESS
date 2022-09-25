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
from less.less import LESSRegressor, rbf
import datasetsr as DS

numOutCV: int = 5
numInCV: int = 4

randomState = 16252329

fname: str = './results/less_cv_'

problems: List[Callable[[str], DataFrame]] = [DS.abalone, DS.airfoil, DS.housing, 
                                              DS.cadata, DS.ccpp, DS.energy,
                                              DS.cpusmallscale, DS.superconduct]

def solve_problem(problem):
    
    pname = problem.__name__.upper()

    # Server
    df = np.array(problem('/scratch/datasets/'))
    # Local
    # df = np.array(problem('./datasetsR/'))
       
    X = df[:, 0:-1]
    y = df[:, -1]
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    y = np.reshape(y, (len(y),))
    
    # Initializing Regressors
    LESSestimator = LESSRegressor(random_state = randomState)
    RFestimator = RandomForestRegressor(random_state=randomState)
    ADAestimator = AdaBoostRegressor(random_state=randomState)
    GBestimator = GradientBoostingRegressor(random_state=randomState)
    KNNestimator = KNeighborsRegressor()
    DTestimator = DecisionTreeRegressor(random_state=randomState)
    SVRestimator = SVR()    
    MLPestimator = MLPRegressor(learning_rate='adaptive', max_iter=10000, random_state=randomState)
    LRestimator = LinearRegression()
    LocRestimator = LocalRegression()
    KRestimator = KernelRidge()
    MAGestimator = Magging(random_state=randomState)
    GPRestimator= GaussianProcessRegressor(random_state=randomState)
    
    # Setting up the parameter grids
    LESS_pgrid = {'frac': [0.01, 0.05, 0.10, 0.20],
                  'distance_function': [None,
                                        lambda data, center: rbf(data, center, coeff=0.01),
                                        lambda data, center: rbf(data, center, coeff=0.10)],
                  'local_estimator':  [LinearRegression,
                                       DecisionTreeRegressor],
                  'global_estimator': [LinearRegression,
                                       RandomForestRegressor]}
    RF_pgrid = {'n_estimators': [100, 200]}
    ADA_pgrid = {'n_estimators': [50, 100]}
    GB_pgrid = {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [100, 200]}
    KNN_pgrid = {'n_neighbors': [3, 5, 10, 20, 50, 100]}
    DT_pgrid = {'min_samples_split': [2, 4]}
    SVR_pgrid = {'C': [0.1, 1.0, 10.0, 100.0]}
    MLP_pgrid = {'alpha': [0.0001, 0.001, 0.01]}
    LR_pgrid = {}
    LocR_pgrid = {'frac': [0.01, 0.05, 0.10, 0.20]}
    MAG_pgrid = {'frac': [0.10, 0.20, 0.30]}
    KR_pgrid = {'kernel': ['poly', 'rbf']}
    GPR_pgrid = {'alpha': [1.0e-1, 1.0e-4, 1.0e-8]}
    
    scores = {'LESS': [], 'RF': [], 'ADA': [], 'GB': [],
              'KNN': [], 'DT': [], 'SVR':[], 'MLP': [], 'LR': [],
              'LocR': [], 'MAG':[], 'KR': [], 'GPR': []}    
    
    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=randomState)

    foldnum = 0
    for train_index, test_index in skf.split(X, y):
        foldnum += 1
        print('---------------------------------')
        print('Problem: {0} \t Fold: {1}'.format(pname, foldnum))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        SC =  StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.transform(X_test)
        
        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=randomState)
        
        for pgrid, est, name in zip((LESS_pgrid, RF_pgrid, ADA_pgrid, GB_pgrid,
                                     KNN_pgrid, DT_pgrid, SVR_pgrid, MLP_pgrid, 
                                     LR_pgrid, LocR_pgrid, MAG_pgrid, KR_pgrid, GPR_pgrid),
                                    (LESSestimator, RFestimator, ADAestimator, GBestimator,
                                     KNNestimator, DTestimator,  SVRestimator, MLPestimator, 
                                     LRestimator, LocRestimator, MAGestimator, KRestimator, GPRestimator),
                                    ('LESS', 'RF', 'ADA', 'GB', 'KNN',
                                     'DT', 'SVR', 'MLP', 'LR', 
                                     'LocR', 'MAG', 'KR', 'GPR')):
            
            # print('Problem: {0} \t Method: {1}'.format(pname, name))
            gcv = GridSearchCV(estimator=est, param_grid=pgrid,
                                n_jobs=4, cv=inner_cv, verbose=0, refit=True)
            gcv_fit = gcv.fit(X_train, y_train)
            if (pgrid == LESS_pgrid):
                print(gcv.best_estimator_)
                print('---------------------------------')

            # Evaluate with the best estimator
            gcv_pred = gcv_fit.best_estimator_.predict(X_test)
            scores[name].append(mean_squared_error(gcv_pred, y_test))

    fnamefull = fname + pname + '.txt'
    with open(fnamefull, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Method: Average & Std. Dev.\n', file=f)
                
        for method in scores.keys():
            txt = '{0}:  \t{1}'.format(method, scores[method])
            print(txt, file=f)
            txt = '{0}:  \t{1:.4f} \t {2:.4f}'.format(method, np.mean(scores[method]), np.std(scores[method]))
            print(txt, file=f)
            
        print('<---\n', file=f)

####################
# Solve all problems
for problem in problems:
    solve_problem(problem)

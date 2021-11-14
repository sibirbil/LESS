#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ilker Birbil @ UvA
"""
# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from less import LESSRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# %%

def synthetic_sine_curve(n_samples=200):
    '''
    A simple function to generate n_samples from sine curve
    in the range (-10, 10) with some amplitude. The function
    returns the dataset (X, y), and plots the function (curve)
    along with the dataset (circles).
    '''

    sns.set_style('ticks')
    sns.color_palette('pastel')
    plt.figure(figsize = (6,2))

    xvals = np.arange(-10, 10, 0.1) # domain

    n_all = n_samples
    X = np.zeros((n_all, 1))
    y = np.zeros(n_all)
    for i in range(n_all):
        xran = -10 + 20*np.random.rand()
        X[i] = xran
        y[i] = 10*np.sin(xran) + 2.5*np.random.randn()

    f1 = sns.lineplot(x=xvals, y=10*np.sin(xvals), color='red')
    f1 = sns.scatterplot(x = X[:,0], y = y, alpha=0.5)
    f1.set_ylim([-15.0, 15.0])
    f1.set(xticklabels=[], yticklabels=[], title='Synthetic Data');
    
    plt.tight_layout()
    plt.show()
        
    return X, y



def compare_1D_plots(X, y):
    '''
    Plots the fitted functions obtained with various
    regressors (using their default values) on the
    one-dimensional dataset (X, y).
    '''
    _ , dim  = X.shape
    
    if (dim > 1):
        raise ValueError('Works only for one-dimensional data.')

    sns.set_style('ticks')
    sns.color_palette('pastel')

    fig, axs = plt.subplots(5, 2, figsize=(10,10))
    fig.subplots_adjust(wspace = 0.2, hspace = 0.3)

    palette = iter(sns.husl_palette(10))
    xlb, xub = np.floor(np.min(X)-1), np.ceil(np.max(X)+1)
    xvals = np.arange(xlb, xub, 0.1) # domain
    ylb, yub = np.floor(np.min(y)-1), np.ceil(np.max(y)+1)
    
    LESSestimator = LESSRegressor()
    LESS_fit = LESSestimator.fit(X, y)
    f1 = sns.lineplot(x=xvals, y=LESS_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[0, 0], color=next(palette))
    f1.set_ylim([ylb, yub])
    f1.set(xticklabels=[], yticklabels=[], title='LESS')

    RF = RandomForestRegressor()
    RF_fit = RF.fit(X, y)
    f2 = sns.lineplot(x=xvals, y=RF_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[0, 1], color=next(palette))
    f2.set_ylim([ylb, yub])
    f2.set(xticklabels=[], yticklabels=[], title='Random Forest')

    ADA = AdaBoostRegressor()
    ADA_fit = ADA.fit(X, y)
    f3 = sns.lineplot(x=xvals, y=ADA_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[1, 0], color=next(palette))
    f3.set_ylim([ylb, yub])
    f3.set(xticklabels=[], yticklabels=[], title='AdaBoost')

    KNN = KNeighborsRegressor(n_neighbors=5)
    KNN_fit = KNN.fit(X, y)
    f4 = sns.lineplot(x=xvals, y=KNN_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[1, 1], color=next(palette))
    f4.set_ylim([ylb, yub])
    f4.set(xticklabels=[], yticklabels=[], title='$K$-Nearest Neighbors')

    DT = DecisionTreeRegressor()
    DT_fit = DT.fit(X, y)
    f5 = sns.lineplot(x=xvals, y=DT_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[2, 0], color=next(palette))
    f5.set_ylim([ylb, yub])
    f5.set(xticklabels=[], yticklabels=[], title='Decision Tree')

    SVM = SVR(kernel='rbf', C=100.0)
    SVM_fit = SVM.fit(X, y)
    f6 = sns.lineplot(x=xvals, y=SVM_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[2, 1], color=next(palette))
    f6.set_ylim([ylb, yub])
    f6.set(xticklabels=[], yticklabels=[], title='Support Vector Regression')

    MLP = MLPRegressor(max_iter=50000)
    MLP_fit =MLP.fit(X, y)
    f7 = sns.lineplot(x=xvals, y=MLP_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[3, 0], color=next(palette))
    f7.set_ylim([ylb, yub])
    f7.set(xticklabels=[], yticklabels=[], title='Neural Network')

    LR = LinearRegression()
    LR_fit = LR.fit(X, y)
    f8 = sns.lineplot(x=xvals, y=LR_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[3, 1], color=next(palette))
    f8.set_ylim([ylb, yub])
    f8.set(xticklabels=[], yticklabels=[], title='Linear Regression')

    KR = KernelRidge(kernel='rbf')
    KR_fit = KR.fit(X, y)
    f9 = sns.lineplot(x=xvals, y=KR_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[4, 0], color=next(palette))
    f9.set_ylim([ylb, yub])
    f9.set(xticklabels=[], yticklabels=[], title='Kernel Ridge (RBF)')

    GPR = GaussianProcessRegressor()
    GPR_fit = GPR.fit(X, y)
    f10 = sns.lineplot(x=xvals, y=GPR_fit.predict(xvals.reshape(len(xvals), 1)), ax=axs[4, 1], color=next(palette))
    f10.set_ylim([ylb, yub])
    f10.set(xticklabels=[], yticklabels=[], title='Gaussian Process')
    
    plt.tight_layout()
    plt.show()

def less_bias_vs_variance(X, y):
    '''
    Plots the changes in the fitted function obtained
    with LESS on the one-dimensional dataset
    (X, y). The panel of plots is given for various 
    values of:
        number of subsets,
        number of neighbors,
        number of replications.
    '''
    
    _ , dim = X.shape
    
    if (dim > 1):
        raise ValueError('Works only for one-dimensional data.')
        
    sns.set_style('ticks')
    sns.color_palette('pastel')

    _ , axs = plt.subplots(3, 4, figsize=(13,8))

    palette = iter(sns.husl_palette(12)) 

    xlb, xub = np.floor(np.min(X)-1), np.ceil(np.max(X)+1)
    xvals = np.arange(xlb, xub, 0.1) # domain
    ylb, yub = np.floor(np.min(y)-1), np.ceil(np.max(y)+1)

    repvals = [5, 10, 20]
    fracvals = [0.20, 0.10, 0.05, 0.02]

    for i, rep in enumerate(repvals):
        for j, fr in enumerate(fracvals):
            LESSestimator = LESSRegressor(frac=fr, n_replications=rep)
            LESS_fit = LESSestimator.fit(X, y)
            fmy = sns.lineplot(x=xvals, y=LESS_fit.predict(xvals.reshape(len(xvals), 1)),\
                ax=axs[i, j], color=next(palette))
            fmy.set_title('Fraction of Samples (frac=' + str(LESS_fit.frac_) + ')')
            fmy.set_ylim([ylb, yub])
            fmy.set(xticklabels=[])
            fmy.set(yticklabels=[])
            txt = str(LESS_fit.n_replications_) + ' replications'
            fmy.text(xlb, yub-3, txt, color='red')
            txt = '(' + str(LESS_fit.n_subsets_) + ' subsets and ' \
                    + str(LESS_fit.n_neighbors_) + ' neighbors)'
            fmy.text(xlb, ylb+1, txt)
            
    plt.tight_layout()
    plt.show()
    
def timings(X, y):
    '''
    Returns a barplot of training times of
    various regressors (using their default
    values) on the one-dimensional dataset (X, y).
    '''

    AllTimes = {}
    
    est = LESSRegressor()
    starttime = time.time()
    est.fit(X, y)
    endtime = time.time()
    AllTimes['LESS'] = [endtime-starttime]
    
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

    f1 = sns.barplot(data=dfalltimes)
    f1.set_title('LESS vs. Other Methods')
    f1.set_yscale('log')
    f1.set_ylabel('Time (log(sec.))')

    plt.tight_layout()
    plt.show()
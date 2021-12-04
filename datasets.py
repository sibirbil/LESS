#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ilker Birbil @ UvA
"""

import pandas as pd

### REGRESSION

def abalone(wd): 
    '''
    4176 x 7
    The first categorical feature is removed
    http://archive.ics.uci.edu/ml/datasets/Abalone
    '''
    df = pd.read_csv(wd+'abalone.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def energy(wd):
    '''
    19735 x 26
    https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    '''
    df = pd.read_csv(wd+'energydata_complete.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

def superconduct(wd):
    '''
    21263 x 81
    https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
    '''
    df = pd.read_csv(wd+'superconduct.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df

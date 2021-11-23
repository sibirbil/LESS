#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ilker Birbil @ UvA
"""
from typing import List, Optional, Callable, NamedTuple
import warnings
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

############################
# Supporting classes

class SklearnEstimator:
    '''
    This base class is dummy. It is used just for guideline. 
    '''
    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError('Needs to implement fit(X, y)')

    def predict(self, X0: np.array):
        raise NotImplementedError('Needs to implement predict(X, y)')

class LocalModelR(NamedTuple):
    estimator: SklearnEstimator
    center: np.array

class ReplicationR(NamedTuple):
    global_estimator: SklearnEstimator
    local_estimators: List[LocalModelR]

############################    


############################
def rbf(data, center, coeff=0.01):
    '''
    RBF kernel - L2 norm
    This is is used by the default distance function in LESS 
    '''
    return np.exp(-coeff * np.linalg.norm(np.array(data - center, dtype=float), ord=2, axis=1))

############################


class LESSRegressor(RegressorMixin, BaseEstimator, SklearnEstimator):
    '''
    Parameters
    ----------
        frac: fraction of total samples used for number of neighbors (default is 0.05)
        n_neighbors : number of neighbors (default is None)
        n_subsets : number of subsets (default is None)
        n_replications : number of replications (default is 50)
        d_normalize : distance normalization (default is True)
        val_size: percentage of samples used for validation (default is None - no validation)
        random_state: initialization of the random seed (default is None)
        tree_method : method used for constructing the nearest neighbor tree,
                e.g., sklearn.neighbors.KDTree (default) or sklearn.neighbors.BallTree
        cluster_method : method used for clustering the subsets,
                e.g., sklearn.cluster.KMeans, sklearn.cluster.SpectralClustering (default is None)
        local_estimator : estimator for training the local models (default is LinearRegression)
        global_estimator : estimator for training the global model (default is LinearRegression)
        distance_function : distance function evaluating the distance from a subset to a sample,
                e.g., df(subset, sample) which returns a vector of distances
                (default is RBF(subset, sample, 1.0/n_subsets^2))
    '''
    def __init__(self, frac=None, n_neighbors=None, n_subsets=None, 
                 n_replications=20, d_normalize=True, val_size=None, random_state=None,
                 tree_method=lambda data, n_subsets: KDTree(data, n_subsets),
                 cluster_method=None,
                 local_estimator=lambda: LinearRegression(),
                 global_estimator=lambda: LinearRegression(),
                 distance_function: Callable[[np.array, np.array], np.array]=None):
        
        self.local_estimator = local_estimator
        self.global_estimator = global_estimator
        self.tree_method = tree_method
        self.cluster_method = cluster_method
        self.distance_function = distance_function
        self.frac = frac
        self.n_neighbors = n_neighbors
        self.n_subsets = n_subsets
        self.n_replications = n_replications
        self.d_normalize = d_normalize
        self.val_size = val_size
        self.random_state = random_state
    
    def _set_local_attributes(self):
        '''
        Storing the local variables and
        checking the given parameters
        '''

        self._rng = np.random.default_rng(self.random_state)
        self._replications: Optional[List[ReplicationR]] = None
        self._isfitted = False
        
        if(self.local_estimator == None):
            raise ValueError('LESS does not work without a local estimator.')
            
        if (self.val_size != None):            
            if(self.val_size <= 0.0 or self.val_size >= 1.0):
                raise ValueError('Parameter val_size should be in the interval (0, 1).')
        
        if(self.frac != None):
            if(self.frac <= 0.0 or self.frac > 1.0):
                raise ValueError('Parameter frac should be in the interval (0, 1].')

        if (self.n_replications < 1):
            raise ValueError('The number of replications should greater than equal to one.')
                        
        if (self.cluster_method != None):
            if (self.frac != None):
                warnings.warn('Both frac and cluster_method parameters are provided. \
                              Proceeding with clustering...')
                self.frac = None
                
            if ('n_clusters' in self.cluster_method().get_params().keys()):
                if (self.cluster_method().get_params()['n_clusters'] == 1):
                    warnings.warn('There is only one cluster, so the \
                                  global estimator is set to none.')
                    # If no global estimator is defined, then we output
                    # the average of the local estimators by assigning 
                    # the weight (1/self.n_subsets) to each local estimator
                    self.global_estimator = None
                    self.d_normalize = True
                    # If there is also no validation step, then there is 
                    # no randomness. So, no need for replications.
                    if (self.val_size == None):
                        warnings.warn('Since validation set is not used, \
                            there is no randomness, and hence, \
                                no need for replications.')                        
                        self.n_replications = 1
        elif(self.frac == None and 
             self.n_neighbors== None and
             self.n_subsets == None):
            self.frac = 0.05
                    
    def _check_input(self, len_X: int):
        '''
        Checks whether the input is valid,
        where len_X is the length of input data
        '''

        if (self.cluster_method == None):
            
            if (self.frac != None):
                self.n_neighbors = int(np.ceil(self.frac * len_X))
                self.n_subsets = int(len_X/self.n_neighbors)
                self.n_neighbors = int(len_X/self.n_subsets)
                
            if (self.n_subsets == None):
                self.n_subsets = int(len_X/self.n_neighbors)
            
            if (self.n_neighbors == None):
                self.n_neighbors = int(len_X/self.n_subsets)
            
            if (self.n_neighbors >= len_X):
                warnings.warn('The number of neighbors is larger than \
                    the number of samples. Setting number of subsets to one.')
                self.n_neighbors = len_X
                self.n_subsets = 1
                
            if (self.n_subsets >= len_X):
                warnings.warn('The number of subsets is larger than \
                    the number of samples. Setting number of neighbors to one.')            
                self.n_neighbors = 1
                self.n_subsets = len_X 
            
            if (self.n_subsets == 1):
                warnings.warn('There is only one subset, so the \
                    global estimator is set to none.')
                self.global_estimator = None
                self.d_normalize = True
        else:
                # When we use clustering, the number of 
                # subsets may differ in each replication
                self.frac = None
                self.n_neighbors=None
                self.n_subsets = []


    def fit(self, X: np.array, y: np.array):
        '''
        Dummy fit function that calls the proper method
        according to validation and clustering parameters.
        Options are:
        - Default fitting (no validation set, no clustering)
        - Fitting with validation set (no clustering)
        - Fitting with clustering (no) validation set)
        - Fitting with validation set and clustering
        '''
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self._set_local_attributes()
        
        if (self.val_size != None):
            # Validation set is not used for
            # global estimation
            if (self.cluster_method == None):
                self._fitval(X, y)
            else:
                self._fitvalc(X, y)
        else:
            # Validation set is used for
            # global estimation
            if (self.cluster_method == None):
                self._fitnoval(X, y)
            else:
                self._fitnovalc(X, y)
        
        self._isfitted = True
        
        return self

    def _fitnoval(self, X: np.array, y: np.array):
        '''
        Fit function: All data is used for global estimator (no validation)
        Tree method is used (no clustering)
        '''

        len_X: int = len(X)
        # Check the validity of the input
        self._check_input(len_X)
        # A nearest neighbor tree is grown for querying
        tree = self.tree_method(X, self.n_subsets)
        self._replications = []
        for i in range(self.n_replications):
            # Select n_subsets many samples to construct the local sample sets
            sample_indices = self._rng.choice(len_X, size=self.n_subsets)
            # Construct the local sample sets
            _, neighbor_indices_list = tree.query(X[sample_indices], k=self.n_neighbors)
            local_models: List[LocalModelR] = []
            dists = np.zeros((len_X, self.n_subsets))            
            predicts = np.zeros((len_X, self.n_subsets))
            for neighbor_i, neighbor_indices in enumerate(neighbor_indices_list):
                Xneighbors, yneighbors = X[neighbor_indices], y[neighbor_indices]
                # Centroid is used as the center of the local sample set
                local_center = np.mean(Xneighbors, axis=0)
                if ('random_state' in self.local_estimator().get_params().keys()):
                    local_model = self.local_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(Xneighbors, yneighbors)
                else:
                    local_model = self.local_estimator().fit(Xneighbors, yneighbors)
                local_models.append(LocalModelR(estimator=local_model, center=local_center))
                predicts[:, neighbor_i] = local_model.predict(X)
                
                if (self.distance_function == None):
                    dists[:, neighbor_i] = rbf(X, local_center, \
                        coeff=1.0/np.power(self.n_subsets, 2.0))
                else:
                    dists[:, neighbor_i] = self.distance_function(X, local_center)

            # Normalize the distances from each sample to the local subsets
            if (self.d_normalize):
                denom = np.sum(dists, axis=1)
                denom[denom < 1.0e-8] = 1.0e-8
                dists = (dists.T/denom).T
            
            if (self.global_estimator != None):
                if ('random_state' in self.global_estimator().get_params().keys()):
                    global_model = self.global_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                        fit(dists * predicts, y)
                else:
                    global_model = self.global_estimator().fit(dists * predicts, y)
            else:
                global_model = None

            self._replications.append(ReplicationR(global_model, local_models))

        return self


    def _fitval(self, X: np.array, y: np.array):
        '''
        Fit function: (val_size x data) is used for global estimator (validation)
        Tree method is used (no clustering)
        '''

        self._replications = []
        for i in range(self.n_replications):
            # Split for global estimation
            X_train, X_val, y_train, y_val = train_test_split(X, y,
                test_size=self.val_size,
                random_state=self._rng.integers(np.iinfo(np.int16).max))

            len_X_val: int = len(X_val)
            len_X_train: int = len(X_train)
            # Check the validity of the input
            if (i==0):
                self._check_input(len_X_train)
            
            # A nearest neighbor tree is grown for querying
            tree = self.tree_method(X_train, self.n_subsets)
            
            # Select n_subsets many samples to construct the local sample sets
            sample_indices = self._rng.choice(len_X_train, size=self.n_subsets)
            # Construct the local sample sets
            _, neighbor_indices_list = tree.query(X_train[sample_indices], k=self.n_neighbors)
            local_models: List[LocalModelR] = []
            dists = np.zeros((len_X_val, self.n_subsets))            
            predicts = np.zeros((len_X_val, self.n_subsets))
            for neighbor_i, neighbor_indices in enumerate(neighbor_indices_list):
                Xneighbors, yneighbors = X_train[neighbor_indices], y_train[neighbor_indices]
                # Centroid is used as the center of the local sample set
                local_center = np.mean(Xneighbors, axis=0)
                if ('random_state' in self.local_estimator().get_params().keys()):
                    local_model = self.local_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(Xneighbors, yneighbors)
                else:                
                    local_model = self.local_estimator().fit(Xneighbors, yneighbors)
                local_models.append(LocalModelR(estimator=local_model, center=local_center))
                predicts[:, neighbor_i] = local_model.predict(X_val)
                
                if (self.distance_function == None):
                    dists[:, neighbor_i] = rbf(X_val, local_center, \
                        coeff=1.0/np.power(self.n_subsets, 2.0))
                else:
                    dists[:, neighbor_i] = self.distance_function(X_val, local_center)

            # Normalize the distances from each sample to the local subsets
            if (self.d_normalize):
                denom = np.sum(dists, axis=1)
                denom[denom < 1.0e-8] = 1.0e-8
                dists = (dists.T/denom).T
            
            if (self.global_estimator != None):
                if ('random_state' in self.global_estimator().get_params().keys()):
                    global_model = self.global_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(dists * predicts, y_val)
                else:
                    global_model = self.global_estimator().fit(dists * predicts, y_val)
            else:
                global_model = None

            self._replications.append(ReplicationR(global_model, local_models))

        return self

    def _fitnovalc(self, X: np.array, y: np.array):
        '''
        Fit function: All data is used for global estimator (no validation)
        Clustering is used (no tree method)
        '''

        len_X: int = len(X)
        # Check the validity of the input
        self._check_input(len_X)
        
        if ('random_state' not in self.cluster_method().get_params().keys()): 
            warnings.warn('Clustering method is not random, so there is \
                no need for replications, unless validaton set is used. \
                    Note that lack of replications may increase the variance.') 
            cluster_fit = self.cluster_method().fit(X)
            self.n_replications = 1

        self._replications = []
        for i in range(self.n_replications):
            
            if (self.n_replications > 1):
                cluster_fit = self.cluster_method().\
                    set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                        fit(X)
                        
            # Some clustering methods may find less number of
            # clusters than requested 'n_clusters'
            self.n_subsets.append(len(np.unique(cluster_fit.labels_)))
            n_subsets = self.n_subsets[i]
            
            local_models: List[LocalModelR] = []
            dists = np.zeros((len_X, n_subsets))            
            predicts = np.zeros((len_X, n_subsets))
            
            if (hasattr(cluster_fit, 'cluster_centers_')):
                use_cluster_centers = True
            else:
                use_cluster_centers = False
                
            for cluster_indx, cluster in enumerate(np.unique(cluster_fit.labels_)):
                neighbors = cluster_fit.labels_ == cluster
                Xneighbors, yneighbors = X[neighbors], y[neighbors]
                # Centroid is used as the center of the local sample set
                if (use_cluster_centers):
                    local_center = cluster_fit.cluster_centers_[cluster_indx]
                else:
                    local_center = np.mean(Xneighbors, axis=0)
                if ('random_state' in self.local_estimator().get_params().keys()):
                    local_model = self.local_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(Xneighbors, yneighbors)
                else:
                    local_model = self.local_estimator().fit(Xneighbors, yneighbors)
                local_models.append(LocalModelR(estimator=local_model, center=local_center))
                predicts[:, cluster_indx] = local_model.predict(X)
                
                if (self.distance_function == None):
                    dists[:, cluster_indx] = rbf(X, local_center, \
                        coeff=1.0/np.power(n_subsets, 2.0))
                else:
                    dists[:, cluster_indx] = self.distance_function(X, local_center)
                    

            # Normalize the distances from each sample to the local subsets
            if (self.d_normalize):
                denom = np.sum(dists, axis=1)
                denom[denom < 1.0e-8] = 1.0e-8
                dists = (dists.T/denom).T
            
            if (self.global_estimator != None):
                if ('random_state' in self.global_estimator().get_params().keys()):
                    global_model = self.global_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                        fit(dists * predicts, y)
                else:
                    global_model = self.global_estimator().fit(dists * predicts, y)
            else:
                global_model = None

            self._replications.append(ReplicationR(global_model, local_models))

        return self

    def _fitvalc(self, X: np.array, y: np.array):
        '''
        Fit function: (val_size x data) is used for global estimator (validation)
        Clustering is used (no tree method)
        '''

        self._replications = []
        for i in range(self.n_replications):
            # Split for global estimation
            X_train, X_val, y_train, y_val = train_test_split(X, y,
                test_size=self.val_size,
                random_state=self._rng.integers(np.iinfo(np.int16).max))

            len_X_val: int = len(X_val)
            len_X_train: int = len(X_train)
            # Check the validity of the input
            if (i == 0):
                self._check_input(len_X_train)
            
            if ('random_state' in self.cluster_method().get_params().keys()):
                cluster_fit = self.cluster_method().\
                    set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                        fit(X_train)
            else:
                cluster_fit = self.cluster_method().fit(X_train)

            if (i == 0):
                if (hasattr(cluster_fit, 'cluster_centers_')):
                    use_cluster_centers = True
                else:
                    use_cluster_centers = False
                
            # Since each replication returns
            self.n_subsets.append(len(np.unique(cluster_fit.labels_)))
            n_subsets = self.n_subsets[i]
            
            local_models: List[LocalModelR] = []
            dists = np.zeros((len_X_val, n_subsets))            
            predicts = np.zeros((len_X_val, n_subsets))
            for cluster_indx, cluster in enumerate(np.unique(cluster_fit.labels_)):
                neighbors = cluster_fit.labels_ == cluster
                Xneighbors, yneighbors = X_train[neighbors], y_train[neighbors]
                # Centroid is used as the center of the local sample set
                if (use_cluster_centers):
                    local_center = cluster_fit.cluster_centers_[cluster_indx]
                else:
                    local_center = np.mean(Xneighbors, axis=0)
                if ('random_state' in self.local_estimator().get_params().keys()):
                    local_model = self.local_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(Xneighbors, yneighbors)
                else:                
                    local_model = self.local_estimator().fit(Xneighbors, yneighbors)
                local_models.append(LocalModelR(estimator=local_model, center=local_center))
                predicts[:, cluster_indx] = local_model.predict(X_val)
                
                if (self.distance_function == None):
                    dists[:, cluster_indx] = rbf(X_val, local_center, \
                        coeff=1.0/np.power(n_subsets, 2.0))
                else:
                    dists[:, cluster_indx] = self.distance_function(X_val, local_center)

            # Normalize the distances from each sample to the local subsets
            if (self.d_normalize):
                denom = np.sum(dists, axis=1)
                denom[denom < 1.0e-8] = 1.0e-8
                dists = (dists.T/denom).T
            
            if (self.global_estimator != None):
                if ('random_state' in self.global_estimator().get_params().keys()):
                    global_model = self.global_estimator().\
                        set_params(random_state=self._rng.integers(np.iinfo(np.int16).max)).\
                            fit(dists * predicts, y_val)
                else:
                    global_model = self.global_estimator().fit(dists * predicts, y_val)
            else:
                global_model = None

            self._replications.append(ReplicationR(global_model, local_models))

        return self


    def predict(self, X0: np.array):
        '''
        Predictions are evaluated for the test samples in X0
        '''
        
        check_is_fitted(self, attributes='_isfitted')
        # Input validation
        X0 = check_array(X0)

        len_X0: int = len(X0)
        yhat = np.zeros(len_X0)
        for i in range(self.n_replications):
            # Get the fitted global and local estimators
            global_model = self._replications[i].global_estimator
            local_models = self._replications[i].local_estimators
            if (self.cluster_method == None):
                n_subsets = self.n_subsets
            else:
                n_subsets = self.n_subsets[i]
            predicts = np.zeros((len_X0, n_subsets))
            dists = np.zeros((len_X0, n_subsets))
            for j in range(n_subsets):
                local_center = local_models[j].center
                local_model = local_models[j].estimator
                predicts[:, j] = local_model.predict(X0)
                
                if (self.distance_function == None):
                    dists[:, j] = rbf(X0, local_center, \
                        coeff=1.0/np.power(n_subsets, 2.0))
                else:
                    dists[:, j] = self.distance_function(X0, local_center)

            # Normalize the distances from each sample to the local subsets
            if (self.d_normalize):
                denom = np.sum(dists, axis=1)
                denom[denom < 1.0e-8] = 1.0e-8
                dists = (dists.T/denom).T

            if (global_model != None):
                yhat += global_model.predict(dists * predicts)
            else:
                yhat += np.sum(dists * predicts, axis=1)

        yhat = yhat/self.n_replications
                    
        return yhat
    
    # AUXILIARY FUNCTIONS
    
    def get_n_subsets(self):
        
        if (self._isfitted == False):
            warnings.warn('You need to fit LESS first.')

        return self.n_subsets
    
    def get_n_neighbors(self):
        
        if (self.cluster_method != None):
            warnings.warn('Number of neighbors is not fixed when clustering is used.')
        elif (self._isfitted == False):
            warnings.warn('You need to fit LESS first.')
            
        return self.n_neighbors
    
    def get_frac(self):
        
        # Fraction is set to None only if clustering method is given
        if (self.cluster_method != None):
            warnings.warn('Parameter frac is not set when clustering is used.')

        return self.frac
    
    def get_n_replications(self):
        
        return self.n_replications
    
    def get_d_normalize(self):
        
        return self.d_normalize
    
    def get_val_size(self):
        
        return self.val_size
    
    def get_random_state(self):
        
        return self.random_state
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.stats import mode
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

_LOG = logging.getLogger("less")
warnings.formatwarning = lambda m, *a, **k: f"\nWARNING: {' '.join(str(m).split())}\n"


def _rbf(x: np.ndarray, center: np.ndarray, coeff: float) -> np.ndarray:
    """Vectorised RBF kernel: exp(‑coeff * ‖x − c‖₂)."""
    return np.exp(-coeff * np.linalg.norm(x - center, axis=1))


@dataclass(slots=True)
class _LocalModel:
    estimator: BaseEstimator
    center: np.ndarray  # (n_features,)


@dataclass(slots=True)
class _Replication:
    scaler: Optional[StandardScaler]
    global_model: Optional[BaseEstimator]
    local_models: List[_LocalModel]


class _LESSBase(BaseEstimator):
    """Shared implementation for LESSClassifier / LESSRegressor."""

    def __init__(
        self,
        *,
        frac: Optional[float] = None,
        n_neighbors: Optional[int] = None,
        n_subsets: Optional[int] = None,
        n_replications: int = 20,
        d_normalize: bool = True,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None,
        tree_method: Callable[[np.ndarray, int], KDTree] = lambda x, _: KDTree(x),
        cluster_method: Optional[Callable[[], BaseEstimator]] = None,
        local_estimator: Callable[[], BaseEstimator] = LinearRegression,
        global_estimator: Optional[Callable[[], BaseEstimator]] = None,
        distance_function: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        scaling: bool = True,
        warnings_: bool = True,
    ):
        self.frac = frac
        self.n_neighbors = n_neighbors
        self.n_subsets = n_subsets
        self.n_replications = n_replications
        self.d_normalize = d_normalize
        self.val_size = val_size
        self.random_state = random_state
        self.tree_method = tree_method
        self.cluster_method = cluster_method
        self.local_estimator = local_estimator
        self.global_estimator = global_estimator
        self.distance_function = distance_function
        self.scaling = scaling
        self.warnings_ = warnings_

        self._rng = np.random.default_rng(random_state)
        self._scaler: Optional[StandardScaler] = None
        self._replications: List[_Replication] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        if self.scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        use_val = self.val_size is not None
        use_clust = self.cluster_method is not None
        self._prepare_hyperparams(len(X))

        self._replications.clear()
        for rep in range(self.n_replications):
            X_train, y_train, X_val, y_val = (
                (
                    *train_test_split(
                        X,
                        y,
                        test_size=self.val_size,
                        random_state=self._rng.integers(2**31 - 1),
                    ),
                )
                if use_val
                else (X, y, None, None)
            )

            # construct subsets
            labels, centers = self._make_subsets(X_train, use_clust)
            local_models = self._train_local(X_train, y_train, labels, centers)

            # design matrices
            Z_train, coeff = self._design_matrix(X_train, local_models)
            Z_val = (
                self._design_matrix(X_val, local_models, coeff)[0] if use_val else None
            )

            # scaling *before* global fit (bug‑fix)
            scaler = StandardScaler() if self.scaling else None
            if scaler is not None:
                Z_train = scaler.fit_transform(Z_train)
                if Z_val is not None:
                    Z_val = scaler.transform(Z_val)

            global_model = self._fit_global(Z_train, y_train, Z_val, y_val)
            self._replications.append(_Replication(scaler, global_model, local_models))

        self._is_fitted = True
        return self

    def _prepare_hyperparams(self, n: int) -> None:
        if self.cluster_method is not None:
            return
        if self.frac is None and self.n_neighbors is None and self.n_subsets is None:
            self.frac = 0.05
        if self.frac is not None:
            self.n_neighbors = max(1, int(np.ceil(self.frac * n)))
            self.n_subsets = max(1, n // self.n_neighbors)
        elif self.n_neighbors is not None and self.n_subsets is None:
            self.n_subsets = max(1, n // self.n_neighbors)
        elif self.n_subsets is not None and self.n_neighbors is None:
            self.n_neighbors = max(1, n // self.n_subsets)
        self.n_neighbors = min(self.n_neighbors, n)
        self.n_subsets = min(self.n_subsets, n)
        if self.n_subsets == 1 and self.warnings_:
            warnings.warn(
                "Only one subset – disabling global estimator & forcing normalisation."
            )
            self.global_estimator = None
            self.d_normalize = True
            if self.val_size is None:
                self.n_replications = 1

    # subset construction
    def _make_subsets(self, X: np.ndarray,  use_clust: bool):
        if use_clust:
            clust_fac = self.cluster_method()
            if "random_state" in clust_fac.get_params():
                clust_fac.set_params(random_state=self._rng.integers(2**31 - 1))
            labels = clust_fac.fit_predict(X)
            centers = (
                clust_fac.cluster_centers_
                if hasattr(clust_fac, "cluster_centers_")
                else np.vstack([X[labels == k].mean(0) for k in np.unique(labels)])
            )
        else:
            tree = self.tree_method(X, self.n_subsets)
            seeds = self._rng.choice(len(X), size=self.n_subsets, replace=False)
            idx = tree.query(X[seeds], k=self.n_neighbors, return_distance=False)
            labels = np.full(len(X), -1, np.int32)
            for s, ind in enumerate(idx):
                labels[ind] = s
            centers = np.vstack([X[labels == s].mean(0) for s in range(self.n_subsets)])
        return labels, centers

    def _train_local(self, X, y, labels, centers):
        models = []
        for s, c in enumerate(centers):
            msk = labels == s
            est = self._clone(self.local_estimator)
            est.fit(X[msk], y[msk])
            models.append(_LocalModel(est, c))
        return models

    def _design_matrix(self, X, local_models, coeff: float | None = None):
        if X is None:
            return None, coeff  # type: ignore
        n_sub = len(local_models)
        coeff = 1.0 / n_sub**2 if coeff is None else coeff
        preds = np.column_stack([lm.estimator.predict(X) for lm in local_models])
        dists = np.column_stack(
            [
                (
                    _rbf(X, lm.center, coeff)
                    if self.distance_function is None
                    else self.distance_function(X, lm.center)
                )
                for lm in local_models
            ]
        )
        if self.d_normalize:
            dists /= np.clip(dists.sum(1, keepdims=True), 1e-8, None)
        return dists * preds, coeff

    def _fit_global(self, Z_tr, y_tr, Z_val, y_val):
        if self.global_estimator is None:
            return None
        est = self._clone(self.global_estimator)
        est.fit(
            Z_val if Z_val is not None else Z_tr, y_val if Z_val is not None else y_tr
        )
        return est

    def _clone(self, fac):
        est = fac()
        if "random_state" in est.get_params():
            est.set_params(random_state=self._rng.integers(2**31 - 1))
        return est


class LESSRegressor(_LESSBase, RegressorMixin):
    def __init__(self, **kw):
        super().__init__(global_estimator=DecisionTreeRegressor, **kw)

    def predict(self, X):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        if self.scaling and self._scaler is not None:
            X = self._scaler.transform(X)
        return np.mean([self._predict_rep(X, r) for r in self._replications], axis=0)

    def _predict_rep(self, X, rep: _Replication):
        Z, _ = self._design_matrix(X, rep.local_models)
        if rep.scaler is not None:
            Z = rep.scaler.transform(Z)
        return rep.global_model.predict(Z) if rep.global_model else Z.sum(1)


class LESSClassifier(_LESSBase, ClassifierMixin):
    def __init__(self, *, multiclass: str = "ovr", **kw):
        super().__init__(global_estimator=DecisionTreeClassifier, **kw)
        self.multiclass = multiclass
        self._strategy = None
        self._is_binary = True

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._y_unique = np.unique(y)
        self._is_binary = len(self._y_unique) == 2
        if self._is_binary:
            return super().fit(X, y)
        self._strategy = self._make_strategy()
        if self.scaling:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._strategy.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        if self.scaling and self._scaler is not None:
            X = self._scaler.transform(X)
        if not self._is_binary:
            return self._strategy.predict(X)
        maj = np.column_stack([self._predict_rep(X, r) for r in self._replications])
        return mode(maj.astype(int), axis=1, keepdims=False).mode.ravel()

    def predict_proba(self, X):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        if self.scaling and self._scaler is not None:
            X = self._scaler.transform(X)
        if not self._is_binary:
            return self._strategy.predict_proba(X)
        preds = np.column_stack([self._predict_rep(X, r) for r in self._replications])
        preds = ((preds + 1) // 2).astype(int)
        maj, cnt = mode(preds, axis=1, keepdims=False)
        cnt = cnt.ravel()
        maj = maj.ravel()
        proba = np.empty((len(X), 2), float)
        proba[:, 0] = np.where(maj == 0, cnt, self.n_replications - cnt)
        proba[:, 1] = self.n_replications - proba[:, 0]
        return proba / self.n_replications

    def _predict_rep(self, X, rep: _Replication):
        Z, _ = self._design_matrix(X, rep.local_models)
        if rep.scaler is not None:
            Z = rep.scaler.transform(Z)
        if rep.global_model:
            return rep.global_model.predict(Z)
        return np.where(Z.sum(1) < 0, -1, 1)

    def _make_strategy(self):
        return {
            "ovo": OneVsOneClassifier(self),
            "occ": OutputCodeClassifier(self),
        }.get(self.multiclass, OneVsRestClassifier(self))

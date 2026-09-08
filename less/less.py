from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
from joblib import effective_n_jobs
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from threadpoolctl import threadpool_limits
from xgboost import DMatrix
from xgboost import train as xgb_train

from ._utils import (
    LocalModel,
    _adjust_dynamic_parameters,
    _validate_static_hyperparameters,
)

INTERNAL_DTYPE = np.float32


def _solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    r"""
    Ridge coefficients for a column-centered *X* and a centered *y*.

    Uses the primal normal equations when ``n_samples >= n_features`` and the
    dual (kernel) form otherwise, which is the same choice scikit-learn's Ridge
    makes internally. The Gram matrix is accumulated by BLAS in the input dtype
    and promoted to float64 before the factorization, so the O(n*d^2) part stays
    cheap while the solve keeps full precision.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Column-centered design matrix.
    y : np.ndarray of shape (n_samples,)
        Centered targets.
    alpha : float
        Regularization strength.

    Returns
    -------
    np.ndarray of shape (n_features,)
        The fitted coefficients, in float64.
    """
    n_samples, n_features = X.shape

    if n_samples >= n_features:
        gram = np.asarray(X.T @ X, dtype=np.float64)
        gram.flat[:: n_features + 1] += alpha
        rhs = np.asarray(X.T @ y, dtype=np.float64)
        try:
            return cho_solve(
                cho_factor(gram, lower=True, check_finite=False),
                rhs,
                check_finite=False,
            )
        except LinAlgError:
            return np.linalg.lstsq(gram, rhs, rcond=None)[0]

    # n_samples < n_features: solving the (n x n) kernel system is cheaper and,
    # unlike the Gram matrix, it is not rank deficient.
    kernel = np.asarray(X @ X.T, dtype=np.float64)
    kernel.flat[:: n_samples + 1] += alpha
    target = np.asarray(y, dtype=np.float64)
    try:
        dual = cho_solve(
            cho_factor(kernel, lower=True, check_finite=False),
            target,
            check_finite=False,
        )
    except LinAlgError:
        dual = np.linalg.lstsq(kernel, target, rcond=None)[0]
    return X.T @ dual


class _ClosedFormRidge:
    """Ridge regressor solved in closed form, exposing ``coef_``/``intercept_``.

    Numerically equivalent to ``Ridge(alpha=alpha)`` but skips scikit-learn's
    input validation, re-centering and condition-number estimation, which
    together dominate the runtime at the subset sizes LESS trains on.
    """

    __slots__ = ("alpha", "coef_", "intercept_")

    def __init__(self, alpha: float = 1e-6):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ClosedFormRidge:
        x_offset = X.mean(axis=0)
        y_offset = float(np.mean(y, dtype=np.float64))
        coef = _solve_ridge(X - x_offset, y - X.dtype.type(y_offset), self.alpha)
        self.coef_ = coef
        self.intercept_ = y_offset - float(coef @ x_offset)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted")
        return X @ self.coef_ + self.intercept_


class _NativeXGBoostRegressor:
    """Lightweight sklearn-compatible wrapper around xgboost.train."""

    __slots__ = ("_booster", "num_boost_round", "params")

    def __init__(self, params: dict[str, Any], num_boost_round: int = 1):
        self.params = params
        self.num_boost_round = num_boost_round
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NativeXGBoostRegressor:
        dtrain = DMatrix(X, label=y)
        self._booster = xgb_train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
        )
        # Training runs one thread per model (models are fitted in parallel), but
        # prediction happens serially in the caller, so give it the whole machine.
        self._booster.set_param({"nthread": 0})
        return self

    def predict(self, X: np.ndarray | DMatrix) -> np.ndarray:
        if self._booster is None:
            raise ValueError("Model is not fitted")
        if isinstance(X, np.ndarray) and X.flags.c_contiguous:
            # Skips DMatrix construction entirely; bitwise-identical output.
            return self._booster.inplace_predict(X)
        dmatrix = X if isinstance(X, DMatrix) else DMatrix(X)
        return self._booster.predict(dmatrix)


class _NativeXGBoostForest:
    """A random forest grown as several single-threaded boosters in parallel.

    XGBoost grows the trees of a ``num_parallel_tree`` forest one after another
    and only parallelizes *inside* a tree. On the narrow ``(n_samples,
    n_subsets)`` matrix LESS feeds the global estimator there is too little work
    per node for that to pay off: measured speedup stays around 1.5x no matter
    how many cores or trees are involved.

    The trees of a forest are independent, so this splits them across several
    boosters that each run on one thread and share a single already-binned
    ``DMatrix``, which uses the tree dimension for parallelism instead. The
    first chunk is grown on all threads: that call is what builds the shared
    gradient index, and its trees are kept rather than thrown away.

    Predictions are the tree-count weighted mean of the chunks, which is what
    the single-booster forest computes internally.
    """

    __slots__ = ("_boosters", "_weights", "n_jobs", "num_boost_round", "params")

    # How many chunks a forest is cut into. Deliberately a constant rather than
    # a function of n_jobs: XGBoost's output does not depend on ``nthread``, so
    # a fixed layout keeps the fitted forest identical whatever the thread
    # budget, and n_jobs only decides how fast the chunks are grown. Measured
    # sweet spot on a 10-core machine; 4 chunks left ~25% on the table and 10
    # was no better than 8.
    _N_CHUNKS = 8

    # Below this many trees the thread hand-off costs more than it saves.
    _MIN_TREES_TO_SPLIT = 5

    def __init__(
        self,
        params: dict[str, Any],
        num_boost_round: int = 1,
        n_jobs: int = -1,
    ):
        self.params = params
        self.num_boost_round = num_boost_round
        self.n_jobs = n_jobs
        self._boosters = None
        self._weights = None

    @classmethod
    def _chunk_sizes(cls, n_trees: int) -> list[int]:
        """Cut *n_trees* into a one-tree warm-up chunk plus even parallel chunks.

        The first chunk is grown before the others because it is the call that
        builds the shared gradient index, so it cannot overlap with anything.
        That makes it the serial part of the fit, and it only gets XGBoost's own
        ~1.5x from the extra threads, so it is kept as small as possible: one
        tree, just enough to trigger the binning.
        """
        if n_trees < cls._MIN_TREES_TO_SPLIT:
            return [n_trees]
        rest = n_trees - 1
        n_chunks = min(cls._N_CHUNKS, rest)
        base, extra = divmod(rest, n_chunks)
        return [1] + [base + (1 if i < extra else 0) for i in range(n_chunks)]

    def _train_chunk(
        self, dtrain: DMatrix, n_trees: int, nthread: int, seed_offset: int
    ) -> Any:
        params = {
            **self.params,
            "num_parallel_tree": n_trees,
            "nthread": nthread,
            "seed": self.params.get("seed", 0) + seed_offset,
        }
        booster = xgb_train(
            params=params, dtrain=dtrain, num_boost_round=self.num_boost_round
        )
        booster.set_param({"nthread": 0})
        return booster

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NativeXGBoostForest:
        chunks = self._chunk_sizes(int(self.params.get("num_parallel_tree", 1)))
        dtrain = DMatrix(X, label=y)

        # The first chunk runs on every thread and, as a side effect, builds the
        # gradient index the remaining chunks then only read.
        boosters = [self._train_chunk(dtrain, chunks[0], 0, 0)]
        rest = list(enumerate(chunks[1:], start=1))

        if rest:
            n_workers = min(effective_n_jobs(self.n_jobs), len(rest))
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    boosters.extend(
                        pool.map(
                            lambda item: self._train_chunk(dtrain, item[1], 1, item[0]),
                            rest,
                        )
                    )
            else:
                # One worker: same chunks, same trees, just grown one after the
                # other with every thread each.
                boosters.extend(
                    self._train_chunk(dtrain, n_trees, 0, offset)
                    for offset, n_trees in rest
                )

        self._boosters = boosters
        weights = np.asarray(chunks, dtype=np.float64)
        self._weights = weights / weights.sum()
        return self

    def predict(self, X: np.ndarray | DMatrix) -> np.ndarray:
        if self._boosters is None:
            raise ValueError("Model is not fitted")
        if isinstance(X, np.ndarray) and X.flags.c_contiguous:
            parts = (booster.inplace_predict(X) for booster in self._boosters)
        else:
            dmatrix = X if isinstance(X, DMatrix) else DMatrix(X)
            parts = (booster.predict(dmatrix) for booster in self._boosters)

        out = None
        for weight, part in zip(self._weights, parts):
            if out is None:
                out = part * INTERNAL_DTYPE(weight)
            else:
                out += part * INTERNAL_DTYPE(weight)
        return out


class BaseLESSRegressor(BaseEstimator, RegressorMixin):
    r"""
    Base class for LESS (Learning with Subset Stacking) Regressors.

    This base class provides common functionality for both boosting
    and averaging variants of the LESS algorithm.

    Parameters
    ----------
    n_subsets : int, default=20
        Number of local subsets to create for training. Must be positive.
        Each subset focuses on a different region of the feature space.
    local_estimator : str or callable, default='linear'
        The local estimator used to model each data subset. Can be a string
        identifying a built-in model ('linear', 'tree') or a callable that
        returns a scikit-learn compatible regressor instance.
    global_estimator : str or callable or None, default='xgboost'
        The global meta-estimator that combines the predictions of local models.
        Can be a string ('xgboost', implemented as a native XGBoost random forest),
        None (for simple averaging), or a callable that returns a scikit-learn
        compatible regressor.
    cluster_method : str or callable, default='tree'
        How the subsets are formed. 'tree' draws `n_subsets` random anchor
        points and gives each one its `n_neighbors` nearest samples, so the
        subsets are equally sized and may overlap. 'kmeans' and 'spectral'
        instead partition the data and use the clusters themselves as the
        subsets, which therefore vary in size, are mutually exclusive, and
        cover every sample; `n_subsets` is then the requested number of
        clusters and `min_neighbors` does not apply. A callable taking an
        `n_clusters` keyword and exposing `labels_` after `fit` can be given
        for custom clustering.
    val_size : float, optional
        The proportion of the dataset to reserve for training the global
        estimator. If specified, the data is split into a local learning set
        and a global learning set. Must be between 0 and 1.
    kernel_coeff : float or None, default=0.1
        The coefficient for the RBF kernel used to calculate distance-based weights.
        If None, the coefficient is dynamically set to `1.0 / (n_subsets**2)`
        for compatibility with the original LESS implementation. A larger `n_subsets`
        will result in a smaller, more localized kernel.
        If a float is provided, it is used as a fixed coefficient. Higher values
        lead to more localized influence.
    min_neighbors : int, default=10
        The minimum number of neighbors for each local subset. This ensures
        that each local model is trained on a sufficient number of samples.
    random_state : int or np.random.RandomState, optional
        Controls the randomness for reproducibility. Can be an integer for
        a new RandomState, or an existing RandomState object.

    Attributes
    ----------
    :attr:`n_features_in_` : int
        The number of features seen during :meth:`fit`.
    :attr:`feature_names_in_` : np.ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        local_estimator: str | Callable[[], Any] = "linear",
        global_estimator: str | Callable[[], Any] | None = "xgboost",
        cluster_method: str | Callable[..., Any] = "tree",
        val_size: float | None = None,
        kernel_coeff: float | None = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_subsets = n_subsets
        self.local_estimator = local_estimator
        self.global_estimator = global_estimator
        self.cluster_method = cluster_method
        self.val_size = val_size
        self.kernel_coeff = kernel_coeff
        self.min_neighbors = min_neighbors
        self.local_n_jobs = local_n_jobs
        self.random_state = random_state

        _validate_static_hyperparameters(self)

        self._native_global_xgb_rf_base_params = (
            self._build_native_global_xgboost_rf_base_params()
        )

        # Initialize random generator. Re-seeded at every fit, so refitting the
        # same instance with an integer seed reproduces the same model.
        self._rng = check_random_state(self.random_state)

    def _get_local_estimator_factory(self) -> Callable[[], Any]:
        """Get the factory function for creating local estimator instances."""
        if self.local_estimator == "linear":
            return lambda: _ClosedFormRidge(alpha=1e-6)
        elif self.local_estimator == "tree":
            return lambda: _NativeXGBoostRegressor(
                params={
                    "tree_method": "hist",
                    "grow_policy": "lossguide",
                    "max_leaves": 31,
                    "max_depth": 0,
                    "objective": "reg:squarederror",
                    "learning_rate": 1.0,
                    "gamma": 0.0,
                    "min_child_weight": 2.0,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 0.0,
                    "reg_alpha": 0.0,
                    "nthread": 1,
                    "verbosity": 0,
                    "seed": self._rng.randint(2**31),
                },
            )
        elif callable(self.local_estimator):
            return self.local_estimator
        else:
            raise ValueError(f"Invalid local_estimator: {self.local_estimator}")

    def _get_global_estimator_factory(self) -> Callable[[], Any] | None:
        """Get the factory function for creating the global estimator instance."""
        if self.global_estimator == "xgboost":
            base = self._native_global_xgb_rf_base_params
            return lambda: _NativeXGBoostForest(
                params={**base, "seed": self._rng.randint(2**31)},
                n_jobs=self.local_n_jobs,
            )
        elif self.global_estimator is None:
            return None
        elif callable(self.global_estimator):
            return self.global_estimator
        else:
            raise ValueError(f"Invalid global_estimator: {self.global_estimator}")

    def _build_native_global_xgboost_rf_base_params(self) -> dict[str, Any]:
        """Build native XGBoost params for standalone random forest training."""
        return {
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "learning_rate": 1.0,
            "num_parallel_tree": 25,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "reg_lambda": 1e-5,
            "verbosity": 0,
        }

    def _safe_normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        r"""
        Safely normalize distance weights in-place to avoid extra allocations.

        Parameters
        ----------
        distances : np.ndarray of shape (n_samples, n_subsets)
            The raw distance weights calculated from the RBF kernel.
            **Modified in-place** and returned.

        Returns
        -------
        np.ndarray of shape (n_samples, n_subsets)
            The normalized distance weights, where each row sums to 1.
        """
        if distances.shape[0] == 0:
            return distances

        # A matrix-vector product against ones beats np.sum(axis=1) by ~8x
        # here: BLAS runs at memory bandwidth, numpy's reduction does not.
        ones = np.ones(distances.shape[1], dtype=distances.dtype)
        distance_sums = np.dot(distances, ones).reshape(-1, 1)

        zero_mask = distance_sums.flatten() < 1e-12
        if np.any(zero_mask):
            uniform_weight = 1.0 / distances.shape[1]
            distances[zero_mask] = uniform_weight
            distance_sums[zero_mask] = 1.0

        np.divide(distances, distance_sums, out=distances)
        return distances

    def _get_kernel_coeff(self, n_subsets: int) -> float:
        """Resolve the effective kernel coefficient for the given subset count."""
        if self.kernel_coeff is None:
            return 1.0 / (n_subsets**2) if n_subsets > 0 else 1.0
        return self.kernel_coeff

    def _compute_distance_matrix(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        kernel_coeff: float,
        x_sq_norms: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute all RBF distances in one vectorized pass (in-place)."""
        if X.shape[0] == 0 or centers.shape[0] == 0:
            return np.zeros((X.shape[0], centers.shape[0]), dtype=X.dtype)

        if x_sq_norms is None:
            x_sq_norms = np.einsum("ij,ij->i", X, X)

        x_sq_col = np.asarray(x_sq_norms, dtype=X.dtype).reshape(-1, 1)
        center_sq_row = np.einsum("ij,ij->i", centers, centers)[np.newaxis, :]

        # Build squared distances in a single buffer
        dist = np.dot(X, centers.T)  # (n_samples, n_subsets)
        dist *= -2.0
        dist += x_sq_col
        dist += center_sq_row
        # Squared distances are non-negative by construction; only cancellation
        # noise can push one below zero, and for those |d| is just as close to
        # the true zero as clipping is. np.abs is ~6x faster than np.maximum,
        # which numpy does not vectorize here.
        np.abs(dist, out=dist)
        np.sqrt(dist, out=dist)
        dist *= -kernel_coeff
        np.exp(dist, out=dist)
        return self._safe_normalize_distances(dist)

    def _find_neighbor_indices(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        n_neighbors: int,
        x_sq_norms: np.ndarray | None = None,
    ) -> np.ndarray:
        """Find exact nearest neighbors with brute-force top-k (in-place)."""
        n_samples = X.shape[0]
        if centers.shape[0] == 0 or n_neighbors == 0:
            return np.zeros((centers.shape[0], 0), dtype=np.intp)

        if n_neighbors >= n_samples:
            return np.broadcast_to(
                np.arange(n_samples, dtype=np.intp),
                (centers.shape[0], n_samples),
            ).copy()

        if x_sq_norms is None:
            x_sq_norms = np.einsum("ij,ij->i", X, X)

        x_sq_norms = np.asarray(x_sq_norms, dtype=X.dtype)

        # Rank keys, not distances: only the order within each row matters, so
        # the per-centre ||c||^2 term (constant along a row) and the clip at
        # zero are both dropped. Adding a constant to a row is strictly
        # monotone, so the selected neighbours are exactly the same.
        sq_dist = np.dot(centers, X.T)  # (n_subsets, n_samples)
        sq_dist *= -2.0
        sq_dist += x_sq_norms[np.newaxis, :]

        # One introselect per centre; the rows are independent, and numpy's
        # axis=1 form runs them one after another on a single thread.
        kth = n_neighbors - 1
        rows = self._map_workers(
            lambda row: np.argpartition(row, kth=kth)[:n_neighbors], sq_dist
        )
        return np.stack(rows)

    def _predict_local_outputs(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        linear_coefs: np.ndarray | None = None,
        linear_intercepts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict local model outputs, using a single matmul for linear models."""
        if not local_models:
            return np.zeros((X.shape[0], 0), dtype=X.dtype)

        # Fast path: cached linear coefficients (in-place add)
        if linear_coefs is not None and linear_intercepts is not None:
            out = np.dot(X, linear_coefs.T)
            out += linear_intercepts
            return out.astype(INTERNAL_DTYPE, copy=False)

        # Fallback: extract coef_/intercept_ on the fly
        if all(
            hasattr(local_model.estimator, "coef_")
            and hasattr(local_model.estimator, "intercept_")
            for local_model in local_models
        ):
            try:
                coefs = np.vstack(
                    [
                        np.ravel(local_model.estimator.coef_)
                        for local_model in local_models
                    ]
                )
                intercepts = np.array(
                    [
                        np.asarray(local_model.estimator.intercept_).reshape(-1)[0]
                        for local_model in local_models
                    ],
                    dtype=coefs.dtype,
                )
                out = np.dot(X, coefs.T)
                out += intercepts
                return out.astype(INTERNAL_DTYPE, copy=False)
            except (AttributeError, IndexError, TypeError, ValueError):
                # Estimators whose coef_/intercept_ are not plain 1-D arrays
                # simply fall through to the per-model prediction path below.
                pass

        # Generic path: pre-allocate output, fill columns (no list+column_stack)
        n_models = len(local_models)
        out = np.empty((X.shape[0], n_models), dtype=INTERNAL_DTYPE)
        for i, local_model in enumerate(local_models):
            try:
                out[:, i] = local_model.estimator.predict(X)
            except Exception as e:
                raise RuntimeError(
                    f"Error predicting with local model {i}: {e!s}"
                ) from e

        return out

    def _get_linear_prediction_params(
        self, local_models: list[LocalModel]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract coefficient caches for linear-compatible local estimators."""
        if not local_models:
            return None, None

        if not all(
            hasattr(local_model.estimator, "coef_")
            and hasattr(local_model.estimator, "intercept_")
            for local_model in local_models
        ):
            return None, None

        try:
            linear_coefs = np.vstack(
                [np.ravel(local_model.estimator.coef_) for local_model in local_models]
            ).astype(INTERNAL_DTYPE, copy=False)
            linear_intercepts = np.array(
                [
                    np.asarray(local_model.estimator.intercept_).reshape(-1)[0]
                    for local_model in local_models
                ],
                dtype=INTERNAL_DTYPE,
            )
            return linear_coefs, linear_intercepts
        except (AttributeError, IndexError, TypeError, ValueError):
            # Not linear-compatible after all; callers fall back to predict().
            return None, None

    def _fit_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        r"""
        Cluster *X* with the configured clustering method and return its labels.

        Parameters
        ----------
        X : np.ndarray
            The input data to cluster.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The cluster label of every sample.

        Raises
        ------
        ValueError
            If `cluster_method` is not a recognized string or a callable.
        RuntimeError
            If the clustering process fails.
        """
        if callable(self.cluster_method):
            # Use custom clustering method
            clusterer = self.cluster_method(n_clusters=self._n_subsets_adjusted)
        elif self.cluster_method == "kmeans":
            # New seed per call, for diversity across iterations
            clusterer = KMeans(
                n_clusters=self._n_subsets_adjusted,
                random_state=self._rng.randint(2**31),
            )
        elif self.cluster_method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=self._n_subsets_adjusted,
                random_state=self._rng.randint(2**31),
            )
        else:
            raise ValueError(f"Invalid cluster_method: {self.cluster_method}")

        try:
            clusterer.fit(X)
            # Only the labels are needed: each cluster *is* a subset, and its
            # center is the centroid of its own members. Requiring
            # 'cluster_centers_' instead would rule out every clusterer that
            # does not expose it, SpectralClustering among them.
            labels = np.asarray(clusterer.labels_)
        except Exception as e:
            raise RuntimeError(f"Error during clustering: {e!s}") from e

        if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"Clustering returned labels of shape {labels.shape} for "
                f"{X.shape[0]} samples"
            )

        return labels

    @staticmethod
    def _subsets_from_labels(labels: np.ndarray) -> list[np.ndarray]:
        """Group sample indices by cluster label, one index array per cluster.

        Sorting once and cutting at the label boundaries touches the labels
        twice in total, where a boolean mask per label would walk the whole
        array once for every cluster.
        """
        order = np.argsort(labels, kind="stable").astype(np.intp, copy=False)
        boundaries = np.flatnonzero(np.diff(labels[order])) + 1
        return [subset for subset in np.split(order, boundaries) if subset.size > 0]

    def _get_subset_indices(
        self, X: np.ndarray, x_sq_norms: np.ndarray | None = None
    ) -> np.ndarray | list[np.ndarray]:
        r"""
        Build the sample subsets for one stage.

        With ``cluster_method='tree'`` the subsets are the anchor
        neighborhoods of the manuscript: `n_subsets` input points are drawn at
        random, and each anchor takes its `n_neighbors` nearest samples. Those
        subsets are equally sized and may overlap.

        A clustering method instead *partitions* the data: the clusters
        themselves are the subsets, so they vary in size, are mutually
        exclusive, and together cover every sample.

        Parameters
        ----------
        X : np.ndarray
            The input data the subsets are drawn from.
        x_sq_norms : np.ndarray, optional
            Precomputed squared row norms of *X*, for the anchor search.

        Returns
        -------
        np.ndarray of shape (n_subsets, n_neighbors) or list[np.ndarray]
            Sample indices per subset: a 2-D array when every subset has the
            same size (the anchor case), a list of index arrays otherwise.
        """
        if self.cluster_method == "tree":
            # Randomly select subset anchors
            anchor_indices = self._rng.choice(
                X.shape[0], size=self._n_subsets_adjusted, replace=False
            )
            # High-dimensional data benefits more from brute-force top-k than
            # tree search.
            return self._find_neighbor_indices(
                X,
                X[anchor_indices],
                self._n_neighbors,
                x_sq_norms=x_sq_norms,
            )

        return self._subsets_from_labels(self._fit_cluster_labels(X))

    @contextmanager
    def _worker_pool(self) -> Iterator[None]:
        """Hold one thread pool open for the whole fit.

        Every stage dispatches the same handful of short tasks, and standing a
        pool up per stage costs more than the tasks themselves: the per-call
        dispatch loop was ~5% of fit time on the profiles.
        """
        n_workers = effective_n_jobs(self.local_n_jobs)
        pool = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
        self._pool = pool
        try:
            yield
        finally:
            self._pool = None
            if pool is not None:
                pool.shutdown(wait=True)

    def _map_workers(self, fn: Callable[..., Any], items: Any) -> list[Any]:
        """Run *fn* over *items*, on the fit-scoped pool when there is one."""
        pool = getattr(self, "_pool", None)
        if pool is None:
            return [fn(item) for item in items]
        return list(pool.map(fn, items))

    def _get_x_sq_norms(self, X: np.ndarray) -> np.ndarray:
        """Squared row norms of *X*, reused for as long as *X* is the same array.

        Every stage needs these for both the neighbour search and the RBF
        weights, and boosting hands the same feature matrix to every stage, so
        recomputing them is a full pass over X thrown away once per stage.
        """
        # Identity, not equality; holding the array also keeps a freed one from
        # being mistaken for a new one at the same address.
        cached = getattr(self, "_x_sq_norms_cache", None)
        if cached is not None and cached[0] is X:
            return cached[1]
        norms = np.einsum("ij,ij->i", X, X)
        self._x_sq_norms_cache = (X, norms)
        return norms

    def _build_local_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prediction_data: np.ndarray | None = None,
    ) -> tuple[list[LocalModel], np.ndarray, np.ndarray | None]:
        r"""
        Build local models for one stage of the algorithm.

        Returns
        -------
        tuple[list[LocalModel], np.ndarray, Optional[np.ndarray]]
            - A list of trained `LocalModel` instances.
            - The center matrix of local models.
            - The weighted feature matrix Z (if *prediction_data* is given).
        """
        x_sq_norms = self._get_x_sq_norms(X)
        subset_indices = self._get_subset_indices(X, x_sq_norms=x_sq_norms)

        local_models = []
        local_centers = []

        local_estimators = [
            self._local_estimator_factory() for _ in range(len(subset_indices))
        ]

        # Clustering partitions the data into subsets of differing size, so the
        # shared scratch buffers only apply to equally sized anchor subsets.
        uniform_subsets = (
            isinstance(subset_indices, np.ndarray) and subset_indices.ndim == 2
        )
        n_neighbors = subset_indices.shape[1] if uniform_subsets else 0
        n_features = X.shape[1]
        use_buffers = uniform_subsets and n_neighbors > 0

        # Train local models
        if self.local_n_jobs == 1 or len(subset_indices) <= 1:
            # Single scratch buffer reused across all subsets
            X_buf = (
                np.empty((n_neighbors, n_features), dtype=X.dtype)
                if use_buffers
                else None
            )
            y_buf = np.empty(n_neighbors, dtype=y.dtype) if use_buffers else None
            results = [
                self._fit_single_local_model(
                    local_est, X, y, neighbors, i, X_buf, y_buf
                )
                for i, (local_est, neighbors) in enumerate(
                    zip(local_estimators, subset_indices)
                )
            ]
        else:
            import threading

            _tls = threading.local()

            def _fit_with_buf(item):
                i, local_est, neighbors = item
                if not use_buffers:
                    return self._fit_single_local_model(local_est, X, y, neighbors, i)
                if not hasattr(_tls, "X_buf"):
                    _tls.X_buf = np.empty((n_neighbors, n_features), dtype=X.dtype)
                    _tls.y_buf = np.empty(n_neighbors, dtype=y.dtype)
                return self._fit_single_local_model(
                    local_est,
                    X,
                    y,
                    neighbors,
                    i,
                    _tls.X_buf,
                    _tls.y_buf,
                )

            work = [
                (i, local_est, neighbors)
                for i, (local_est, neighbors) in enumerate(
                    zip(local_estimators, subset_indices)
                )
            ]
            with threadpool_limits(limits=1):
                results = self._map_workers(_fit_with_buf, work)

        for local_model, center in results:
            local_models.append(local_model)
            local_centers.append(center)

        if local_centers:
            center_matrix = np.vstack(local_centers).astype(INTERNAL_DTYPE, copy=False)
        else:
            center_matrix = np.zeros((0, X.shape[1]), dtype=INTERNAL_DTYPE)

        if prediction_data is None:
            return local_models, center_matrix, None

        linear_coefs, linear_intercepts = self._get_linear_prediction_params(
            local_models
        )
        prediction_x_sq_norms = x_sq_norms if prediction_data is X else None
        Z = self._compute_weighted_features(
            prediction_data,
            local_models,
            center_matrix=center_matrix,
            x_sq_norms=prediction_x_sq_norms,
            linear_coefs=linear_coefs,
            linear_intercepts=linear_intercepts,
        )

        return local_models, center_matrix, Z

    def _fit_single_local_model(
        self,
        local_estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        neighbors: np.ndarray,
        index: int,
        X_buf: np.ndarray | None = None,
        y_buf: np.ndarray | None = None,
    ) -> tuple[LocalModel, np.ndarray]:
        """Fit a single local estimator for a subset.

        When *X_buf* / *y_buf* scratch buffers are provided the subset is
        written into them via ``np.take`` instead of allocating a fresh array
        through fancy indexing.
        """
        try:
            if X_buf is not None and y_buf is not None:
                np.take(X, neighbors, axis=0, out=X_buf)
                np.take(y, neighbors, axis=0, out=y_buf)
                X_local = X_buf
                y_local = y_buf
            else:
                X_local = X[neighbors]
                y_local = y[neighbors]
            center = np.mean(X_local, axis=0)
            if isinstance(local_estimator, (_ClosedFormRidge, LinearRegression, Ridge)):
                # Scale in-place to fix conditioning; un-scale coefs after fit.
                X_local -= center
                # Column standard deviations read straight off the values we
                # just centered: one pass and no temporary, where np.std would
                # re-derive the mean and materialise the squared deviations.
                std = np.sqrt(
                    np.einsum("ij,ij->j", X_local, X_local)
                    / X_local.dtype.type(X_local.shape[0])
                )
                # Mask for constant / near-constant features:
                # leave them unscaled (std=1) so coef stays 0 after fit.
                safe = std > 1e-7
                std[~safe] = 1.0
                X_local /= std
                if isinstance(local_estimator, _ClosedFormRidge):
                    # X_local is already centered, so solve on it directly and
                    # skip the float64 copy sklearn would force.
                    y_offset = np.mean(y_local, dtype=np.float64)
                    coef = _solve_ridge(
                        X_local,
                        y_local - y_local.dtype.type(y_offset),
                        local_estimator.alpha,
                    )
                    intercept = y_offset
                else:
                    local_estimator.fit(
                        np.asarray(X_local, dtype=np.float64),
                        np.asarray(y_local, dtype=np.float64),
                    )
                    coef = np.asarray(local_estimator.coef_).ravel()
                    intercept = local_estimator.intercept_
                # Un-scale only the features that were actually scaled.
                coef[safe] /= std[safe]
                coef[~safe] = 0.0
                local_estimator.coef_ = coef
                local_estimator.intercept_ = intercept - coef @ center
            else:
                local_estimator.fit(X_local, y_local)
            return LocalModel(local_estimator, center), center
        except Exception as e:
            raise RuntimeError(f"Error training local model {index}: {e!s}") from e

    def _compute_weighted_features(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        center_matrix: np.ndarray | None = None,
        x_sq_norms: np.ndarray | None = None,
        linear_coefs: np.ndarray | None = None,
        linear_intercepts: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Compute Z = distances * local_preds in one pass.

        Returns the weighted feature matrix directly, avoiding the need
        for callers to hold both local_preds and distances simultaneously.

        Returns
        -------
        np.ndarray of shape (n_samples, n_subsets)
            The weighted feature matrix Z.
        """
        if center_matrix is None and local_models:
            center_matrix = np.vstack(
                [local_model.center for local_model in local_models]
            )
        elif center_matrix is None:
            center_matrix = np.zeros((0, X.shape[1]), dtype=X.dtype)

        kernel_coeff = self._get_kernel_coeff(len(local_models))

        local_preds = self._predict_local_outputs(
            X,
            local_models,
            linear_coefs=linear_coefs,
            linear_intercepts=linear_intercepts,
        )
        distances = self._compute_distance_matrix(
            X,
            center_matrix,
            kernel_coeff,
            x_sq_norms=x_sq_norms,
        )

        # Fuse into Z in-place, freeing local_preds at scope exit
        np.multiply(distances, local_preds, out=distances)
        return distances

    def _store_sklearn_attributes(self, X: np.ndarray) -> None:
        r"""
        Store attributes required by scikit-learn.

        Parameters
        ----------
        X : np.ndarray
            The input data from which to infer attributes.
        """
        _, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

    def _validate_prediction_input(self, X: np.ndarray) -> np.ndarray:
        r"""
        Validate the input data for prediction.

        Parameters
        ----------
        X : np.ndarray
            The input features for prediction.

        Returns
        -------
        np.ndarray
            The validated and converted `X` array.
        """
        # Check if fitted
        check_is_fitted(self)

        # Validate input
        X = check_array(
            X,
            accept_sparse=False,
            dtype=INTERNAL_DTYPE,
            order="C",
            ensure_min_samples=0,
        )

        # Check feature count consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but regressor "
                f"is expecting {self.n_features_in_} features as seen in fit."
            )

        return X

    def _prepare_fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Prepare for fitting by validating data and setting up estimators.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target values.
        sample_weight : np.ndarray, optional
            Sample weights.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the validated X and y.
        """
        _validate_static_hyperparameters(self)
        self._rng = check_random_state(self.random_state)
        self._x_sq_norms_cache = None

        # Validate and prepare data
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=False,
            dtype=INTERNAL_DTYPE,
            order="C",
        )
        y = np.asarray(y, dtype=INTERNAL_DTYPE)

        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not currently supported and will be ignored",
                UserWarning,
                stacklevel=2,
            )

        self._store_sklearn_attributes(X)

        if self.val_size is not None and not (0 < self.val_size < 1):
            raise ValueError("val_size must be a float between 0 and 1.")

        # Validate and adjust parameters based on training data
        self._n_subsets_adjusted, self._n_neighbors = _adjust_dynamic_parameters(
            self, X.shape[0]
        )

        # Initialize estimator factories
        self._local_estimator_factory = self._get_local_estimator_factory()
        self._global_estimator_factory = self._get_global_estimator_factory()

        return X, y

    def fit(self, X, y, sample_weight=None):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")


class LESSBRegressor(BaseLESSRegressor):
    r"""
    LESSB (Learning with Subset Stacking Boosting) Regressor.

    This regressor implements the boosting variant of the LESS algorithm.
    It iteratively fits stages, where each stage consists of a set of local
    models that predict the residuals of the previous stage.

    Parameters
    ----------
    n_subsets : int, default=20
        Number of local subsets to create for training.
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each stage.
    local_estimator : str or callable, default='linear'
        The local estimator for modeling data subsets.
    global_estimator : str or callable or None, default='xgboost'
        The global meta-estimator for combining local model predictions.
        The built-in 'xgboost' option uses a native XGBoost random forest.
    cluster_method : str or callable, default='tree'
        How the subsets are formed: 'tree' uses random anchors with their
        nearest neighbors, while clustering methods use the clusters
        themselves as the subsets.
    val_size : float, optional
        The proportion of the dataset to reserve for the global estimator.
    kernel_coeff : float or None, default=0.1
        The RBF kernel coefficient for distance weighting.
    min_neighbors : int, default=10
        The minimum number of neighbors for each local subset.
    early_stopping_tolerance : float, default=1e-8
        Tolerance for early stopping based on residual improvement.
    random_state : int or np.random.RandomState, optional
        Controls the randomness for reproducibility.

    Attributes
    ----------
    :attr:`n_features_in_` : int
        The number of features seen during :meth:`fit`.
    :attr:`feature_names_in_` : np.ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`.
    _local_models_stages : list[list[LocalModel]]
        A list containing the lists of local models for each boosting stage.
    _global_models_stages : list[Any]
        A list containing the global model for each boosting stage.
    _base_prediction : float
        The initial base prediction, typically the mean of the target values.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        local_estimator: str | Callable[[], Any] = "linear",
        global_estimator: str | Callable[[], Any] | None = "xgboost",
        cluster_method: str | Callable[..., Any] = "tree",
        val_size: float | None = None,
        kernel_coeff: float | None = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        early_stopping_tolerance: float = 1e-8,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            n_subsets=n_subsets,
            local_estimator=local_estimator,
            global_estimator=global_estimator,
            cluster_method=cluster_method,
            val_size=val_size,
            kernel_coeff=kernel_coeff,
            min_neighbors=min_neighbors,
            local_n_jobs=local_n_jobs,
            random_state=random_state,
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_tolerance = early_stopping_tolerance

        # The base constructor ran before these existed, so re-check them here.
        _validate_static_hyperparameters(self)

    def _reset_state(self) -> None:
        """Reset the internal state of the regressor for refitting."""
        self._local_models_stages = []
        self._local_center_matrices_stages = []
        self._local_linear_coefs_stages = []
        self._local_linear_intercepts_stages = []
        self._global_models_stages = []
        self._base_prediction = 0.0

    def _build_stage(
        self,
        local_models: list[LocalModel],
        center_matrix: np.ndarray,
        linear_coefs: np.ndarray | None,
        linear_intercepts: np.ndarray | None,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        cached_train_features: np.ndarray | None = None,
    ) -> Any | None:
        r"""
        Build the global model for a single boosting stage.

        Parameters
        ----------
        local_models : list[LocalModel]
            The trained local models for the current stage.
        X_train : np.ndarray
            The training features.
        y_train : np.ndarray
            The training targets (residuals).
        X_val : np.ndarray, optional
            The validation features.
        y_val : np.ndarray, optional
            The validation targets (residuals).

        Returns
        -------
        Optional[Any]
            The trained global model for the stage, or None.
        """
        global_est = None
        if self._global_estimator_factory is not None:
            # If validation set exists, train global model on its predictions
            if X_val is not None and y_val is not None:
                Z_global = self._compute_weighted_features(
                    X_val,
                    local_models,
                    center_matrix=center_matrix,
                    linear_coefs=linear_coefs,
                    linear_intercepts=linear_intercepts,
                )
                y_global = y_val
            else:
                if cached_train_features is not None:
                    Z_global = cached_train_features
                else:
                    Z_global = self._compute_weighted_features(
                        X_train,
                        local_models,
                        center_matrix=center_matrix,
                        linear_coefs=linear_coefs,
                        linear_intercepts=linear_intercepts,
                    )
                y_global = y_train

            try:
                global_est = self._global_estimator_factory()
                global_est.fit(Z_global, y_global)
            except Exception as e:
                raise RuntimeError(f"Error training global model: {e!s}") from e

        return global_est

    def _predict_stage(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        center_matrix: np.ndarray,
        linear_coefs: np.ndarray | None,
        linear_intercepts: np.ndarray | None,
        global_model: Any | None,
        x_sq_norms: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""
        Make predictions for a single boosting stage.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        local_models : list[LocalModel]
            The local models for the stage.
        global_model : any, optional
            The global model for the stage.

        Returns
        -------
        np.ndarray
            The predictions for the stage.
        """
        Z = self._compute_weighted_features(
            X,
            local_models,
            center_matrix=center_matrix,
            x_sq_norms=x_sq_norms,
            linear_coefs=linear_coefs,
            linear_intercepts=linear_intercepts,
        )

        if global_model is not None:
            return global_model.predict(Z)
        return np.sum(Z, axis=1)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> LESSBRegressor:
        r"""
        Fit the LESSB regressor using boosting.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Sample weights. Not currently used.

        Returns
        -------
        LESSBRegressor
            The fitted regressor.
        """
        self._reset_state()
        X, y = self._prepare_fit(X, y, sample_weight)

        self._base_prediction = np.mean(y, dtype=INTERNAL_DTYPE).astype(INTERNAL_DTYPE)
        if not np.isfinite(self._base_prediction):
            raise ValueError("Target values contain non-finite values")

        current_predictions = np.full(
            y.shape,
            self._base_prediction,
            dtype=INTERNAL_DTYPE,
        )
        learning_rate = INTERNAL_DTYPE(self.learning_rate)
        fit_x_sq_norms = self._get_x_sq_norms(X) if self.val_size is not None else None
        residuals = np.empty_like(y)

        with self._worker_pool():
            for stage in range(self.n_estimators):
                try:
                    np.subtract(y, current_predictions, out=residuals)

                    if self.val_size is not None:
                        X_train, X_val, residuals_train, residuals_val = (
                            train_test_split(
                                X,
                                residuals,
                                test_size=self.val_size,
                                random_state=self._rng,
                            )
                        )
                    else:
                        X_train, residuals_train = X, residuals
                        X_val, residuals_val = None, None

                    prediction_data = X_train if self.val_size is None else None
                    local_models, center_matrix, Z_stage = self._build_local_models(
                        X_train, residuals_train, prediction_data=prediction_data
                    )
                    linear_coefs, linear_intercepts = (
                        self._get_linear_prediction_params(local_models)
                    )

                    if self.val_size is None:
                        if Z_stage is None:
                            raise RuntimeError(
                                "Training predictions were not computed for the stage"
                            )
                        global_model = self._build_stage(
                            local_models,
                            center_matrix,
                            linear_coefs,
                            linear_intercepts,
                            X_train,
                            residuals_train,
                            None,
                            None,
                            cached_train_features=Z_stage,
                        )
                        if global_model is not None:
                            stage_predictions = global_model.predict(Z_stage)
                        else:
                            stage_predictions = np.sum(Z_stage, axis=1)
                    else:
                        global_model = self._build_stage(
                            local_models,
                            center_matrix,
                            linear_coefs,
                            linear_intercepts,
                            X_train,
                            residuals_train,
                            X_val,
                            residuals_val,
                        )
                        stage_predictions = self._predict_stage(
                            X,
                            local_models,
                            center_matrix,
                            linear_coefs,
                            linear_intercepts,
                            global_model,
                            x_sq_norms=fit_x_sq_norms,
                        )

                    if not np.all(np.isfinite(stage_predictions)):
                        warnings.warn(
                            f"Non-finite predictions in stage {stage}, skipping",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                    # In-place: avoids a temp from learning_rate * stage_predictions
                    stage_predictions *= learning_rate
                    current_predictions += stage_predictions

                    self._local_models_stages.append(local_models)
                    self._local_center_matrices_stages.append(center_matrix)
                    self._local_linear_coefs_stages.append(linear_coefs)
                    self._local_linear_intercepts_stages.append(linear_intercepts)
                    self._global_models_stages.append(global_model)

                    # Reuse pre-allocated residuals for early stopping check
                    np.subtract(y, current_predictions, out=residuals)
                    mean_abs_residual = np.mean(np.abs(residuals))
                    if mean_abs_residual < self.early_stopping_tolerance and stage > 0:
                        break

                except Exception as e:
                    warnings.warn(
                        f"Error in boosting stage {stage}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
                    if not self._local_models_stages:
                        raise RuntimeError(
                            "No boosting stages completed successfully"
                        ) from e
                    break

        if not self._local_models_stages:
            raise RuntimeError("No boosting stages completed successfully")

        return self

    def predict(self, X: np.ndarray, n_rounds: int | None = None) -> np.ndarray:
        r"""
        Predict using the fitted LESSB regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples to predict.
        n_rounds : int, optional
            The number of boosting stages to use for prediction. If None, all
            stages are used.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_prediction_input(X)

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=INTERNAL_DTYPE)

        # Determine number of rounds to use
        available_rounds = len(self._local_models_stages)
        if n_rounds is None:
            n_rounds = available_rounds
        else:
            if not isinstance(n_rounds, int) or n_rounds <= 0:
                raise ValueError(f"n_rounds must be a positive integer, got {n_rounds}")
            n_rounds = min(n_rounds, available_rounds)

        # Start with base prediction
        predictions = np.full(
            n_samples,
            self._base_prediction,
            dtype=INTERNAL_DTYPE,
        )
        learning_rate = INTERNAL_DTYPE(self.learning_rate)
        x_sq_norms = self._get_x_sq_norms(X)

        # Add predictions from specified number of stages
        for stage in range(n_rounds):
            try:
                local_models = self._local_models_stages[stage]
                center_matrix = self._local_center_matrices_stages[stage]
                linear_coefs = self._local_linear_coefs_stages[stage]
                linear_intercepts = self._local_linear_intercepts_stages[stage]
                global_model = self._global_models_stages[stage]
                stage_predictions = self._predict_stage(
                    X,
                    local_models,
                    center_matrix,
                    linear_coefs,
                    linear_intercepts,
                    global_model,
                    x_sq_norms=x_sq_norms,
                )

                if np.all(np.isfinite(stage_predictions)):
                    stage_predictions *= learning_rate
                    predictions += stage_predictions
                else:
                    warnings.warn(
                        f"Non-finite predictions in stage {stage}, skipping",
                        UserWarning,
                        stacklevel=2,
                    )

            except Exception as e:  # noqa: BLE001 - one bad stage must not kill predict
                warnings.warn(
                    f"Error in prediction stage {stage}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        return predictions


class LESSARegressor(BaseLESSRegressor):
    r"""
    LESSV (Learning with Subset Stacking Averaging) Regressor.

    This regressor implements the averaging variant of the LESS algorithm.
    It trains multiple iterations of local and global models and averages
    their predictions.

    Parameters
    ----------
    n_subsets : int, default=20
        Number of local subsets to create for training.
    n_estimators : int, default=100
        The number of averaging iterations to perform.
    local_estimator : str or callable, default='linear'
        The local estimator for modeling data subsets.
    global_estimator : str or callable or None, default='xgboost'
        The global meta-estimator for combining local model predictions.
        The built-in 'xgboost' option uses a native XGBoost random forest.
    cluster_method : str or callable, default='tree'
        How the subsets are formed: 'tree' uses random anchors with their
        nearest neighbors, while clustering methods use the clusters
        themselves as the subsets.
    val_size : float, optional
        The proportion of the dataset to reserve for the global estimator.
    kernel_coeff : float or None, default=0.1
        The RBF kernel coefficient for distance weighting.
    min_neighbors : int, default=10
        The minimum number of neighbors for each local subset.
    random_state : int or np.random.RandomState, optional
        Controls the randomness for reproducibility.

    Attributes
    ----------
    :attr:`n_features_in_` : int
        The number of features seen during :meth:`fit`.
    :attr:`feature_names_in_` : np.ndarray of shape (`n_features_in_`,)
        Names of features seen during :meth:`fit`.
    _local_models_iterations : list[list[LocalModel]]
        A list containing the lists of local models for each iteration.
    _global_models_iterations : list[Any]
        A list containing the global model for each iteration.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        n_estimators: int = 100,
        local_estimator: str | Callable[[], Any] = "linear",
        global_estimator: str | Callable[[], Any] | None = "xgboost",
        cluster_method: str | Callable[..., Any] = "tree",
        val_size: float | None = None,
        kernel_coeff: float | None = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(
            n_subsets=n_subsets,
            local_estimator=local_estimator,
            global_estimator=global_estimator,
            cluster_method=cluster_method,
            val_size=val_size,
            kernel_coeff=kernel_coeff,
            min_neighbors=min_neighbors,
            local_n_jobs=local_n_jobs,
            random_state=random_state,
        )

        self.n_estimators = n_estimators

        # The base constructor ran before this existed, so re-check it here.
        _validate_static_hyperparameters(self)

    def _reset_state(self) -> None:
        """Reset the internal state of the regressor for refitting."""
        self._local_models_iterations = []
        self._local_center_matrices_iterations = []
        self._local_linear_coefs_iterations = []
        self._local_linear_intercepts_iterations = []
        self._global_models_iterations = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> LESSARegressor:
        r"""
        Fit the LESSA regressor using model averaging.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Sample weights. Not currently used.

        Returns
        -------
        LESSARegressor
            The fitted regressor.
        """
        self._reset_state()
        X, y = self._prepare_fit(X, y, sample_weight)

        with self._worker_pool():
            for _ in range(self.n_estimators):
                try:
                    if self.val_size is not None:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size=self.val_size, random_state=self._rng
                        )
                    else:
                        X_train, y_train = X, y
                        X_val, y_val = None, None

                    prediction_data = (
                        X_train
                        if self.val_size is None
                        and self._global_estimator_factory is not None
                        else None
                    )
                    local_models, center_matrix, Z_train = self._build_local_models(
                        X_train, y_train, prediction_data=prediction_data
                    )
                    linear_coefs, linear_intercepts = (
                        self._get_linear_prediction_params(local_models)
                    )

                    global_est = None
                    if self._global_estimator_factory is not None:
                        if X_val is not None and y_val is not None:
                            Z_global = self._compute_weighted_features(
                                X_val,
                                local_models,
                                center_matrix=center_matrix,
                                linear_coefs=linear_coefs,
                                linear_intercepts=linear_intercepts,
                            )
                            y_global = y_val
                        else:
                            if Z_train is None:
                                raise RuntimeError(
                                    "Training predictions were not computed for "
                                    "the global estimator"
                                )
                            Z_global = Z_train
                            y_global = y_train

                        global_est = self._global_estimator_factory()
                        global_est.fit(Z_global, y_global)

                    self._local_models_iterations.append(local_models)
                    self._local_center_matrices_iterations.append(center_matrix)
                    self._local_linear_coefs_iterations.append(linear_coefs)
                    self._local_linear_intercepts_iterations.append(linear_intercepts)
                    self._global_models_iterations.append(global_est)

                except Exception as e:
                    warnings.warn(f"Error in iteration: {e}", UserWarning, stacklevel=2)
                    if not self._local_models_iterations:
                        raise RuntimeError(
                            "No iterations completed successfully"
                        ) from e
                    continue

        if not self._local_models_iterations:
            raise RuntimeError("No iterations completed successfully")

        return self

    def predict(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        r"""
        Predict using the fitted LESSA regressor.

        This method averages the predictions of all trained iterations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples to predict.
        n_estimators : int, optional
            The number of iterations to use for prediction. If None, all
            available iterations are used.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The averaged predicted values.
        """
        X = self._validate_prediction_input(X)

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=INTERNAL_DTYPE)

        # Determine number of iterations to use
        available_iterations = len(self._local_models_iterations)
        if n_estimators is None:
            n_estimators = available_iterations
        else:
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                raise ValueError(
                    f"n_estimators must be a positive integer, got {n_estimators}"
                )
            n_estimators = min(n_estimators, available_iterations)

        prediction_sum = np.zeros(n_samples, dtype=INTERNAL_DTYPE)
        valid_prediction_count = 0
        x_sq_norms = self._get_x_sq_norms(X)

        for iteration in range(n_estimators):
            try:
                local_models = self._local_models_iterations[iteration]
                center_matrix = self._local_center_matrices_iterations[iteration]
                linear_coefs = self._local_linear_coefs_iterations[iteration]
                linear_intercepts = self._local_linear_intercepts_iterations[iteration]
                global_model = self._global_models_iterations[iteration]

                Z = self._compute_weighted_features(
                    X,
                    local_models,
                    center_matrix=center_matrix,
                    x_sq_norms=x_sq_norms,
                    linear_coefs=linear_coefs,
                    linear_intercepts=linear_intercepts,
                )

                if global_model is not None:
                    iteration_predictions = global_model.predict(Z)
                else:
                    iteration_predictions = np.sum(Z, axis=1)

                # Validate predictions
                if np.all(np.isfinite(iteration_predictions)):
                    prediction_sum += iteration_predictions
                    valid_prediction_count += 1
                else:
                    warnings.warn(
                        f"Non-finite predictions in iteration {iteration}, skipping",
                        UserWarning,
                        stacklevel=2,
                    )

            except Exception as e:  # noqa: BLE001 - one bad round must not kill predict
                warnings.warn(
                    f"Error in prediction iteration {iteration}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        if valid_prediction_count == 0:
            raise RuntimeError("No valid predictions from any iteration")

        predictions = prediction_sum / INTERNAL_DTYPE(valid_prediction_count)

        return predictions

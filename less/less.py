import warnings
from typing import Optional, Callable, Union, Any
import numpy as np
from joblib import Parallel, delayed
from ._utils import (
    LocalModel,
    _validate_static_hyperparameters,
    _adjust_dynamic_parameters,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from xgboost import DMatrix, train as xgb_train

INTERNAL_DTYPE = np.float32


class _NativeXGBoostRegressor:
    """Lightweight sklearn-compatible wrapper around xgboost.train."""

    __slots__ = ("params", "num_boost_round", "_booster")

    def __init__(self, params: dict[str, Any], num_boost_round: int = 1):
        self.params = params
        self.num_boost_round = num_boost_round
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NativeXGBoostRegressor":
        dtrain = DMatrix(X, label=y)
        self._booster = xgb_train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, X: Union[np.ndarray, DMatrix]) -> np.ndarray:
        if self._booster is None:
            raise ValueError("Model is not fitted")
        dmatrix = X if isinstance(X, DMatrix) else DMatrix(X)
        return self._booster.predict(dmatrix)


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
        The method for selecting subset centers. 'tree' uses random sampling,
        while 'kmeans' and 'spectral' use clustering. A callable can be
        provided for custom clustering.
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
        local_estimator: Union[str, Callable[[], Any]] = "linear",
        global_estimator: Union[str, Callable[[], Any], None] = "xgboost",
        cluster_method: Union[str, Callable[..., Any]] = "tree",
        val_size: Optional[float] = None,
        kernel_coeff: Optional[float] = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
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

        # Initialize random generator
        self._rng = np.random.RandomState(self.random_state)

    def _get_local_estimator_factory(self) -> Callable[[], Any]:
        """Get the factory function for creating local estimator instances."""
        if self.local_estimator == "linear":
            return lambda: Ridge(alpha=1e-6, copy_X=False)
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

    def _get_global_estimator_factory(self) -> Optional[Callable[[], Any]]:
        """Get the factory function for creating the global estimator instance."""
        if self.global_estimator == "xgboost":
            base = self._native_global_xgb_rf_base_params
            return lambda: _NativeXGBoostRegressor(
                params={**base, "seed": self._rng.randint(2**31)},
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

        distance_sums = np.sum(distances, axis=1, keepdims=True)

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
        x_sq_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute all RBF distances in one vectorized pass (in-place)."""
        if X.shape[0] == 0 or centers.shape[0] == 0:
            return np.zeros((X.shape[0], centers.shape[0]), dtype=X.dtype)

        if x_sq_norms is None:
            x_sq_norms = np.einsum("ij,ij->i", X, X)

        x_sq_col = np.asarray(x_sq_norms, dtype=X.dtype).reshape(-1, 1)
        center_sq_row = np.einsum("ij,ij->i", centers, centers)[np.newaxis, :]

        # Build squared distances in a single buffer
        dist = np.dot(X, centers.T)          # (n_samples, n_subsets)
        dist *= -2.0
        dist += x_sq_col
        dist += center_sq_row
        np.maximum(dist, 0.0, out=dist)
        np.sqrt(dist, out=dist)
        dist *= -kernel_coeff
        np.exp(dist, out=dist)
        return self._safe_normalize_distances(dist)

    def _find_neighbor_indices(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        n_neighbors: int,
        x_sq_norms: Optional[np.ndarray] = None,
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
        center_sq_norms = np.einsum("ij,ij->i", centers, centers).reshape(-1, 1)

        # Build squared distances in a single buffer
        sq_dist = np.dot(centers, X.T)        # (n_subsets, n_samples)
        sq_dist *= -2.0
        sq_dist += center_sq_norms
        sq_dist += x_sq_norms[np.newaxis, :]
        np.maximum(sq_dist, 0.0, out=sq_dist)

        kth = n_neighbors - 1
        return np.argpartition(sq_dist, kth=kth, axis=1)[:, :n_neighbors]

    def _predict_local_outputs(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        linear_coefs: Optional[np.ndarray] = None,
        linear_intercepts: Optional[np.ndarray] = None,
        shared_dmatrix: Optional[DMatrix] = None,
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
                    [np.ravel(local_model.estimator.coef_) for local_model in local_models]
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
            except Exception:
                pass

        # Generic path: pre-allocate output, fill columns (no list+column_stack)
        if all(
            isinstance(local_model.estimator, _NativeXGBoostRegressor)
            for local_model in local_models
        ):
            if shared_dmatrix is None:
                shared_dmatrix = DMatrix(X)
            predict_input = shared_dmatrix
        else:
            predict_input = X

        n_models = len(local_models)
        out = np.empty((X.shape[0], n_models), dtype=INTERNAL_DTYPE)
        for i, local_model in enumerate(local_models):
            try:
                out[:, i] = local_model.estimator.predict(predict_input)
            except Exception as e:
                raise RuntimeError(
                    f"Error predicting with local model {i}: {str(e)}"
                ) from e

        return out

    def _get_linear_prediction_params(
        self, local_models: list[LocalModel]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        except Exception:
            return None, None

    def _get_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        r"""
        Select subset centers using the specified clustering method.

        Parameters
        ----------
        X : np.ndarray
            The input data from which to select centers.

        Returns
        -------
        np.ndarray
            The coordinates of the selected subset centers.

        Raises
        ------
        ValueError
            If `cluster_method` is not a recognized string or a callable.
        RuntimeError
            If the clustering process fails.
        """
        if self.cluster_method == "tree":
            # Randomly select subset centers
            center_indices = self._rng.choice(
                X.shape[0], size=self._n_subsets_adjusted, replace=False
            )
            return X[center_indices]
        elif callable(self.cluster_method):
            # Use custom clustering method
            clusterer = self.cluster_method(n_clusters=self._n_subsets_adjusted)
            clusterer.fit(X)
            return clusterer.cluster_centers_
        elif isinstance(self.cluster_method, str):
            # Use sklearn clustering
            try:
                if self.cluster_method == "kmeans":
                    # Use a new random seed for each call to ensure diversity across iterations
                    clusterer = KMeans(
                        n_clusters=self._n_subsets_adjusted,
                        random_state=self._rng.randint(2**31),
                    )
                elif self.cluster_method == "spectral":
                    clusterer = SpectralClustering(
                        n_clusters=self._n_subsets_adjusted, random_state=self._rng
                    )
                else:
                    raise ValueError(f"Invalid cluster_method: {self.cluster_method}")

                clusterer.fit(X)
                return clusterer.cluster_centers_
            except Exception as e:
                raise RuntimeError(f"Error during clustering: {str(e)}") from e
        else:
            raise ValueError(f"Invalid cluster_method: {self.cluster_method}")

    def _build_local_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prediction_data: Optional[np.ndarray] = None,
    ) -> tuple[list[LocalModel], np.ndarray, Optional[np.ndarray]]:
        r"""
        Build local models for one stage of the algorithm.

        Returns
        -------
        tuple[list[LocalModel], np.ndarray, Optional[np.ndarray]]
            - A list of trained `LocalModel` instances.
            - The center matrix of local models.
            - The weighted feature matrix Z (if *prediction_data* is given).
        """
        # Get cluster centers
        centers = self._get_cluster_centers(X)
        x_sq_norms = np.einsum("ij,ij->i", X, X)

        # High-dimensional data benefits more from brute-force top-k than tree search.
        neighbor_indices = self._find_neighbor_indices(
            X,
            centers,
            self._n_neighbors,
            x_sq_norms=x_sq_norms,
        )

        local_models = []
        local_centers = []

        local_estimators = [
            self._local_estimator_factory() for _ in range(len(neighbor_indices))
        ]

        n_neighbors = neighbor_indices.shape[1] if neighbor_indices.ndim == 2 else 0
        n_features = X.shape[1]

        # Train local models
        if self.local_n_jobs == 1 or len(neighbor_indices) <= 1:
            # Single scratch buffer reused across all subsets
            X_buf = np.empty((n_neighbors, n_features), dtype=X.dtype)
            y_buf = np.empty(n_neighbors, dtype=y.dtype)
            results = [
                self._fit_single_local_model(
                    local_est, X, y, neighbors, i, X_buf, y_buf
                )
                for i, (local_est, neighbors) in enumerate(
                    zip(local_estimators, neighbor_indices)
                )
            ]
        else:
            import threading

            _tls = threading.local()

            def _fit_with_buf(local_est, neighbors, i):
                if not hasattr(_tls, "X_buf"):
                    _tls.X_buf = np.empty(
                        (n_neighbors, n_features), dtype=X.dtype
                    )
                    _tls.y_buf = np.empty(n_neighbors, dtype=y.dtype)
                return self._fit_single_local_model(
                    local_est, X, y, neighbors, i, _tls.X_buf, _tls.y_buf,
                )

            with threadpool_limits(limits=1):
                results = Parallel(n_jobs=self.local_n_jobs, prefer="threads")(
                    delayed(_fit_with_buf)(local_est, neighbors, i)
                    for i, (local_est, neighbors) in enumerate(
                        zip(local_estimators, neighbor_indices)
                    )
                )

        for local_model, center in results:
            local_models.append(local_model)
            local_centers.append(center)

        if local_centers:
            center_matrix = np.vstack(local_centers).astype(INTERNAL_DTYPE, copy=False)
        else:
            center_matrix = np.zeros((0, X.shape[1]), dtype=INTERNAL_DTYPE)

        if prediction_data is None:
            return local_models, center_matrix, None

        linear_coefs, linear_intercepts = self._get_linear_prediction_params(local_models)
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
        X_buf: Optional[np.ndarray] = None,
        y_buf: Optional[np.ndarray] = None,
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
            if isinstance(local_estimator, (LinearRegression, Ridge)):
                # Scale in-place to fix conditioning; un-scale coefs after fit.
                std = np.std(X_local, axis=0)
                # Mask for constant / near-constant features:
                # leave them unscaled (std=1) so coef stays 0 after fit.
                safe = std > 1e-7
                std[~safe] = 1.0
                X_local -= center
                X_local /= std
                local_estimator.fit(
                    np.asarray(X_local, dtype=np.float64),
                    np.asarray(y_local, dtype=np.float64),
                )
                # Un-scale only the features that were actually scaled.
                coef = np.asarray(local_estimator.coef_).ravel()
                coef[safe] /= std[safe]
                coef[~safe] = 0.0
                local_estimator.coef_ = coef
                local_estimator.intercept_ -= coef @ center
            else:
                local_estimator.fit(X_local, y_local)
            return LocalModel(local_estimator, center), center
        except Exception as e:
            raise RuntimeError(f"Error training local model {index}: {str(e)}") from e

    def _compute_weighted_features(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        center_matrix: Optional[np.ndarray] = None,
        x_sq_norms: Optional[np.ndarray] = None,
        linear_coefs: Optional[np.ndarray] = None,
        linear_intercepts: Optional[np.ndarray] = None,
        shared_dmatrix: Optional[DMatrix] = None,
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
        local_preds = self._predict_local_outputs(
            X,
            local_models,
            linear_coefs=linear_coefs,
            linear_intercepts=linear_intercepts,
            shared_dmatrix=shared_dmatrix,
        )
        if center_matrix is None and local_models:
            center_matrix = np.vstack([local_model.center for local_model in local_models])
        elif center_matrix is None:
            center_matrix = np.zeros((0, X.shape[1]), dtype=X.dtype)

        kernel_coeff = self._get_kernel_coeff(len(local_models))
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
        )

        # Check feature count consistency
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but regressor "
                f"is expecting {self.n_features_in_} features as seen in fit."
            )

        return X

    def _prepare_fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
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
            )

        self._store_sklearn_attributes(X)

        if self.val_size is not None:
            if not (0 < self.val_size < 1):
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
        The method for selecting subset centers.
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
        local_estimator: Union[str, Callable[[], Any]] = "linear",
        global_estimator: Union[str, Callable[[], Any], None] = "xgboost",
        cluster_method: Union[str, Callable[..., Any]] = "tree",
        val_size: Optional[float] = None,
        kernel_coeff: Optional[float] = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        early_stopping_tolerance: float = 1e-8,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
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
        linear_coefs: Optional[np.ndarray],
        linear_intercepts: Optional[np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        cached_train_features: Optional[np.ndarray] = None,
    ) -> Optional[Any]:
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
                raise RuntimeError(f"Error training global model: {str(e)}") from e

        return global_est

    def _predict_stage(
        self,
        X: np.ndarray,
        local_models: list[LocalModel],
        center_matrix: np.ndarray,
        linear_coefs: Optional[np.ndarray],
        linear_intercepts: Optional[np.ndarray],
        global_model: Optional[Any],
        x_sq_norms: Optional[np.ndarray] = None,
        shared_dmatrix: Optional[DMatrix] = None,
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
            shared_dmatrix=shared_dmatrix,
        )

        if global_model is not None:
            return global_model.predict(Z)
        return np.sum(Z, axis=1)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LESSBRegressor":
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
        fit_x_sq_norms = (
            np.einsum("ij,ij->i", X, X) if self.val_size is not None else None
        )
        residuals = np.empty_like(y)

        for stage in range(self.n_estimators):
            try:
                np.subtract(y, current_predictions, out=residuals)

                if self.val_size is not None:
                    X_train, X_val, residuals_train, residuals_val = train_test_split(
                        X, residuals, test_size=self.val_size, random_state=self._rng
                    )
                else:
                    X_train, residuals_train = X, residuals
                    X_val, residuals_val = None, None

                prediction_data = X_train if self.val_size is None else None
                local_models, center_matrix, Z_stage = self._build_local_models(
                    X_train, residuals_train, prediction_data=prediction_data
                )
                linear_coefs, linear_intercepts = self._get_linear_prediction_params(
                    local_models
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
                    )
                    continue

                # In-place accumulation: avoid temp from learning_rate * stage_predictions
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
                warnings.warn(f"Error in boosting stage {stage}: {str(e)}", UserWarning)
                if not self._local_models_stages:
                    raise RuntimeError(
                        "No boosting stages completed successfully"
                    ) from e
                break

        if not self._local_models_stages:
            raise RuntimeError("No boosting stages completed successfully")

        return self

    def predict(self, X: np.ndarray, n_rounds: Optional[int] = None) -> np.ndarray:
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
        x_sq_norms = np.einsum("ij,ij->i", X, X)
        shared_dmatrix = None

        # Add predictions from specified number of stages
        for stage in range(n_rounds):
            try:
                local_models = self._local_models_stages[stage]
                center_matrix = self._local_center_matrices_stages[stage]
                linear_coefs = self._local_linear_coefs_stages[stage]
                linear_intercepts = self._local_linear_intercepts_stages[stage]
                global_model = self._global_models_stages[stage]
                if shared_dmatrix is None and all(
                    isinstance(local_model.estimator, _NativeXGBoostRegressor)
                    for local_model in local_models
                ):
                    shared_dmatrix = DMatrix(X)
                stage_predictions = self._predict_stage(
                    X,
                    local_models,
                    center_matrix,
                    linear_coefs,
                    linear_intercepts,
                    global_model,
                    x_sq_norms=x_sq_norms,
                    shared_dmatrix=shared_dmatrix,
                )

                if np.all(np.isfinite(stage_predictions)):
                    stage_predictions *= learning_rate
                    predictions += stage_predictions
                else:
                    warnings.warn(
                        f"Non-finite predictions in stage {stage}, skipping",
                        UserWarning,
                    )

            except Exception as e:
                warnings.warn(
                    f"Error in prediction stage {stage}: {str(e)}", UserWarning
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
        The method for selecting subset centers.
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
        local_estimator: Union[str, Callable[[], Any]] = "linear",
        global_estimator: Union[str, Callable[[], Any], None] = "xgboost",
        cluster_method: Union[str, Callable[..., Any]] = "tree",
        val_size: Optional[float] = None,
        kernel_coeff: Optional[float] = 0.1,
        min_neighbors: int = 10,
        local_n_jobs: int = -1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
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

    def _reset_state(self) -> None:
        """Reset the internal state of the regressor for refitting."""
        self._local_models_iterations = []
        self._local_center_matrices_iterations = []
        self._local_linear_coefs_iterations = []
        self._local_linear_intercepts_iterations = []
        self._global_models_iterations = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LESSARegressor":
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
                    if self.val_size is None and self._global_estimator_factory is not None
                    else None
                )
                local_models, center_matrix, Z_train = self._build_local_models(
                    X_train, y_train, prediction_data=prediction_data
                )
                linear_coefs, linear_intercepts = self._get_linear_prediction_params(
                    local_models
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
                                "Training predictions were not computed for the global estimator"
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
                warnings.warn(f"Error in iteration: {str(e)}", UserWarning)
                if not self._local_models_iterations:
                    raise RuntimeError("No iterations completed successfully") from e
                continue

        if not self._local_models_iterations:
            raise RuntimeError("No iterations completed successfully")

        return self

    def predict(self, X: np.ndarray, n_estimators: Optional[int] = None) -> np.ndarray:
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
        x_sq_norms = np.einsum("ij,ij->i", X, X)
        shared_dmatrix = None

        for iteration in range(n_estimators):
            try:
                local_models = self._local_models_iterations[iteration]
                center_matrix = self._local_center_matrices_iterations[iteration]
                linear_coefs = self._local_linear_coefs_iterations[iteration]
                linear_intercepts = self._local_linear_intercepts_iterations[iteration]
                global_model = self._global_models_iterations[iteration]
                if shared_dmatrix is None and all(
                    isinstance(local_model.estimator, _NativeXGBoostRegressor)
                    for local_model in local_models
                ):
                    shared_dmatrix = DMatrix(X)

                Z = self._compute_weighted_features(
                    X,
                    local_models,
                    center_matrix=center_matrix,
                    x_sq_norms=x_sq_norms,
                    linear_coefs=linear_coefs,
                    linear_intercepts=linear_intercepts,
                    shared_dmatrix=shared_dmatrix,
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
                    )

            except Exception as e:
                warnings.warn(
                    f"Error in prediction iteration {iteration}: {str(e)}", UserWarning
                )
                continue

        if valid_prediction_count == 0:
            raise RuntimeError("No valid predictions from any iteration")

        predictions = prediction_sum / INTERNAL_DTYPE(valid_prediction_count)

        return predictions

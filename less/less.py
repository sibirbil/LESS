import warnings
from typing import Optional, Callable, Union, Any
import numpy as np
from ._utils import (
    LocalModel,
    rbf_kernel,
    _validate_static_hyperparameters,
    _adjust_dynamic_parameters,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor


class BaseLESSRegressor(BaseEstimator, RegressorMixin):
    r"""
    Base class for LESS (Learning with Subset Stacking) Regressors.

    This base class provides common functionality for both gradient boosting
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
        Can be a string ('xgboost'), None (for simple averaging), or a
        callable that returns a scikit-learn compatible regressor.
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
    n_features_in_ : int
        The number of features seen during :meth:`fit`.
    feature_names_in_ : np.ndarray of shape (`n_features_in_`,)
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
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_subsets = n_subsets
        self.local_estimator = local_estimator
        self.global_estimator = global_estimator
        self.cluster_method = cluster_method
        self.val_size = val_size
        self.kernel_coeff = kernel_coeff
        self.min_neighbors = min_neighbors
        self.random_state = random_state

        _validate_static_hyperparameters(self)

        # Initialize random generator
        self._rng = np.random.RandomState(self.random_state)

    def _get_local_estimator_factory(self) -> Callable[[], Any]:
        """Get the factory function for creating local estimator instances."""
        if self.local_estimator == "linear":
            return lambda: LinearRegression()
        elif self.local_estimator == "tree":
            return lambda: DecisionTreeRegressor(
                max_leaf_nodes=31,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self._rng.randint(2**31),
            )
        elif callable(self.local_estimator):
            return self.local_estimator
        else:
            raise ValueError(f"Invalid local_estimator: {self.local_estimator}")

    def _get_global_estimator_factory(self) -> Optional[Callable[[], Any]]:
        """Get the factory function for creating the global estimator instance."""
        if self.global_estimator == "xgboost":
            return lambda: XGBRFRegressor(
                n_estimators=25, random_state=self._rng.randint(2**31), verbosity=0
            )
        elif self.global_estimator is None:
            return None
        elif callable(self.global_estimator):
            return self.global_estimator
        else:
            raise ValueError(f"Invalid global_estimator: {self.global_estimator}")

    def _safe_normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        r"""
        Safely normalize distance weights to avoid numerical instabilities.

        The method normalizes the distances so that each row sums to 1. It handles
        cases where the sum of distances in a row is close to zero by
        assigning a uniform weight to prevent division by zero.

        Parameters
        ----------
        distances : np.ndarray of shape (n_samples, n_subsets)
            The raw distance weights calculated from the RBF kernel.

        Returns
        -------
        np.ndarray of shape (n_samples, n_subsets)
            The normalized distance weights, where each row sums to 1.
        """
        if distances.shape[0] == 0:
            return distances

        # Calculate row sums with numerical stability
        distance_sums = np.sum(distances, axis=1, keepdims=True)

        # Handle edge cases where all distances are very small
        zero_mask = distance_sums.flatten() < 1e-12

        if np.any(zero_mask):
            # Use uniform distribution for problematic rows
            uniform_weight = 1.0 / distances.shape[1]
            distances[zero_mask] = uniform_weight
            distance_sums[zero_mask] = 1.0

        # Normalize with numerical stability
        normalized = np.divide(
            distances,
            distance_sums,
            out=np.zeros_like(distances),
            where=distance_sums != 0,
        )

        # Final normalization check
        row_sums = np.sum(normalized, axis=1, keepdims=True)
        normalized = np.divide(
            normalized, row_sums, out=normalized, where=row_sums != 0
        )

        return normalized

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
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[list[LocalModel], np.ndarray, np.ndarray]:
        r"""
        Build local models for one stage of the algorithm.

        This method selects subset centers, finds their nearest neighbors, and
        trains a local estimator for each subset.

        Parameters
        ----------
        X : np.ndarray
            The training features.
        y : np.ndarray
            The target values or residuals for boosting.

        Returns
        -------
        tuple[list[LocalModel], np.ndarray, np.ndarray]
            A tuple containing:
            - A list of trained `LocalModel` instances.
            - An array of predictions from each local model on `X`.
            - An array of distance-based weights for each sample to each subset.
        """
        n_samples = X.shape[0]

        # Get cluster centers
        centers = self._get_cluster_centers(X)

        # Build KDTree for efficient neighbor search
        try:
            tree = KDTree(X)
        except Exception as e:
            raise ValueError(f"Error building KDTree: {str(e)}") from e

        # Find neighbors for each center
        _, neighbor_indices = tree.query(centers, k=self._n_neighbors)

        local_models = []
        predictions = np.zeros((n_samples, self._n_subsets_adjusted))
        distances = np.zeros((n_samples, self._n_subsets_adjusted))

        # Determine kernel coefficient
        if self.kernel_coeff is None:
            kernel_coeff = (
                1.0 / (self._n_subsets_adjusted**2)
                if self._n_subsets_adjusted > 0
                else 1.0
            )
        else:
            kernel_coeff = self.kernel_coeff

        # Train local models
        for i, neighbors in enumerate(neighbor_indices):
            try:
                X_local = X[neighbors]
                y_local = y[neighbors]

                # Calculate subset center
                center = np.mean(X_local, axis=0)

                # Create and train local estimator
                local_est = self._local_estimator_factory()
                local_est.fit(X_local, y_local)

                # Store model
                local_models.append(LocalModel(local_est, center))

                # Get predictions and distances for all samples
                predictions[:, i] = local_est.predict(X)
                distances[:, i] = rbf_kernel(X, center, kernel_coeff)

            except Exception as e:
                raise RuntimeError(f"Error training local model {i}: {str(e)}") from e

        # Normalize distances safely
        distances = self._safe_normalize_distances(distances)

        return local_models, predictions, distances

    def _predict_with_models(
        self, X: np.ndarray, local_models: list[LocalModel]
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate predictions from a list of trained local models.

        Parameters
        ----------
        X : np.ndarray
            The input features for which to generate predictions.
        local_models : list[LocalModel]
            A list of trained `LocalModel` instances.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - An array of predictions from each local model.
            - An array of distance-based weights.
        """
        n_samples = X.shape[0]

        # Get local predictions and distances
        local_preds = np.zeros((n_samples, len(local_models)))
        distances = np.zeros((n_samples, len(local_models)))

        n_subsets = len(local_models)
        if self.kernel_coeff is None:
            kernel_coeff = 1.0 / (n_subsets**2) if n_subsets > 0 else 1.0
        else:
            kernel_coeff = self.kernel_coeff

        for i, local_model in enumerate(local_models):
            try:
                local_preds[:, i] = local_model.estimator.predict(X)
                distances[:, i] = rbf_kernel(X, local_model.center, kernel_coeff)
            except Exception as e:
                raise RuntimeError(
                    f"Error predicting with local model {i}: {str(e)}"
                ) from e

        # Normalize distances safely
        distances = self._safe_normalize_distances(distances)

        return local_preds, distances

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
        X = check_array(X, accept_sparse=False, dtype=np.float64)

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
            dtype=np.float64,
        )

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


class LESSGBRegressor(BaseLESSRegressor):
    r"""
    LESSGB (Learning with Subset Stacking Gradient Boosting) Regressor.

    This regressor implements the gradient boosting variant of the LESS algorithm.
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
    n_features_in_ : int
        The number of features seen during :meth:`fit`.
    feature_names_in_ : np.ndarray of shape (`n_features_in_`,)
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
            random_state=random_state,
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_tolerance = early_stopping_tolerance

    def _reset_state(self) -> None:
        """Reset the internal state of the regressor for refitting."""
        self._local_models_stages = []
        self._global_models_stages = []
        self._base_prediction = 0.0

    def _build_stage(
        self,
        local_models: list[LocalModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
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
                local_preds, distances = self._predict_with_models(X_val, local_models)
                Z_global = distances * local_preds
                y_global = y_val
            # Otherwise, train global model on the training set predictions
            else:
                local_preds, distances = self._predict_with_models(
                    X_train, local_models
                )
                Z_global = distances * local_preds
                y_global = y_train

            try:
                global_est = self._global_estimator_factory()
                global_est.fit(Z_global, y_global)
            except Exception as e:
                raise RuntimeError(f"Error training global model: {str(e)}") from e

        return global_est

    def _predict_stage(
        self, X: np.ndarray, local_models: list[LocalModel], global_model: Optional[Any]
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
        local_preds, distances = self._predict_with_models(X, local_models)
        Z = distances * local_preds

        if global_model is not None:
            try:
                return global_model.predict(Z)
            except Exception as e:
                raise RuntimeError(
                    f"Error predicting with global model: {str(e)}"
                ) from e
        else:
            return np.sum(Z, axis=1)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LESSGBRegressor":
        r"""
        Fit the LESSGB regressor using gradient boosting.

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
        LESSGBRegressor
            The fitted regressor.
        """
        self._reset_state()
        X, y = self._prepare_fit(X, y, sample_weight)

        self._base_prediction = np.mean(y)
        if not np.isfinite(self._base_prediction):
            raise ValueError("Target values contain non-finite values")

        current_predictions = np.full(y.shape, self._base_prediction)

        for stage in range(self.n_estimators):
            try:
                residuals = y - current_predictions

                if self.val_size is not None:
                    X_train, X_val, residuals_train, residuals_val = train_test_split(
                        X, residuals, test_size=self.val_size, random_state=self._rng
                    )
                else:
                    X_train, residuals_train = X, residuals
                    X_val, residuals_val = None, None

                local_models, _, _ = self._build_local_models(
                    X_train, residuals_train
                )

                global_model = self._build_stage(
                    local_models, X_train, residuals_train, X_val, residuals_val
                )

                stage_predictions = self._predict_stage(X, local_models, global_model)

                if not np.all(np.isfinite(stage_predictions)):
                    warnings.warn(
                        f"Non-finite predictions in stage {stage}, skipping",
                        UserWarning,
                    )
                    continue

                current_predictions += self.learning_rate * stage_predictions

                self._local_models_stages.append(local_models)
                self._global_models_stages.append(global_model)

                mean_abs_residual = np.mean(np.abs(y - current_predictions))
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
        Predict using the fitted LESSGB regressor.

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
            return np.array([])

        # Determine number of rounds to use
        available_rounds = len(self._local_models_stages)
        if n_rounds is None:
            n_rounds = available_rounds
        else:
            if not isinstance(n_rounds, int) or n_rounds <= 0:
                raise ValueError(f"n_rounds must be a positive integer, got {n_rounds}")
            n_rounds = min(n_rounds, available_rounds)

        # Start with base prediction
        predictions = np.full(n_samples, self._base_prediction)

        # Add predictions from specified number of stages
        for stage in range(n_rounds):
            try:
                local_models = self._local_models_stages[stage]
                global_model = self._global_models_stages[stage]
                stage_predictions = self._predict_stage(X, local_models, global_model)

                # Validate stage predictions
                if np.all(np.isfinite(stage_predictions)):
                    predictions += self.learning_rate * stage_predictions
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


class LESSAVRegressor(BaseLESSRegressor):
    r"""
    LESSAV (Learning with Subset Stacking Averaging) Regressor.

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
    n_features_in_ : int
        The number of features seen during :meth:`fit`.
    feature_names_in_ : np.ndarray of shape (`n_features_in_`,)
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
            random_state=random_state,
        )

        self.n_estimators = n_estimators

    def _reset_state(self) -> None:
        """Reset the internal state of the regressor for refitting."""
        self._local_models_iterations = []
        self._global_models_iterations = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LESSAVRegressor":
        r"""
        Fit the LESSAV regressor using model averaging.

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
        LESSAVRegressor
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

                local_models, train_preds, train_dists = self._build_local_models(
                    X_train, y_train
                )

                global_est = None
                if self._global_estimator_factory is not None:
                    if X_val is not None and y_val is not None:
                        val_preds, val_dists = self._predict_with_models(
                            X_val, local_models
                        )
                        Z_global = val_dists * val_preds
                        y_global = y_val
                    else:
                        Z_global = train_dists * train_preds
                        y_global = y_train

                    global_est = self._global_estimator_factory()
                    global_est.fit(Z_global, y_global)

                self._local_models_iterations.append(local_models)
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
        Predict using the fitted LESSAV regressor.

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
            return np.array([])

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

        # Collect predictions from all iterations
        all_predictions = []

        for iteration in range(n_estimators):
            try:
                local_models = self._local_models_iterations[iteration]
                global_model = self._global_models_iterations[iteration]

                # Get local predictions and distances
                local_preds, distances = self._predict_with_models(X, local_models)

                # Create features
                Z = distances * local_preds

                # Predict based on whether global model exists
                if global_model is not None:
                    iteration_predictions = global_model.predict(Z)
                else:
                    # Simple sum of weighted predictions
                    iteration_predictions = np.sum(Z, axis=1)

                # Validate predictions
                if np.all(np.isfinite(iteration_predictions)):
                    all_predictions.append(iteration_predictions)
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

        if len(all_predictions) == 0:
            raise RuntimeError("No valid predictions from any iteration")

        # Average all predictions
        predictions = np.mean(all_predictions, axis=0)

        return predictions
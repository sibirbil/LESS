import warnings
from typing import Optional, Callable, Union, Any
import numpy as np
from ._utils import LocalModel, rbf_kernel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBRFRegressor


class BaseLESSRegressor(BaseEstimator, RegressorMixin):
    """
    Base class for LESS (Learning with Subset Selection) Regressors.

    This base class provides common functionality for both gradient boosting
    and averaging variants of the LESS algorithm.

    Parameters
    ----------
    n_subsets : int, default=20
        Number of local subsets to create for training. Must be positive.
        Each subset focuses on a different region of the feature space.

    local_estimator : str or callable, default='linear'
        Local estimator type or factory function.
        - 'linear': Linear regression
        - 'tree': Decision tree with controlled complexity
        - callable: Custom estimator factory function

    global_estimator : str or callable, default='xgboost'
        Global meta-estimator factory function.
        - 'xgboost': XGBRFRegressor
        - None: Simple sum of weighted local predictions
        - callable: Custom estimator factory function

    kernel_coeff : float, default=0.1
        RBF kernel coefficient for distance weighting. Must be positive.
        Lower values create more localized influence.

    min_neighbors : int, default=10
        Minimum number of neighbors per subset. Must be positive.
        Ensures each local model has sufficient training data.

    random_state : int, RandomState instance or None, default=None
        Controls randomness of subset selection and estimator initialization.
        Pass int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        local_estimator: Union[str, Callable] = "linear",
        global_estimator: Union[str, Callable, None] = "xgboost",
        kernel_coeff: float = 0.1,
        min_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        self.n_subsets = n_subsets
        self.local_estimator = local_estimator
        self.global_estimator = global_estimator
        self.kernel_coeff = kernel_coeff
        self.min_neighbors = min_neighbors
        self.random_state = random_state

        # Initialize random generator
        self._rng = np.random.RandomState(random_state)

    def _get_local_estimator_factory(self) -> Callable:
        """Get local estimator factory function with validation."""
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

    def _get_global_estimator_factory(self) -> Optional[Callable]:
        """Get global estimator factory function."""
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

    def _validate_and_adjust_params(self, n_samples: int) -> None:
        """Validate and adjust parameters based on data size."""
        # Adjust n_subsets if larger than sample size
        if self.n_subsets > n_samples:
            warnings.warn(
                f"n_subsets ({self.n_subsets}) is larger than n_samples ({n_samples}). "
                f"Setting n_subsets to {n_samples}.",
                UserWarning,
            )
            self._n_subsets_adjusted = n_samples
        else:
            self._n_subsets_adjusted = self.n_subsets

        # Calculate optimal number of neighbors
        suggested_neighbors = max(
            self.min_neighbors, n_samples // self._n_subsets_adjusted
        )
        self._n_neighbors = min(suggested_neighbors, n_samples)

        # Warn if neighbors are too few
        if self._n_neighbors < self.min_neighbors:
            warnings.warn(
                f"Each subset will have only {self._n_neighbors} neighbors, "
                f"which is less than min_neighbors ({self.min_neighbors}). "
                "Consider reducing n_subsets or increasing sample size.",
                UserWarning,
            )

    def _safe_normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        """
        Safely normalize distance weights to avoid numerical instabilities.

        Parameters
        ----------
        distances : np.ndarray of shape (n_samples, n_subsets)
            Raw distance weights from RBF kernel

        Returns
        -------
        np.ndarray of shape (n_samples, n_subsets)
            Normalized distance weights that sum to 1 for each sample
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

    def _build_local_models(self, X: np.ndarray, y: np.ndarray, tree: KDTree) -> tuple:
        """
        Build local models for one stage/iteration.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Target values (or residuals for boosting)
        tree : KDTree
            KDTree for efficient neighbor search

        Returns
        -------
        tuple of (local_models, predictions, distances)
        """
        n_samples = X.shape[0]

        # Randomly select subset centers
        center_indices = self._rng.choice(
            n_samples, size=self._n_subsets_adjusted, replace=False
        )

        # Find neighbors for each center
        _, neighbor_indices = tree.query(X[center_indices], k=self._n_neighbors)

        local_models = []
        predictions = np.zeros((n_samples, self._n_subsets_adjusted))
        distances = np.zeros((n_samples, self._n_subsets_adjusted))

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
                distances[:, i] = rbf_kernel(X, center, self.kernel_coeff)

            except Exception as e:
                raise RuntimeError(f"Error training local model {i}: {str(e)}") from e

        # Normalize distances safely
        distances = self._safe_normalize_distances(distances)

        return local_models, predictions, distances

    def _predict_with_models(self, X: np.ndarray, local_models: list) -> np.ndarray:
        """
        Make predictions using trained local models.

        Parameters
        ----------
        X : np.ndarray
            Input features
        local_models : list
            List of LocalModel instances

        Returns
        -------
        tuple of (predictions, distances)
        """
        n_samples = X.shape[0]

        # Get local predictions and distances
        local_preds = np.zeros((n_samples, len(local_models)))
        distances = np.zeros((n_samples, len(local_models)))

        for i, local_model in enumerate(local_models):
            try:
                local_preds[:, i] = local_model.estimator.predict(X)
                distances[:, i] = rbf_kernel(X, local_model.center, self.kernel_coeff)
            except Exception as e:
                raise RuntimeError(
                    f"Error predicting with local model {i}: {str(e)}"
                ) from e

        # Normalize distances safely
        distances = self._safe_normalize_distances(distances)

        return local_preds, distances

    def _validate_input_data(self, X, y, sample_weight=None):
        """Common input validation for both variants."""
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

        return X, y

    def _store_sklearn_attributes(self, X):
        """Store sklearn-required attributes."""
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

    def _validate_prediction_input(self, X):
        """Common prediction input validation."""
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

    def fit(self, X, y, sample_weight=None):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")


class LESSGBRegressor(BaseLESSRegressor):
    """
    LESSGB (Learning with Subset Selection Gradient Boosting) Regressor.

    Additional Parameters
    ---------------------
    n_estimators : int, default=100
        Number of boosting iterations. Must be positive.
        More iterations may improve accuracy but increase computation time.

    learning_rate : float, default=0.1
        Learning rate for gradient boosting. Must be in (0, 1].
        Lower values require more estimators but may provide better generalization.

    early_stopping_tolerance : float, default=1e-8
        Tolerance for early stopping based on residual improvement.
        Training stops if mean absolute residual falls below this threshold.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        local_estimator: Union[str, Callable] = "linear",
        global_estimator: Union[str, Callable, None] = "xgboost",
        kernel_coeff: float = 0.1,
        min_neighbors: int = 10,
        early_stopping_tolerance: float = 1e-8,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_subsets=n_subsets,
            local_estimator=local_estimator,
            global_estimator=global_estimator,
            kernel_coeff=kernel_coeff,
            min_neighbors=min_neighbors,
            random_state=random_state,
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_tolerance = early_stopping_tolerance

    def _reset_state(self) -> None:
        """Reset internal state for refitting."""
        self._local_models_stages = []
        self._global_models_stages = []
        self._base_prediction = None

    def _build_stage(self, X: np.ndarray, residuals: np.ndarray, tree: KDTree) -> tuple:
        """Build one boosting stage with local and global models."""
        # Build local models
        local_models, predictions, distances = self._build_local_models(
            X, residuals, tree
        )

        # Create weighted features for global model
        Z = distances * predictions

        # Train global estimator if available
        global_est = None
        if self._global_estimator_factory is not None:
            try:
                global_est = self._global_estimator_factory()
                global_est.fit(Z, residuals)
            except Exception as e:
                raise RuntimeError(f"Error training global model: {str(e)}") from e

        return local_models, global_est

    def _predict_stage(
        self, X: np.ndarray, local_models: list, global_model: Optional[Any]
    ) -> np.ndarray:
        """Make predictions for one boosting stage."""
        # Get local predictions and distances
        local_preds, distances = self._predict_with_models(X, local_models)

        # Create features
        Z = distances * local_preds

        # Predict based on whether global model exists
        if global_model is not None:
            try:
                return global_model.predict(Z)
            except Exception as e:
                raise RuntimeError(f"Error predicting with global model: {str(e)}") from e
        else:
            # Simple sum of weighted predictions
            return np.sum(Z, axis=1)

    def fit(self, X, y, sample_weight=None):
        """Fit the LESSGB regressor using gradient boosting."""
        # Reset state for refitting
        self._reset_state()

        # Validate input
        X, y = self._validate_input_data(X, y, sample_weight)
        self._store_sklearn_attributes(X)

        n_samples = X.shape[0]

        # Validate and adjust parameters based on data
        self._validate_and_adjust_params(n_samples)

        # Initialize estimator factories
        self._local_estimator_factory = self._get_local_estimator_factory()
        self._global_estimator_factory = self._get_global_estimator_factory()

        # Build KDTree for efficient neighbor search
        try:
            tree = KDTree(X)
        except Exception as e:
            raise ValueError(f"Error building KDTree: {str(e)}") from e

        # Initialize predictions
        self._base_prediction = np.mean(y)
        if not np.isfinite(self._base_prediction):
            raise ValueError("Target values contain non-finite values")

        current_predictions = np.full(n_samples, self._base_prediction)
        residuals = y - current_predictions

        # Boosting iterations
        for stage in range(self.n_estimators):
            try:
                # Build stage to predict residuals
                local_models, global_model = self._build_stage(X, residuals, tree)

                # Get stage predictions
                stage_predictions = self._predict_stage(X, local_models, global_model)

                # Validate predictions
                if not np.all(np.isfinite(stage_predictions)):
                    warnings.warn(
                        f"Non-finite predictions in stage {stage}, skipping",
                        UserWarning,
                    )
                    continue

                # Update predictions with learning rate
                current_predictions += self.learning_rate * stage_predictions
                residuals = y - current_predictions

                # Store stage models
                self._local_models_stages.append(local_models)
                self._global_models_stages.append(global_model)

                # Early stopping check
                mean_abs_residual = np.mean(np.abs(residuals))
                if mean_abs_residual < self.early_stopping_tolerance:
                    if stage > 0:  # Only stop if we've made some progress
                        break

            except Exception as e:
                warnings.warn(f"Error in boosting stage {stage}: {str(e)}", UserWarning)
                # If no stages completed successfully, raise error
                if len(self._local_models_stages) == 0:
                    raise RuntimeError(
                        "No boosting stages completed successfully"
                    ) from e
                break

        if len(self._local_models_stages) == 0:
            raise RuntimeError("No boosting stages completed successfully")

        return self

    def predict(self, X, n_rounds: Optional[int] = None):
        """Predict using the LESSGB regressor."""
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
    """
    LESSAV (Learning with Subset Selection Averaging) Regressor.

    This variant uses model averaging instead of gradient boosting.
    Multiple sets of local models are trained and their predictions are averaged.

    Additional Parameters
    ---------------------
    n_iterations : int, default=10
        Number of averaging iterations. Each iteration creates a new set of local models.
        More iterations may improve stability and accuracy.
    """

    def __init__(
        self,
        n_subsets: int = 20,
        n_iterations: int = 100,
        local_estimator: Union[str, Callable] = "linear",
        global_estimator: Union[str, Callable, None] = "xgboost",
        kernel_coeff: float = 0.1,
        min_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_subsets=n_subsets,
            local_estimator=local_estimator,
            global_estimator=global_estimator,
            kernel_coeff=kernel_coeff,
            min_neighbors=min_neighbors,
            random_state=random_state,
        )

        self.n_iterations = n_iterations

    def _reset_state(self) -> None:
        """Reset internal state for refitting."""
        self._local_models_iterations = []
        self._global_models_iterations = []

    def fit(self, X, y, sample_weight=None):
        """Fit the LESSAV regressor using model averaging."""
        # Reset state for refitting
        self._reset_state()

        # Validate input
        X, y = self._validate_input_data(X, y, sample_weight)
        self._store_sklearn_attributes(X)

        n_samples = X.shape[0]

        # Validate and adjust parameters based on data
        self._validate_and_adjust_params(n_samples)

        # Initialize estimator factories
        self._local_estimator_factory = self._get_local_estimator_factory()
        self._global_estimator_factory = self._get_global_estimator_factory()

        # Build KDTree for efficient neighbor search
        try:
            tree = KDTree(X)
        except Exception as e:
            raise ValueError(f"Error building KDTree: {str(e)}") from e

        # Train multiple iterations of models
        for iteration in range(self.n_iterations):
            try:
                # Build local models for this iteration
                local_models, predictions, distances = self._build_local_models(
                    X, y, tree
                )

                # Create weighted features for global model
                Z = distances * predictions

                # Train global estimator if available
                global_est = None
                if self._global_estimator_factory is not None:
                    global_est = self._global_estimator_factory()
                    global_est.fit(Z, y)

                # Store models for this iteration
                self._local_models_iterations.append(local_models)
                self._global_models_iterations.append(global_est)

            except Exception as e:
                warnings.warn(f"Error in iteration {iteration}: {str(e)}", UserWarning)
                # If no iterations completed successfully, raise error
                if len(self._local_models_iterations) == 0:
                    raise RuntimeError("No iterations completed successfully") from e
                continue

        if len(self._local_models_iterations) == 0:
            raise RuntimeError("No iterations completed successfully")

        return self

    def predict(self, X, n_iterations: Optional[int] = None):
        """Predict using the LESSAV regressor by averaging predictions from multiple iterations."""
        X = self._validate_prediction_input(X)

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([])

        # Determine number of iterations to use
        available_iterations = len(self._local_models_iterations)
        if n_iterations is None:
            n_iterations = available_iterations
        else:
            if not isinstance(n_iterations, int) or n_iterations <= 0:
                raise ValueError(
                    f"n_iterations must be a positive integer, got {n_iterations}"
                )
            n_iterations = min(n_iterations, available_iterations)

        # Collect predictions from all iterations
        all_predictions = []

        for iteration in range(n_iterations):
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
import numpy as np
from typing import NamedTuple
import warnings


class LocalModel(NamedTuple):
    """Local estimator with its center point"""

    estimator: object
    center: np.array


def rbf_kernel(data, center, coeff=0.5):
    """RBF kernel using L2 norm with fixed coefficient"""
    return np.exp(-coeff * np.linalg.norm(data - center, ord=2, axis=1))


def _validate_static_hyperparameters(self) -> None:
    r"""
    Validate data-independent hyperparameters at initialization.

    Parameters
    ----------
    self : BaseLESSRegressor
        The regressor instance.

    Raises
    ------
    ValueError
        If any of the hyperparameters have invalid types or values.
    """
    if not isinstance(self.n_subsets, int) or self.n_subsets <= 0:
        raise ValueError(f"n_subsets must be a positive integer, got {self.n_subsets}")

    if hasattr(self, "n_estimators") and (
        not isinstance(self.n_estimators, int) or self.n_estimators <= 0
    ):
        raise ValueError(
            f"n_estimators must be a positive integer, got {self.n_estimators}"
        )

    if hasattr(self, "n_iterations") and (
        not isinstance(self.n_iterations, int) or self.n_iterations <= 0
    ):
        raise ValueError(
            f"n_iterations must be a positive integer, got {self.n_iterations}"
        )

    if hasattr(self, "learning_rate") and (
        not isinstance(self.learning_rate, (float, int))
        or not (0 < self.learning_rate <= 1)
    ):
        raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

    if not isinstance(self.kernel_coeff, (float, int)) or self.kernel_coeff <= 0:
        raise ValueError(f"kernel_coeff must be positive, got {self.kernel_coeff}")

    if not isinstance(self.min_neighbors, int) or self.min_neighbors <= 0:
        raise ValueError(
            f"min_neighbors must be a positive integer, got {self.min_neighbors}"
        )

    if hasattr(self, "early_stopping_tolerance") and (
        not isinstance(self.early_stopping_tolerance, (float, int))
        or self.early_stopping_tolerance < 0
    ):
        raise ValueError(
            f"early_stopping_tolerance must be non-negative, got {self.early_stopping_tolerance}"
        )

    if isinstance(self.local_estimator, str) and self.local_estimator not in [
        "linear",
        "tree",
    ]:
        raise ValueError(
            f"local_estimator string must be 'linear' or 'tree', got {self.local_estimator}"
        )

    if not isinstance(self.random_state, (type(None), int, np.random.RandomState)):
        raise ValueError(
            f"random_state must be None, an integer, or a RandomState instance, got {self.random_state}"
        )


def _adjust_dynamic_parameters(self, n_samples: int) -> tuple[int, int]:
    r"""
    Adjust data-dependent parameters at fit time.

    Parameters
    ----------
    self : BaseLESSRegressor
        The regressor instance.
    n_samples : int
        The number of samples in the training data.

    Returns
    -------
    tuple[int, int]
        A tuple containing the adjusted number of subsets and neighbors.
    """
    if self.n_subsets > n_samples:
        warnings.warn(
            f"n_subsets ({self.n_subsets}) is larger than n_samples ({n_samples}). "
            f"Setting n_subsets to {n_samples}.",
            UserWarning,
        )
        n_subsets_adjusted = n_samples
    else:
        n_subsets_adjusted = self.n_subsets

    if n_subsets_adjusted == 0:
        # This case can happen if n_samples is 0
        return 0, 0

    suggested_neighbors = max(self.min_neighbors, n_samples // n_subsets_adjusted)
    n_neighbors = min(suggested_neighbors, n_samples)

    if n_neighbors < self.min_neighbors:
        warnings.warn(
            f"Each subset will have only {n_neighbors} neighbors, which is less than "
            f"min_neighbors ({self.min_neighbors}). Consider reducing n_subsets or increasing sample size.",
            UserWarning,
        )

    return n_subsets_adjusted, n_neighbors

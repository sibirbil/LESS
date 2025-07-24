import numpy as np
from typing import NamedTuple


class LocalModel(NamedTuple):
    """Local estimator with its center point"""
    estimator: object
    center: np.array


def rbf_kernel(data, center, coeff=0.5):
    """RBF kernel using L2 norm with fixed coefficient"""
    return np.exp(-coeff * np.linalg.norm(data - center, ord=2, axis=1))

def _validate_hyperparameters(
    n_subsets,
    n_estimators,
    learning_rate,
    kernel_coeff,
    min_neighbors,
    early_stopping_tolerance,
    local_estimator,
    global_estimator,
    random_state
) -> None:
    """Validate hyperparameters before fitting."""
    if not isinstance(n_subsets, int) or n_subsets <= 0:
        raise ValueError(f"n_subsets must be a positive integer, got {n_subsets}")
        
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError(f"n_estimators must be a positive integer, got {n_estimators}")
        
    if not isinstance(learning_rate, (int, float)) or not (0 < learning_rate <= 1):
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        
    if not isinstance(kernel_coeff, (int, float)) or kernel_coeff <= 0:
        raise ValueError(f"kernel_coeff must be positive, got {kernel_coeff}")
        
    if not isinstance(min_neighbors, int) or min_neighbors <= 0:
        raise ValueError(f"min_neighbors must be a positive integer, got {min_neighbors}")
        
    if not isinstance(early_stopping_tolerance, (int, float)) or early_stopping_tolerance < 0:
        raise ValueError(f"early_stopping_tolerance must be non-negative, got {early_stopping_tolerance}")
        
    if local_estimator not in ['linear', 'tree'] and not callable(local_estimator):
        raise ValueError(f"local_estimator must be 'linear', 'tree', or callable, got {local_estimator}")
    
    # If local_estimator is callable, check if it has fit and predict methods
    if callable(local_estimator):
        if not hasattr(local_estimator, 'fit'):
            raise ValueError("local_estimator must have a 'fit' method")
        if not hasattr(local_estimator, 'predict'):
            raise ValueError("local_estimator must have a 'predict' method")
        
    if global_estimator is not None:
        if not callable(global_estimator):
            raise ValueError("global_estimator must be callable or None")
        # If global_estimator is callable, check if it has fit and predict methods
        if not hasattr(global_estimator, 'fit'):
            raise ValueError("global_estimator must have a 'fit' method")
        if not hasattr(global_estimator, 'predict'):
            raise ValueError("global_estimator must have a 'predict' method")
    
    if random_state is not None:
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError(f"random_state must be a non-negative integer or None, got {random_state}")
import numpy as np
import pytest

from less import LESSARegressor, LESSBRegressor


def reference_weights(X, centers, kernel_coeff):
    """Normalized exp(-k * ||x - c||) computed in float64 with a stable softmax."""
    X = X.astype(np.float64)
    centers = centers.astype(np.float64)
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    logits = -kernel_coeff * d
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("kernel_coeff", [0.0, 0.0025, 0.1, 1.0, 10.0])
def test_weights_match_reference(kernel_coeff):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(7, 5)).astype(np.float32)
    X = rng.normal(size=(50, 5)).astype(np.float32)
    w = LESSBRegressor()._compute_distance_matrix(X, centers, kernel_coeff)
    np.testing.assert_allclose(
        w, reference_weights(X, centers, kernel_coeff), atol=1e-5
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)


def test_far_points_keep_their_weights():
    # Every kernel term underflows: exp(-1 * 30) is below any useful threshold.
    centers = np.zeros((3, 1), dtype=np.float32)
    centers[:, 0] = [0.0, 1.0, 2.0]
    X = np.array([[32.0], [-30.0]], dtype=np.float32)
    w = LESSBRegressor()._compute_distance_matrix(X, centers, 1.0)
    np.testing.assert_allclose(w, reference_weights(X, centers, 1.0), atol=1e-5)
    # The nearest center must dominate, not a uniform 1/3 split.
    assert w[0].argmax() == 2 and w[1].argmax() == 0
    assert w.max(axis=1).min() > 0.6


def test_single_center_gets_all_weight():
    X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32) * 100
    w = LESSBRegressor()._compute_distance_matrix(X, np.zeros((1, 3), np.float32), 1.0)
    np.testing.assert_array_equal(w, np.ones((10, 1), np.float32))


@pytest.mark.parametrize("bad", [np.inf, np.nan, -0.1])
def test_invalid_kernel_coeff_rejected(bad):
    X = np.random.default_rng(2).normal(size=(40, 2))
    with pytest.raises(ValueError, match="kernel_coeff"):
        LESSBRegressor(kernel_coeff=bad).fit(X, X[:, 0])


@pytest.mark.parametrize("cls", [LESSARegressor, LESSBRegressor])
def test_far_test_points_predict_finite(cls):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] - 2 * X[:, 1]
    model = cls(n_subsets=5, kernel_coeff=1.0, n_estimators=5, random_state=0).fit(X, y)
    pred = model.predict(X * 50)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("cls", [LESSARegressor, LESSBRegressor])
@pytest.mark.parametrize("local_estimator", ["linear", "tree"])
def test_column_offset_does_not_change_the_model(cls, local_estimator):
    # Shifting every column by a constant moves no point relative to another,
    # so neighbours, weights and predictions must stay the same. Uncentered,
    # float32 distances at an offset of 1e4 picked the wrong neighbours.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]
    X_test = rng.normal(size=(200, 5))
    y_test = np.sin(2 * X_test[:, 0]) + X_test[:, 1] * X_test[:, 2]
    kwargs = {
        "n_subsets": 10,
        "n_estimators": 5,
        "local_estimator": local_estimator,
        "kernel_coeff": 1.0,
        "random_state": 0,
    }
    base = cls(**kwargs).fit(X, y).predict(X_test)
    shifted = cls(**kwargs).fit(X + 1e4, y).predict(X_test + 1e4)
    mse_base = np.mean((base - y_test) ** 2)
    mse_shifted = np.mean((shifted - y_test) ** 2)
    assert mse_shifted == pytest.approx(mse_base, rel=0.05)

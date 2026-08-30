import numpy as np
import pytest
import torch

from sentiment_manifold.directions import KMeansFitter, LogisticRegressionFitter, create_fitter


@pytest.fixture
def separated_data():
    rng = np.random.default_rng(0)
    negative = rng.normal(loc=-2.0, scale=0.1, size=(30, 5))
    positive = rng.normal(loc=2.0, scale=0.1, size=(30, 5))
    return np.concatenate([negative, positive]), np.array([0] * 30 + [1] * 30)


@pytest.mark.parametrize("method", ["mean_diff", "kmeans", "logistic_regression", "pca"])
def test_closed_form_fitters_return_positive_unit_lines(method, separated_data):
    activations, labels = separated_data
    kwargs = {"random_state": 0} if method in {"kmeans", "logistic_regression"} else {}
    result = create_fitter(method, **kwargs).fit(activations, labels)
    mean_difference = activations[labels == 1].mean(0) - activations[labels == 0].mean(0)
    assert result.direction.shape == (activations.shape[1],)
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    assert result.direction @ mean_difference > 0


def test_reference_fitter_hyperparameters():
    assert KMeansFitter().n_init == 10
    logistic = LogisticRegressionFitter()
    assert logistic.c == pytest.approx(1.0)
    assert logistic.solver == "liblinear"
    assert logistic.max_iter == 1000
    assert logistic.tol == pytest.approx(1e-4)


def test_random_control_matches_upstream_layer_sequence(separated_data):
    activations, labels = separated_data
    result = create_fitter("random", layer=2, seed=42).fit(activations, labels)
    generator = torch.Generator(device="cpu").manual_seed(42)
    expected = None
    for _ in range(3):
        expected = torch.randn(activations.shape[1], generator=generator).numpy()
    assert expected is not None
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.direction, expected)

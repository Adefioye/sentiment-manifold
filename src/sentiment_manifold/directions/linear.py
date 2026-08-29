"""The four non-DAS direction fits used by Tigges et al."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .base import DirectionFitter, FitResult, FloatArray, IntArray


class MeanDifferenceFitter(DirectionFitter):
    name = "mean_diff"

    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        x, y = self.validate(activations, labels)
        direction = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        return FitResult(
            self.name,
            direction,
            {"n_samples": len(x), **self.orientation_diagnostics(direction, x, y)},
        )


class KMeansFitter(DirectionFitter):
    name = "kmeans"

    def __init__(self, *, random_state: int = 0, n_init: int = 10) -> None:
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        x, y = self.validate(activations, labels)
        model = KMeans(n_clusters=2, random_state=self.random_state, n_init=self.n_init).fit(x)
        cluster_sentiment = [float(y[model.labels_ == cluster].mean()) for cluster in range(2)]
        positive_cluster = int(np.argmax(cluster_sentiment))
        negative_cluster = 1 - positive_cluster
        direction = (
            model.cluster_centers_[positive_cluster] - model.cluster_centers_[negative_cluster]
        )
        predicted = (model.labels_ == positive_cluster).astype(int)
        return FitResult(
            self.name,
            direction,
            {
                "train_accuracy": float((predicted == y).mean()),
                "inertia": float(model.inertia_),
                "n_init": self.n_init,
                **self.orientation_diagnostics(direction, x, y),
            },
        )


class LogisticRegressionFitter(DirectionFitter):
    name = "logistic_regression"

    def __init__(self, *, random_state: int = 0, max_iter: int = 1000, tol: float = 1e-4) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        x, y = self.validate(activations, labels)
        model = LogisticRegression(
            solver="liblinear",
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(x, y)
        direction = model.coef_[0]
        return FitResult(
            self.name,
            direction,
            {
                "train_accuracy": float(model.score(x, y)),
                "intercept": float(model.intercept_[0]),
                "solver": "liblinear",
                "max_iter": self.max_iter,
                "tol": self.tol,
                **self.orientation_diagnostics(direction, x, y),
            },
        )


class PCAFitter(DirectionFitter):
    name = "pca"

    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        x, y = self.validate(activations, labels)
        model = PCA(n_components=1).fit(x)
        raw_direction = model.components_[0]
        direction = self.orient(raw_direction, x, y)
        return FitResult(
            self.name,
            direction,
            {
                "explained_variance_ratio": float(model.explained_variance_ratio_[0]),
                "upstream_raw_sign_is_arbitrary": True,
                **self.orientation_diagnostics(raw_direction, x, y),
            },
        )


class RandomDirectionFitter(DirectionFitter):
    """Tigges's layer-indexed unit-Gaussian random control (global seed 42)."""

    name = "random"

    def __init__(self, *, layer: int, seed: int = 42) -> None:
        self.layer = layer
        self.seed = seed

    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        x, y = self.validate(activations, labels)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        direction = None
        for _ in range(self.layer + 1):
            direction = torch.randn(x.shape[1], generator=generator).numpy()
        assert direction is not None
        return FitResult(
            self.name,
            direction,
            {
                "random_seed": self.seed,
                "random_sequence_index": self.layer,
                "orientation_convention": "unoriented_random_control",
            },
        )

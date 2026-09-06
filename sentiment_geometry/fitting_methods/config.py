"""Configuration shared by direction and subspace fitting methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FittingConfig:
    kmeans_n_init: int = 10
    logistic_c: float = 1.0
    logistic_solver: str = "liblinear"
    logistic_max_iter: int = 1000
    logistic_tol: float = 1e-4


@dataclass
class DASConfig:
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    max_grad_norm: float = 1.0
    implementation: str = "tigges_rotation"


__all__ = ["DASConfig", "FittingConfig"]

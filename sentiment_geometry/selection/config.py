"""Validation-only hyperparameter-selection configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TuningConfig:
    validation_fraction: float = 0.25
    selection_metric: str = "toy_validation_logit_diff_percent"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    kmeans_n_init: list[int] = field(default_factory=lambda: [10, 50])
    logistic_c: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    das_learning_rate: list[float] = field(default_factory=lambda: [3e-4, 1e-3])
    das_weight_decay: list[float] = field(default_factory=lambda: [0.0])
    das_epochs: list[int] = field(default_factory=lambda: [32, 64])
    das_batch_size: list[int] = field(default_factory=lambda: [128])
    das_max_grad_norm: list[float] = field(default_factory=lambda: [1.0])


__all__ = ["TuningConfig"]

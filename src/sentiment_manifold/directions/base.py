"""Common API for sentiment-direction fitting methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass
class FitResult:
    method: str
    direction: FloatArray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=np.float32)
        if direction.ndim == 1:
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError(f"{self.method} produced a zero or non-finite direction")
            self.direction = direction / norm
            return
        if direction.ndim != 2 or direction.shape[1] < 1:
            raise ValueError("Directions must have shape [d_model] or [d_model, d_subspace]")
        if not np.isfinite(direction).all():
            raise ValueError(f"{self.method} produced a non-finite subspace")
        q, r = np.linalg.qr(direction)
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        self.direction = (q[:, : direction.shape[1]] * signs).astype(np.float32)


class DirectionFitter(ABC):
    name: str

    @abstractmethod
    def fit(self, activations: FloatArray, labels: IntArray) -> FitResult:
        """Fit a positive-oriented unit direction."""

    @staticmethod
    def validate(activations: FloatArray, labels: IntArray) -> tuple[FloatArray, IntArray]:
        x = np.asarray(activations, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError(
                f"Expected activations [samples, features] and aligned labels, got {x.shape}"
            )
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Both binary classes must be present")
        if not np.isfinite(x).all():
            raise ValueError("Activations contain NaN or infinity")
        return x, y

    @staticmethod
    def orient(direction: FloatArray, activations: FloatArray, labels: IntArray) -> FloatArray:
        positive = activations[labels == 1].mean(axis=0)
        negative = activations[labels == 0].mean(axis=0)
        return direction if np.dot(direction, positive - negative) >= 0 else -direction

    @staticmethod
    def orientation_diagnostics(
        raw_direction: FloatArray, activations: FloatArray, labels: IntArray
    ) -> dict[str, Any]:
        positive = activations[labels == 1].mean(axis=0)
        negative = activations[labels == 0].mean(axis=0)
        dot = float(np.dot(raw_direction, positive - negative))
        return {
            "orientation_convention": "negative_to_positive",
            "orientation_reference": "toy_train_class_mean_difference",
            "raw_orientation_dot": dot,
            "orientation_sign_flipped": bool(dot < 0),
        }

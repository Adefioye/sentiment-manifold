"""Correlational direction diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def projection_accuracy(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    threshold: float,
) -> float:
    scores = np.asarray(activations) @ np.asarray(direction)
    predictions = (scores >= threshold).astype(np.int64)
    return float((predictions == np.asarray(labels)).mean())


def projection_threshold(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
) -> float:
    scores = np.asarray(activations) @ np.asarray(direction)
    return float(0.5 * (scores[labels == 0].mean() + scores[labels == 1].mean()))


def cosine_similarity_table(
    directions: dict[str, np.ndarray], *, absolute: bool = True
) -> pd.DataFrame:
    names = list(directions)
    matrix = np.stack([np.asarray(directions[name]) for name in names])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarities = matrix @ matrix.T
    if absolute:
        similarities = np.abs(similarities)
    return pd.DataFrame(similarities, index=names, columns=names)

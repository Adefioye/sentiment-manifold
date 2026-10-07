"""Native and midpoint-threshold metrics shared by binary probe families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score


def _validate(scores: NDArray[np.floating], labels: NDArray[np.integer]) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(score_values) != len(label_values) or not len(label_values):
        raise ValueError("Scores and labels must be non-empty and aligned")
    if set(np.unique(label_values)) != {0, 1}:
        raise ValueError("Both binary classes must be present")
    if not np.isfinite(score_values).all():
        raise ValueError("Probe scores contain NaN or infinity")
    return score_values, label_values


def midpoint_threshold(
    training_scores: NDArray[np.floating], training_labels: NDArray[np.integer]
) -> float:
    """Return the midpoint between the two projected training-class means."""

    scores, labels = _validate(training_scores, training_labels)
    return float(0.5 * (scores[labels == 0].mean() + scores[labels == 1].mean()))


@dataclass(frozen=True)
class ProbeEvaluation:
    native_accuracy: float
    native_balanced_accuracy: float
    midpoint_accuracy: float
    midpoint_balanced_accuracy: float
    prediction_agreement: float
    native_threshold: float
    midpoint_threshold: float
    n_examples: int
    native_predictions: NDArray[np.int64]
    midpoint_predictions: NDArray[np.int64]


def evaluate_binary_probe(
    *,
    training_midpoint_scores: NDArray[np.floating],
    training_labels: NDArray[np.integer],
    native_scores: NDArray[np.floating],
    midpoint_scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    native_threshold: float,
) -> ProbeEvaluation:
    """Evaluate a fitted probe with its native and fitted-midpoint decisions."""

    train_scores, train_labels = _validate(training_midpoint_scores, training_labels)
    native_values, eval_labels = _validate(native_scores, labels)
    midpoint_values, midpoint_labels = _validate(midpoint_scores, labels)
    if not np.array_equal(eval_labels, midpoint_labels):
        raise ValueError("Native and midpoint evaluations use different labels")
    threshold = midpoint_threshold(train_scores, train_labels)
    native_predictions = (native_values >= native_threshold).astype(np.int64)
    midpoint_predictions = (midpoint_values >= threshold).astype(np.int64)
    return ProbeEvaluation(
        native_accuracy=float((native_predictions == eval_labels).mean()),
        native_balanced_accuracy=float(
            balanced_accuracy_score(eval_labels, native_predictions)
        ),
        midpoint_accuracy=float((midpoint_predictions == eval_labels).mean()),
        midpoint_balanced_accuracy=float(
            balanced_accuracy_score(eval_labels, midpoint_predictions)
        ),
        prediction_agreement=float((native_predictions == midpoint_predictions).mean()),
        native_threshold=float(native_threshold),
        midpoint_threshold=threshold,
        n_examples=len(eval_labels),
        native_predictions=native_predictions,
        midpoint_predictions=midpoint_predictions,
    )


__all__ = ["ProbeEvaluation", "evaluate_binary_probe", "midpoint_threshold"]

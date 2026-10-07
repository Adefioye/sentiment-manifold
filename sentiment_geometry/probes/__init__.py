"""Binary probes and random-label controls for representation diagnostics."""

from .controls import balanced_label_permutation
from .metrics import ProbeEvaluation, evaluate_binary_probe, midpoint_threshold
from .models import (
    BinaryProbe,
    LogisticProbeConfig,
    MLP1ProbeConfig,
    fit_logistic_probe,
    fit_mean_difference_probe,
    fit_mlp1_probe,
)

__all__ = [
    "BinaryProbe",
    "LogisticProbeConfig",
    "MLP1ProbeConfig",
    "ProbeEvaluation",
    "balanced_label_permutation",
    "evaluate_binary_probe",
    "fit_logistic_probe",
    "fit_mean_difference_probe",
    "fit_mlp1_probe",
    "midpoint_threshold",
]

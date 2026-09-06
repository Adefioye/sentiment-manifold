"""Reusable tools for causal analysis of sentiment representation geometry."""

from .activations import extract_activations
from .fitting_methods import DirectionArtifact, create_fitter, list_fitters
from .models import DeviceSpec, resolve_device

__all__ = [
    "DeviceSpec",
    "DirectionArtifact",
    "create_fitter",
    "extract_activations",
    "list_fitters",
    "resolve_device",
]

__version__ = "0.1.0"

"""Geometric and correlational analysis of sentiment representations."""

from .direction_alignment import (
    DirectionAlignmentConfig,
    DirectionAlignmentResult,
    DirectionAlignmentSource,
    SelectedDirection,
    run_direction_alignment_analysis,
)
from .projections import cosine_similarity_table, projection_accuracy, projection_threshold

__all__ = [
    "DirectionAlignmentConfig",
    "DirectionAlignmentResult",
    "DirectionAlignmentSource",
    "SelectedDirection",
    "cosine_similarity_table",
    "projection_accuracy",
    "projection_threshold",
    "run_direction_alignment_analysis",
]

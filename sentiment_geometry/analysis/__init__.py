"""Geometric and correlational analysis of sentiment representations."""

from .direction_alignment import (
    DirectionAlignmentConfig,
    DirectionAlignmentResult,
    DirectionAlignmentSource,
    PairwiseDirectionAlignmentConfig,
    PairwiseDirectionAlignmentResult,
    SelectedDirection,
    load_direction_alignment_result,
    load_pairwise_direction_alignment_result,
    run_direction_alignment_analysis,
    run_pairwise_direction_alignment_analysis,
)
from .projections import cosine_similarity_table, projection_accuracy, projection_threshold

__all__ = [
    "DirectionAlignmentConfig",
    "DirectionAlignmentResult",
    "DirectionAlignmentSource",
    "PairwiseDirectionAlignmentConfig",
    "PairwiseDirectionAlignmentResult",
    "SelectedDirection",
    "cosine_similarity_table",
    "load_direction_alignment_result",
    "load_pairwise_direction_alignment_result",
    "projection_accuracy",
    "projection_threshold",
    "run_direction_alignment_analysis",
    "run_pairwise_direction_alignment_analysis",
]

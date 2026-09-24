"""Publication-facing result selection, tables, and plots."""

from .ait_valence import (
    AITValenceReportData,
    load_ait_valence_report_data,
    plot_ait_valence_run,
)
from .direction_alignment import plot_direction_alignment
from .sentiment_position import (
    DATASET_ORDER,
    MODEL_ORDER,
    SentimentPositionReportData,
    plot_cosine_similarity_grid,
    plot_cross_position_cosines,
    plot_logit_difference_grid,
    selected_layer_table,
)
from .table1 import (
    select_table1_best_layers,
    table1_cell_text,
    validate_best_layers,
)

__all__ = [
    "DATASET_ORDER",
    "MODEL_ORDER",
    "AITValenceReportData",
    "SentimentPositionReportData",
    "load_ait_valence_report_data",
    "plot_ait_valence_run",
    "plot_cosine_similarity_grid",
    "plot_cross_position_cosines",
    "plot_direction_alignment",
    "plot_logit_difference_grid",
    "select_table1_best_layers",
    "selected_layer_table",
    "table1_cell_text",
    "validate_best_layers",
]

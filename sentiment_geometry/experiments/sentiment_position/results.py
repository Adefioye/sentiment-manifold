"""Pure result aggregation for sentiment fitting-position comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExperimentTables:
    """In-memory tables produced by a sentiment-position comparison."""

    metrics: list[dict[str, Any]] = field(default_factory=list)
    patching_records: list[dict[str, Any]] = field(default_factory=list)
    direction_metadata: list[dict[str, Any]] = field(default_factory=list)
    das_losses: list[dict[str, Any]] = field(default_factory=list)
    direction_similarities: list[dict[str, Any]] = field(default_factory=list)


def select_best_layers(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each dataset/metric maximum independently, breaking ties low."""

    group_columns = ["model", "method", "fit_position", "dataset"]
    metric_columns = ("logit_difference_percent", "logit_flip_percent")
    required = {*group_columns, "layer", *metric_columns}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Cannot select best layers; missing columns: {missing}")
    rows: list[dict] = []
    ordered = metrics.sort_values([*group_columns, "layer"], kind="stable")
    for keys, group in ordered.groupby(group_columns, sort=False):
        for metric_column in metric_columns:
            valid = group.dropna(subset=[metric_column])
            if valid.empty:
                selected_layer: int | None = None
                value = float("nan")
            else:
                selected = valid.loc[valid[metric_column].idxmax()]
                selected_layer = int(selected["layer"])
                value = float(selected[metric_column])
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric_column.removesuffix("_percent"),
                    "metric_column": metric_column,
                    "layer": selected_layer,
                    "value_percent": value,
                }
            )
    return pd.DataFrame(rows)


def direction_similarity_rows(
    *,
    model: str,
    layer: int,
    directions: Mapping[tuple[str, str], np.ndarray],
) -> list[dict]:
    """Return signed and absolute pairwise cosines for named directions."""

    units = {key: np.asarray(vector) / np.linalg.norm(vector) for key, vector in directions.items()}
    rows: list[dict] = []
    for (left_position, left_method), left in units.items():
        for (right_position, right_method), right in units.items():
            signed = float(left @ right)
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "position_a": left_position,
                    "method_a": left_method,
                    "direction_a": f"{left_position}:{left_method}",
                    "position_b": right_position,
                    "method_b": right_method,
                    "direction_b": f"{right_position}:{right_method}",
                    "signed_cosine": signed,
                    "absolute_cosine": abs(signed),
                }
            )
    return rows


__all__ = ["ExperimentTables", "direction_similarity_rows", "select_best_layers"]

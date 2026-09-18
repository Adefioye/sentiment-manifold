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
    das_epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    direction_similarities: list[dict[str, Any]] = field(default_factory=list)


def select_layers_by_validation_metric(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    metric: str = "logit_flip_percent",
) -> pd.DataFrame:
    """Select one layer per fitted direction using one declared validation metric."""

    group_columns = ["model", "method", "fit_position"]
    required = {*group_columns, "dataset", "layer", metric}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Cannot select best layers; missing columns: {missing}")
    candidates = metrics[metrics["dataset"] == dataset].copy()
    if "phase" in candidates:
        candidates = candidates[candidates["phase"] == "layer_selection"]
    if candidates.empty:
        raise ValueError(f"No layer-selection metrics found for {dataset!r}")
    rows: list[dict] = []
    ordered = candidates.sort_values([*group_columns, "layer"], kind="stable")
    for keys, group in ordered.groupby(group_columns, sort=False):
        valid = group.dropna(subset=[metric])
        if valid.empty:
            raise RuntimeError(
                f"Layer-selection metric {metric!r} is non-finite for {dict(zip(group_columns, keys))}"
            )
        selected = valid.loc[valid[metric].idxmax()]
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "selection_dataset": dataset,
                "selection_metric": metric,
                "selected_layer": int(selected["layer"]),
                "selection_value_percent": float(selected[metric]),
                "tie_break_rule": "lowest_layer",
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


__all__ = [
    "ExperimentTables",
    "direction_similarity_rows",
    "select_layers_by_validation_metric",
]

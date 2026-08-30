"""Paper-parity result selection and table formatting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


TABLE1_METRICS = (
    ("toy_movie_review", "logit_difference", "toy_logit_diff_percent"),
    ("toy_movie_review", "logit_flip", "toy_logit_flip_percent"),
    ("sst", "logit_difference", "sst_logit_diff_percent"),
    ("sst", "logit_flip", "sst_logit_flip_percent"),
)

BEST_LAYER_COLUMNS = (
    "model",
    "method",
    "dataset",
    "metric",
    "metric_column",
    "layer",
    "value_percent",
)


def _format_table1_cell(row: pd.Series) -> str:
    value = float(row["value_percent"])
    if round(value, 1) == 0:
        value = 0.0
    return f"{value:.1f}%\n(layer {int(row['layer'])})"


def select_table1_best_layers(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each Table 1 dataset/metric maximum independently across layers.

    The paper reports the best value across layers for every method and table
    column. A method can therefore have four different selected layers. Ties
    are resolved deterministically in favor of the lower layer boundary.
    """

    metric_columns = [column for _, _, column in TABLE1_METRICS]
    required = {"model", "method", "layer", *metric_columns}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Cannot build Table 1 best layers; missing columns: {missing}")
    if metrics.empty:
        return pd.DataFrame(columns=BEST_LAYER_COLUMNS)

    ordered = metrics.sort_values(["model", "method", "layer"], kind="stable").reset_index(
        drop=True
    )
    rows: list[dict] = []
    for dataset, metric, metric_column in TABLE1_METRICS:
        missing_groups = [
            f"{model}/{method}"
            for (model, method), group in ordered.groupby(["model", "method"], sort=False)
            if not group[metric_column].notna().any()
        ]
        if missing_groups:
            raise ValueError(
                f"Cannot select {metric_column}; no non-missing result for: {missing_groups}"
            )
        valid = ordered.dropna(subset=[metric_column])
        best_indices = valid.groupby(["model", "method"], sort=False)[metric_column].idxmax()
        for _, selected in valid.loc[best_indices].iterrows():
            rows.append(
                {
                    "model": selected["model"],
                    "method": selected["method"],
                    "dataset": dataset,
                    "metric": metric,
                    "metric_column": metric_column,
                    "layer": int(selected["layer"]),
                    "value_percent": selected[metric_column],
                }
            )

    best = pd.DataFrame(rows, columns=BEST_LAYER_COLUMNS)
    if best.empty:
        return best
    dataset_order = {"toy_movie_review": 0, "sst": 1}
    metric_order = {"logit_difference": 0, "logit_flip": 1}
    best = best.assign(
        _dataset_order=best["dataset"].map(dataset_order),
        _metric_order=best["metric"].map(metric_order),
    )
    return (
        best.sort_values(["model", "method", "_dataset_order", "_metric_order"], kind="stable")
        .drop(columns=["_dataset_order", "_metric_order"])
        .reset_index(drop=True)
    )


def validate_best_layers(best_layers: pd.DataFrame) -> None:
    """Validate the long-form schema consumed by reporting functions."""

    missing = sorted(set(BEST_LAYER_COLUMNS) - set(best_layers.columns))
    if missing:
        raise ValueError(f"best_layers.csv uses an unsupported schema; missing columns: {missing}")

    expected_pairs = {(dataset, metric) for dataset, metric, _ in TABLE1_METRICS}
    actual_pairs = set(zip(best_layers["dataset"], best_layers["metric"]))
    unexpected = sorted(actual_pairs - expected_pairs)
    if unexpected:
        raise ValueError(f"best_layers.csv contains unsupported dataset/metric pairs: {unexpected}")

    duplicate_keys: Sequence[str] = ("model", "method", "dataset", "metric")
    if best_layers.duplicated(list(duplicate_keys)).any():
        raise ValueError("best_layers.csv contains duplicate model/method/dataset/metric rows")

    incomplete = []
    for (model, method), group in best_layers.groupby(["model", "method"], sort=False):
        group_pairs = set(zip(group["dataset"], group["metric"]))
        if group_pairs != expected_pairs:
            incomplete.append(f"{model}/{method}")
    if incomplete:
        raise ValueError(f"best_layers.csv is missing Table 1 cells for: {incomplete}")


def table1_cell_text(best_layers: pd.DataFrame) -> pd.DataFrame:
    """Return a compact method-by-metric table with values and selected layers."""

    validate_best_layers(best_layers)
    if best_layers.empty:
        return pd.DataFrame()
    if best_layers["model"].nunique() != 1:
        raise ValueError("A paper-style result table must contain exactly one model")

    labels = {
        ("toy_movie_review", "logit_difference"): "ToyMovieReview\nlogit difference",
        ("toy_movie_review", "logit_flip"): "ToyMovieReview\nlogit flip",
        ("sst", "logit_difference"): "SST\nlogit difference",
        ("sst", "logit_flip"): "SST\nlogit flip",
    }
    formatted = best_layers.copy()
    formatted["column"] = [
        labels[(dataset, metric)]
        for dataset, metric in zip(formatted["dataset"], formatted["metric"])
    ]
    formatted["cell"] = formatted.apply(_format_table1_cell, axis=1)
    table = formatted.pivot(index="method", columns="column", values="cell")
    ordered_columns = list(labels.values())
    return table.reindex(columns=ordered_columns)

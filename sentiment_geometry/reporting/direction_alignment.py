"""Plots for frozen sentiment-versus-valence direction geometry."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


_METHOD_LABELS = {"mean_diff": "Mean difference", "das": "DAS (1D)"}
_COMPARISON_LABELS = {
    "within_sentiment": "Sentiment directions",
    "within_valence": "Valence directions",
    "valence_vs_sentiment": "Valence versus sentiment directions",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _validate_tables(similarities: pd.DataFrame, selections: pd.DataFrame) -> None:
    similarity_columns = {
        "comparison",
        "model",
        "row_representation",
        "row_method",
        "row_layer",
        "column_representation",
        "column_method",
        "column_layer",
        "signed_cosine",
        "absolute_cosine",
    }
    selection_columns = {"representation", "model", "method", "selected_layer"}
    if not similarity_columns <= set(similarities):
        raise ValueError(
            "Direction similarities are missing columns: "
            f"{sorted(similarity_columns - set(similarities))}"
        )
    if not selection_columns <= set(selections):
        raise ValueError(
            "Selected directions are missing columns: "
            f"{sorted(selection_columns - set(selections))}"
        )


def _axis_labels(
    selections: pd.DataFrame,
    *,
    model: str,
    representation: str,
    methods: Sequence[str],
) -> list[str]:
    subset = selections[
        (selections["model"] == model)
        & (selections["representation"] == representation)
    ]
    layer_by_method = {
        str(row.method): int(row.selected_layer)
        for row in subset.itertuples(index=False)
    }
    if set(layer_by_method) != set(methods):
        raise ValueError(
            f"Incomplete selected layers for {model}/{representation}: {layer_by_method}"
        )
    return [
        f"{_METHOD_LABELS.get(method, method)}\nL{layer_by_method[method]:02d}"
        for method in methods
    ]


def plot_direction_alignment(
    similarities: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    figure_dir: str | Path,
    model_names: Sequence[str],
    methods: Sequence[str] = ("mean_diff", "das"),
) -> list[Path]:
    """Save signed and absolute heatmaps for all three alignment comparisons."""

    _validate_tables(similarities, selections)
    models = tuple(dict.fromkeys(str(name) for name in model_names))
    ordered_methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not models or not ordered_methods:
        raise ValueError("At least one model and method are required for alignment plots")
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    outputs: list[Path] = []

    for comparison, comparison_title in _COMPARISON_LABELS.items():
        subset = similarities[similarities["comparison"] == comparison]
        expected_rows = len(models) * len(ordered_methods) ** 2
        if len(subset) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows for {comparison}; found {len(subset)}"
            )
        row_representation = (
            "valence" if comparison == "valence_vs_sentiment" else comparison.removeprefix("within_")
        )
        column_representation = (
            "sentiment" if comparison == "valence_vs_sentiment" else row_representation
        )
        for metric, metric_title, cmap, minimum, maximum, center in (
            ("signed_cosine", "Signed cosine", "vlag", -1.0, 1.0, 0.0),
            ("absolute_cosine", "Absolute cosine", "Reds", 0.0, 1.0, None),
        ):
            figure, axes = plt.subplots(
                1,
                len(models),
                figsize=(6.0 * len(models), 5.0),
                squeeze=False,
            )
            for index, model in enumerate(models):
                axis = axes[0, index]
                model_rows = subset[subset["model"] == model]
                table = model_rows.pivot(
                    index="row_method", columns="column_method", values=metric
                ).reindex(index=ordered_methods, columns=ordered_methods)
                if table.isna().any().any():
                    raise ValueError(f"Incomplete {comparison}/{metric} matrix for {model}")
                table.index = _axis_labels(
                    selections,
                    model=model,
                    representation=row_representation,
                    methods=ordered_methods,
                )
                table.columns = _axis_labels(
                    selections,
                    model=model,
                    representation=column_representation,
                    methods=ordered_methods,
                )
                sns.heatmap(
                    table,
                    vmin=minimum,
                    vmax=maximum,
                    center=center,
                    cmap=cmap,
                    annot=True,
                    fmt=".3f",
                    square=True,
                    linewidths=0.5,
                    cbar=index == len(models) - 1,
                    ax=axis,
                )
                axis.set_title(model)
                axis.set_xlabel(column_representation.title())
                axis.set_ylabel(row_representation.title())
            figure.suptitle(f"{comparison_title} — {metric_title}", y=1.02)
            figure.tight_layout()
            path = figure_dir / f"{_slug(metric)}_{_slug(comparison)}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            outputs.append(path)
    return outputs


__all__ = ["plot_direction_alignment"]

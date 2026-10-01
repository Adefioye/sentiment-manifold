"""Plots for cross-run frozen-direction geometry."""

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
_MODEL_LABELS = {
    "gpt2-small": "GPT-2 Small",
    "qwen-0.6b": "Qwen3-0.6B Base",
}
_COMPARISON_LABELS = {
    "within_sentiment": "Sentiment directions",
    "within_valence": "Valence directions",
    "valence_vs_sentiment": "Valence versus sentiment directions",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _validate_similarity_table(similarities: pd.DataFrame) -> None:
    similarity_columns = {
        "comparison",
        "model",
        "row_representation",
        "row_method",
        "column_representation",
        "column_method",
        "absolute_cosine",
    }
    if not similarity_columns <= set(similarities):
        raise ValueError(
            "Direction similarities are missing columns: "
            f"{sorted(similarity_columns - set(similarities))}"
        )


def plot_direction_alignment(
    similarities: pd.DataFrame,
    *,
    figure_dir: str | Path,
    model_names: Sequence[str],
    methods: Sequence[str] = ("mean_diff", "das"),
) -> list[Path]:
    """Save absolute-cosine heatmaps with method-only labels."""

    _validate_similarity_table(similarities)
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
        figure, axes = plt.subplots(
            1,
            len(models),
            figsize=(6.0 * len(models), 5.0),
            squeeze=False,
        )
        method_labels = [_METHOD_LABELS.get(method, method) for method in ordered_methods]
        for index, model in enumerate(models):
            axis = axes[0, index]
            model_rows = subset[subset["model"] == model]
            table = model_rows.pivot(
                index="row_method", columns="column_method", values="absolute_cosine"
            ).reindex(index=ordered_methods, columns=ordered_methods)
            if table.isna().any().any():
                raise ValueError(
                    f"Incomplete {comparison}/absolute_cosine matrix for {model}"
                )
            table.index = method_labels
            table.columns = method_labels
            sns.heatmap(
                table,
                vmin=0.0,
                vmax=1.0,
                cmap="Reds",
                annot=True,
                fmt=".3f",
                square=True,
                linewidths=0.5,
                cbar=index == len(models) - 1,
                ax=axis,
            )
            axis.set_title(_MODEL_LABELS.get(model, model))
            axis.set_xlabel(column_representation.title())
            axis.set_ylabel(row_representation.title())
        figure.suptitle(f"{comparison_title} — Cosine alignment", y=1.02)
        figure.tight_layout()
        path = figure_dir / f"cosine_alignment_{_slug(comparison)}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_pairwise_absolute_direction_alignment(
    absolute_cosines: pd.DataFrame,
    *,
    figure_dir: str | Path,
    model_names: Sequence[str],
    methods: Sequence[str] = ("mean_diff", "das"),
    row_label: str,
    column_label: str,
) -> Path:
    """Save one absolute-cosine heatmap with method-only labels for each model."""

    required = {
        "model",
        "row_source",
        "row_method",
        "column_source",
        "column_method",
        "absolute_cosine",
    }
    missing = required - set(absolute_cosines)
    if missing:
        raise ValueError(f"Pairwise cosine table is missing columns: {sorted(missing)}")
    models = tuple(dict.fromkeys(str(name) for name in model_names))
    ordered_methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not models or not ordered_methods:
        raise ValueError("At least one model and method are required for alignment plots")
    if not row_label.strip() or not column_label.strip():
        raise ValueError("Pairwise alignment row and column labels cannot be empty")
    expected_rows = len(models) * len(ordered_methods) ** 2
    if len(absolute_cosines) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} pairwise cosine rows; found {len(absolute_cosines)}"
        )

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.5 * len(models), 4.75),
        squeeze=False,
    )
    method_labels = [_METHOD_LABELS.get(method, method) for method in ordered_methods]
    for index, model in enumerate(models):
        axis = axes[0, index]
        model_rows = absolute_cosines[absolute_cosines["model"] == model]
        table = model_rows.pivot(
            index="row_method",
            columns="column_method",
            values="absolute_cosine",
        ).reindex(index=ordered_methods, columns=ordered_methods)
        if table.isna().any().any():
            raise ValueError(f"Incomplete pairwise absolute-cosine matrix for {model}")
        table.index = method_labels
        table.columns = method_labels
        sns.heatmap(
            table,
            vmin=0.0,
            vmax=1.0,
            cmap="Reds",
            annot=True,
            fmt=".3f",
            square=True,
            linewidths=0.5,
            cbar=index == len(models) - 1,
            ax=axis,
        )
        axis.set_title(_MODEL_LABELS.get(model, model))
        axis.set_xlabel(column_label)
        axis.set_ylabel(row_label)
    figure.suptitle(f"Cosine alignment: {row_label} versus {column_label}", y=1.02)
    figure.tight_layout()
    path = figure_dir / (
        f"absolute_cosine_{_slug(row_label.lower())}_vs_{_slug(column_label.lower())}.png"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = ["plot_direction_alignment", "plot_pairwise_absolute_direction_alignment"]

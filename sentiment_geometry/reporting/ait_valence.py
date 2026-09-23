"""Plots for completed AIT valence-direction experiments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class AITValenceReportData:
    """Saved AIT result tables combined from explicit per-model directories."""

    dataset_summary: pd.DataFrame
    metrics: pd.DataFrame
    final_metrics: pd.DataFrame
    direction_metadata: pd.DataFrame
    direction_similarities: pd.DataFrame
    layer_selection: pd.DataFrame


def _validated_model_names(model_names: Sequence[str]) -> tuple[str, ...]:
    requested_models = tuple(dict.fromkeys(str(name) for name in model_names))
    if not requested_models:
        raise ValueError("At least one AIT report model is required")
    if len(requested_models) != len(model_names):
        raise ValueError("AIT report model names must be unique")
    return requested_models


def _load_model_table(results_dir: Path, model_names: Sequence[str], name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name in model_names:
        path = results_dir / model_name / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            raise ValueError(f"AIT {path} is missing the model column")
        observed_models = set(frame["model"].astype(str))
        if observed_models != {model_name}:
            raise ValueError(
                f"AIT {path} contains models {sorted(observed_models)}, "
                f"expected only {model_name!r}"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_ait_valence_report_data(
    results_dir: str | Path,
    *,
    model_names: Sequence[str],
) -> AITValenceReportData:
    """Load a coherent multi-model report from durable per-model AIT CSVs."""

    requested_models = _validated_model_names(model_names)
    results_dir = Path(results_dir)
    return AITValenceReportData(
        dataset_summary=_load_model_table(results_dir, requested_models, "dataset_summary"),
        metrics=_load_model_table(results_dir, requested_models, "metrics"),
        final_metrics=_load_model_table(results_dir, requested_models, "final_metrics"),
        direction_metadata=_load_model_table(
            results_dir, requested_models, "direction_metadata"
        ),
        direction_similarities=_load_model_table(
            results_dir, requested_models, "direction_similarities"
        ),
        layer_selection=_load_model_table(results_dir, requested_models, "layer_selection"),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def plot_ait_valence_run(
    results_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
    model_names: Sequence[str] | None = None,
) -> list[Path]:
    """Render AIT layer curves and snapshot similarities from saved CSVs only."""

    results_dir = Path(results_dir)
    figure_dir = Path(figure_dir) if figure_dir is not None else results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if model_names is None:
        metrics_path = results_dir / "all_models_metrics.csv"
        similarities_path = results_dir / "all_models_direction_similarities.csv"
        selection_path = results_dir / "all_models_layer_selection.csv"
        for path in (metrics_path, similarities_path, selection_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = pd.read_csv(metrics_path)
        similarities = pd.read_csv(similarities_path)
        selection = pd.read_csv(selection_path)
    else:
        requested_models = _validated_model_names(model_names)
        metrics = _load_model_table(results_dir, requested_models, "metrics")
        similarities = _load_model_table(
            results_dir, requested_models, "direction_similarities"
        )
        selection = _load_model_table(results_dir, requested_models, "layer_selection")
    required_metrics = {
        "model",
        "method",
        "layer",
        "phase",
        "logit_flip_percent",
        "sign_flip_percent",
    }
    if not required_metrics <= set(metrics):
        missing = sorted(required_metrics - set(metrics))
        raise ValueError(f"AIT metrics are missing columns: {missing}")
    required_similarities = {
        "model",
        "layer",
        "method_a",
        "method_b",
        "absolute_cosine",
    }
    if not required_similarities <= set(similarities):
        missing = sorted(required_similarities - set(similarities))
        raise ValueError(f"AIT similarities are missing columns: {missing}")

    sns.set_theme(style="whitegrid")
    outputs: list[Path] = []
    layer_metrics = metrics[metrics["phase"] == "layer_selection"].copy()
    for metric in ("logit_flip_percent", "sign_flip_percent"):
        grid = sns.relplot(
            data=layer_metrics,
            x="layer",
            y=metric,
            hue="method",
            col="model",
            col_wrap=2,
            kind="line",
            marker="o",
            facet_kws={"sharey": False},
            height=4,
            aspect=1.25,
        )
        grid.set_axis_labels("Residual boundary", "Percent")
        grid.figure.suptitle(metric.replace("_", " ").title(), y=1.02)
        path = figure_dir / f"{metric}_by_layer.png"
        grid.figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(grid.figure)
        outputs.append(path)

    required_selection = {
        "model",
        "method",
        "selected_layer",
        "selection_value_percent",
    }
    if not required_selection <= set(selection):
        missing = sorted(required_selection - set(selection))
        raise ValueError(f"AIT layer selection is missing columns: {missing}")
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=selection,
        x="model",
        y="selection_value_percent",
        hue="method",
        ax=axis,
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("Selected-layer logit flip percent")
    selection_roles = sorted(set(selection.get("selection_role", ["configured"])))
    selection_label = "/".join(str(role) for role in selection_roles)
    axis.set_title(f"AIT last-token directions selected by {selection_label}-role logit flip")
    figure.tight_layout()
    path = figure_dir / "selected_layer_logit_flip_percent.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path)

    for (model, layer), subset in similarities.groupby(["model", "layer"], sort=False):
        table = subset.pivot(
            index="method_a",
            columns="method_b",
            values="absolute_cosine",
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            table,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            square=True,
            ax=axis,
        )
        axis.set_title(f"{model}: absolute direction cosine — boundary {int(layer)}")
        figure.tight_layout()
        path = figure_dir / (f"absolute_cosine_{_slug(str(model))}_boundary{int(layer):02d}.png")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


__all__ = [
    "AITValenceReportData",
    "load_ait_valence_report_data",
    "plot_ait_valence_run",
]

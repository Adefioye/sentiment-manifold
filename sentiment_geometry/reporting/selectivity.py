"""Plots reconstructed from fixed-layer selectivity result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_fixed_layer_selectivity(run_dir: str | Path) -> tuple[Path, ...]:
    root = Path(run_dir)
    metrics = pd.read_csv(root / "metrics.csv")
    summary = pd.read_csv(root / "selectivity_summary.csv")
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: list[Path] = []

    held_out = metrics[metrics["split"].isin(["validation", "test"])].copy()
    melted = held_out.melt(
        id_vars=["model", "dataset", "method", "task", "seed", "split"],
        value_vars=["native_balanced_accuracy", "midpoint_balanced_accuracy"],
        var_name="decision_rule",
        value_name="balanced_accuracy",
    )
    grid = sns.catplot(
        data=melted,
        x="method",
        y="balanced_accuracy",
        hue="task",
        col="dataset",
        row="split",
        kind="bar",
        errorbar="sd",
        height=3.2,
        aspect=1.45,
    )
    grid.set_axis_labels("Method", "Balanced accuracy")
    grid.set_xticklabels(rotation=25)
    grid.figure.suptitle("Real versus random-label accuracy", y=1.02)
    path = figure_dir / "real_vs_random_accuracy.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)

    selected = summary[
        (summary["split"].isin(["validation", "test"]))
        & (summary["metric"].isin(["native_balanced_accuracy", "midpoint_balanced_accuracy"]))
    ].copy()
    grid = sns.catplot(
        data=selected,
        x="method",
        y="selectivity_mean",
        hue="metric",
        col="dataset",
        row="split",
        kind="bar",
        height=3.2,
        aspect=1.45,
    )
    grid.set_axis_labels("Method", "Real − random balanced accuracy")
    grid.set_xticklabels(rotation=25)
    grid.figure.suptitle("Paired random-label selectivity", y=1.02)
    path = figure_dir / "selectivity.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)

    real = held_out[held_out.task == "real"]
    grid = sns.relplot(
        data=real,
        x="native_balanced_accuracy",
        y="midpoint_balanced_accuracy",
        hue="method",
        style="model",
        col="dataset",
        row="split",
        kind="scatter",
        height=3.2,
        aspect=1.25,
    )
    for axis in grid.axes.flat:
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    grid.figure.suptitle("Native versus midpoint accuracy", y=1.02)
    path = figure_dir / "native_vs_midpoint.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)
    return tuple(paths)


__all__ = ["plot_fixed_layer_selectivity"]

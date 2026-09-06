"""Publication-oriented summary plots from saved CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .table1 import select_table1_best_layers, table1_cell_text, validate_best_layers


def _load_best_layers(run_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    """Load the current schema, deriving it in memory for legacy run folders."""

    path = run_dir / "best_layers.csv"
    if path.exists():
        best = pd.read_csv(path)
        try:
            validate_best_layers(best)
        except ValueError:
            # Older runs stored one complete SST-recovery-selected metrics row
            # per method. Recompute the paper table from their metrics without
            # mutating the archived run.
            return select_table1_best_layers(metrics)
        return best
    return select_table1_best_layers(metrics)


def _plot_table1_results(best: pd.DataFrame, figure_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for model, model_best in best.groupby("model", sort=False):
        table = table1_cell_text(model_best)
        if table.empty:
            continue
        figure_width = max(10.0, 2.4 * len(table.columns))
        figure_height = max(3.0, 0.55 * len(table.index) + 1.8)
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        axis.axis("off")
        rendered = axis.table(
            cellText=table.values,
            rowLabels=[str(method).replace("_", " ") for method in table.index],
            colLabels=table.columns,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(9)
        rendered.scale(1.0, 1.65)
        axis.set_title(f"Table 1 best-across-layer results — {model}", pad=20)
        figure.tight_layout()
        suffix = "" if best["model"].nunique() == 1 else f"_{model}"
        path = figure_dir / f"table1_best_results{suffix}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    if {"fit_position", "dataset", "logit_difference_percent"} <= set(metrics.columns):
        return plot_sentiment_position_comparison(run_dir, metrics=metrics)
    best = _load_best_layers(run_dir, metrics)
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    for metric in (
        "toy_logit_diff_percent",
        "toy_logit_flip_percent",
        "sst_logit_diff_percent",
        "sst_logit_flip_percent",
    ):
        if metric not in metrics:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=metrics, x="layer", y=metric, hue="method", marker="o", ax=axis)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("Percent (%)")
        figure.tight_layout()
        path = figure_dir / f"{metric}_by_layer.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    outputs.extend(_plot_table1_results(best, figure_dir))

    loss_path = run_dir / "das_losses.csv"
    if loss_path.exists():
        losses = pd.read_csv(loss_path)
        required_loss_columns = {"epoch", "evaluation_loss", "layer", "method"}
        if not losses.empty and required_loss_columns <= set(losses):
            grid = sns.relplot(
                data=losses,
                x="epoch",
                y="evaluation_loss",
                hue="method",
                col="layer",
                col_wrap=4,
                kind="line",
                marker="o",
                facet_kws={"sharey": False},
            )
            grid.set_axis_labels("Epoch", "Normalized logit-difference loss")
            grid.figure.suptitle("DAS training loss by layer", y=1.02)
            path = figure_dir / "das_loss_by_epoch.png"
            grid.figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(grid.figure)
            outputs.append(path)

    similarity_path = run_dir / "direction_similarities.csv"
    if not similarity_path.exists():
        return outputs
    similarities = pd.read_csv(similarity_path)
    for layer in sorted(best.layer.unique()):
        subset = similarities[similarities.layer == layer]
        table = subset.pivot(index="method_a", columns="method_b", values="absolute_cosine")
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(table, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="Reds", ax=axis)
        axis.set_title(f"Direction similarity — layer {layer}")
        figure.tight_layout()
        path = figure_dir / f"similarity_layer{int(layer):02d}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    return outputs


def plot_sentiment_position_comparison(
    run_dir: str | Path, *, metrics: pd.DataFrame | None = None
) -> list[Path]:
    """Render every RQ2 Question 1 figure from saved CSV tables only."""

    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = pd.read_csv(run_dir / "metrics.csv")
    required = {"model", "method", "fit_position", "layer", "dataset"}
    if not required <= set(metrics.columns):
        raise ValueError("Run directory does not contain RQ2 Question 1 metrics")
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    for metric in ("logit_difference_percent", "logit_flip_percent"):
        grid = sns.relplot(
            data=metrics,
            x="layer",
            y=metric,
            hue="method",
            style="fit_position",
            col="dataset",
            col_wrap=2,
            kind="line",
            marker="o",
            facet_kws={"sharey": False},
            height=4,
            aspect=1.25,
        )
        grid.set_axis_labels("Residual boundary", "Percent (%)")
        grid.figure.suptitle(metric.replace("_", " ").title(), y=1.02)
        path = figure_dir / f"{metric}_by_layer.png"
        grid.figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(grid.figure)
        outputs.append(path)

    best_path = run_dir / "best_layers.csv"
    if best_path.exists():
        best = pd.read_csv(best_path)
        for metric, subset in best.groupby("metric", sort=False):
            plot_data = subset.copy()
            plot_data["method_position"] = (
                plot_data["method"].str.replace("_", " ") + " / " + plot_data["fit_position"]
            )
            grid = sns.catplot(
                data=plot_data,
                x="dataset",
                y="value_percent",
                hue="method_position",
                kind="bar",
                height=5,
                aspect=1.8,
            )
            grid.set_axis_labels("Evaluation dataset", "Best value across boundaries (%)")
            grid.figure.suptitle(str(metric).replace("_", " ").title(), y=1.02)
            path = figure_dir / f"best_{metric}.png"
            grid.figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(grid.figure)
            outputs.append(path)

    similarity_path = run_dir / "direction_similarities.csv"
    if similarity_path.exists():
        similarities = pd.read_csv(similarity_path)
        for layer in sorted(similarities["layer"].unique()):
            subset = similarities[similarities["layer"] == layer]
            table = subset.pivot(
                index="direction_a", columns="direction_b", values="absolute_cosine"
            )
            figure, axis = plt.subplots(figsize=(9, 7))
            sns.heatmap(table, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="Reds", ax=axis)
            axis.set_title(f"Absolute direction cosine — boundary {int(layer)}")
            figure.tight_layout()
            path = figure_dir / f"absolute_cosine_boundary{int(layer):02d}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            outputs.append(path)

    loss_path = run_dir / "das_losses.csv"
    if loss_path.exists():
        losses = pd.read_csv(loss_path)
        if not losses.empty:
            grid = sns.relplot(
                data=losses,
                x="epoch",
                y="evaluation_loss",
                hue="fit_position",
                col="layer",
                col_wrap=4,
                kind="line",
                facet_kws={"sharey": False},
            )
            grid.set_axis_labels("Epoch", "Normalized logit-difference loss")
            path = figure_dir / "das_loss_by_position_and_layer.png"
            grid.figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(grid.figure)
            outputs.append(path)
    return outputs

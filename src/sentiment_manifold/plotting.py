"""Publication-oriented summary plots from saved CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .reporting import select_table1_best_layers, table1_cell_text, validate_best_layers


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

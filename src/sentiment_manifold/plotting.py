"""Publication-oriented summary plots from saved CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
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
    best_path = run_dir / "best_layers.csv"
    if not similarity_path.exists() or not best_path.exists():
        return outputs
    similarities = pd.read_csv(similarity_path)
    best = pd.read_csv(best_path)
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

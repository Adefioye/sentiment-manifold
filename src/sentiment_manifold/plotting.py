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

    for metric in ("toy_patch_recovery", "sst_patch_recovery", "sst_flip_rate"):
        if metric not in metrics:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=metrics, x="layer", y=metric, hue="method", marker="o", ax=axis)
        axis.set_title(metric.replace("_", " ").title())
        figure.tight_layout()
        path = figure_dir / f"{metric}_by_layer.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    similarities = pd.read_csv(run_dir / "direction_similarities.csv")
    best = pd.read_csv(run_dir / "best_layers.csv")
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

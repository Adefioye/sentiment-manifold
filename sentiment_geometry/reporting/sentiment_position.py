"""Read-only tables and figures for sentiment fitting-position results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

MODEL_ORDER = ("gpt2-small", "qwen-0.6b")
POSITION_ORDER = ("adjective", "final")
METHOD_ORDER = ("mean_diff", "logistic_regression", "das")
DATASET_ORDER = ("toy_adjectives", "toy_verbs", "toy_adverbs", "sst")

MODEL_LABELS = {
    "gpt2-small": "GPT-2 Small",
    "qwen-0.6b": "Qwen3-0.6B Base",
}
POSITION_LABELS = {
    "adjective": "Adjective position",
    "final": "Final prompt-token position",
}
METHOD_LABELS = {
    "mean_diff": "Mean difference",
    "logistic_regression": "Logistic regression",
    "das": "DAS",
}
DATASET_LABELS = {
    "toy_adjectives": "Toy adjectives",
    "toy_verbs": "Toy verbs",
    "toy_adverbs": "Toy adverbs",
    "sst": "SST",
}
METRIC_LABELS = {
    "logit_difference": "Logit difference",
    "logit_flip": "Logit flip",
}
METHOD_COLORS = {
    "mean_diff": "#0072B2",
    "logistic_regression": "#D55E00",
    "das": "#009E73",
}


def _require_columns(table: pd.DataFrame, required: set[str], *, table_name: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _load_model_table(results_dir: Path, models: tuple[str, ...], filename: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for model in models:
        path = results_dir / model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        if "model" not in table.columns:
            table.insert(0, "model", model)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


@dataclass(frozen=True)
class SentimentPositionReportData:
    """Analysis-ready tables loaded without modifying an experiment run."""

    results_dir: Path
    models: tuple[str, ...]
    metrics: pd.DataFrame
    best_layers: pd.DataFrame
    direction_similarities: pd.DataFrame
    dataset_summary: pd.DataFrame

    @classmethod
    def load(
        cls,
        results_dir: str | Path,
        *,
        models: tuple[str, ...] = MODEL_ORDER,
    ) -> SentimentPositionReportData:
        results_dir = Path(results_dir)
        metrics = _load_model_table(results_dir, models, "metrics.csv")
        best_layers = _load_model_table(results_dir, models, "best_layers.csv")
        direction_similarities = _load_model_table(
            results_dir, models, "direction_similarities.csv"
        )
        dataset_summary = _load_model_table(results_dir, models, "dataset_summary.csv")
        _require_columns(
            metrics,
            {
                "model",
                "method",
                "fit_position",
                "layer",
                "dataset",
                "logit_difference_percent",
                "logit_flip_percent",
            },
            table_name="metrics.csv",
        )
        _require_columns(
            best_layers,
            {"model", "method", "fit_position", "dataset", "metric", "layer", "value_percent"},
            table_name="best_layers.csv",
        )
        _require_columns(
            direction_similarities,
            {
                "model",
                "layer",
                "position_a",
                "method_a",
                "position_b",
                "method_b",
                "signed_cosine",
                "absolute_cosine",
            },
            table_name="direction_similarities.csv",
        )
        return cls(
            results_dir=results_dir,
            models=models,
            metrics=metrics,
            best_layers=best_layers,
            direction_similarities=direction_similarities,
            dataset_summary=dataset_summary,
        )

    @property
    def evaluation_datasets(self) -> tuple[str, ...]:
        present = set(self.metrics["dataset"])
        return tuple(dataset for dataset in DATASET_ORDER if dataset in present)

    @property
    def has_training_causal_metrics(self) -> bool:
        return "toy_train" in set(self.metrics["dataset"])


def figure4_style_table(
    best_layers: pd.DataFrame,
    *,
    model: str,
    fit_position: str,
    datasets: tuple[str, ...] = DATASET_ORDER,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> pd.DataFrame:
    """Format best-across-layer causal metrics in the style of Tigges et al. Figure 4."""

    subset = best_layers[
        (best_layers["model"] == model) & (best_layers["fit_position"] == fit_position)
    ].copy()
    expected = {
        (method, dataset, metric)
        for method in methods
        for dataset in datasets
        for metric in METRIC_LABELS
    }
    actual = set(zip(subset["method"], subset["dataset"], subset["metric"]))
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            f"Best-layer results are incomplete for model={model}, position={fit_position}: {missing}"
        )
    if subset.duplicated(["method", "dataset", "metric"]).any():
        raise ValueError(f"Best-layer results contain duplicate cells for {model}/{fit_position}")

    indexed = subset.set_index(["method", "dataset", "metric"])
    columns = [
        f"{DATASET_LABELS.get(dataset, dataset)}\n{METRIC_LABELS[metric]}"
        for dataset in datasets
        for metric in METRIC_LABELS
    ]
    rows: list[list[str]] = []
    for method in methods:
        cells: list[str] = []
        for dataset in datasets:
            for metric in METRIC_LABELS:
                result = indexed.loc[(method, dataset, metric)]
                cells.append(f"{float(result['value_percent']):.1f}%\n(L{int(result['layer']):02d})")
        rows.append(cells)
    return pd.DataFrame(
        rows,
        index=[METHOD_LABELS.get(method, method) for method in methods],
        columns=columns,
    )


def plot_cosine_similarity_grid(
    similarities: pd.DataFrame,
    *,
    model: str,
    positions: tuple[str, ...] = POSITION_ORDER,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> Figure:
    """Plot within-position method cosines at the saved comparison boundaries."""

    model_data = similarities[similarities["model"] == model]
    layers = tuple(sorted(int(layer) for layer in model_data["layer"].unique()))
    if not layers:
        raise ValueError(f"No direction similarities are available for {model}")
    sns.set_theme(style="white", context="notebook")
    figure, axes = plt.subplots(
        len(positions),
        len(layers),
        figsize=(4.2 * len(layers), 3.8 * len(positions)),
        squeeze=False,
    )
    colorbar_axis = figure.add_axes((0.92, 0.16, 0.018, 0.68))
    for row, position in enumerate(positions):
        for column, layer in enumerate(layers):
            axis = axes[row, column]
            subset = model_data[
                (model_data["layer"] == layer)
                & (model_data["position_a"] == position)
                & (model_data["position_b"] == position)
                & model_data["method_a"].isin(methods)
                & model_data["method_b"].isin(methods)
            ]
            matrix = subset.pivot(
                index="method_a", columns="method_b", values="absolute_cosine"
            ).reindex(index=methods, columns=methods)
            if matrix.isna().any().any():
                raise ValueError(f"Incomplete cosine matrix for {model}/{position}/layer {layer}")
            matrix.index = [METHOD_LABELS[method] for method in methods]
            matrix.columns = [METHOD_LABELS[method] for method in methods]
            draw_colorbar = row == 0 and column == len(layers) - 1
            sns.heatmap(
                matrix,
                vmin=0,
                vmax=1,
                cmap="mako",
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.8,
                linecolor="white",
                cbar=draw_colorbar,
                cbar_ax=colorbar_axis if draw_colorbar else None,
                ax=axis,
            )
            axis.set_title(f"Boundary {layer}")
            axis.set_xlabel("")
            axis.set_ylabel(POSITION_LABELS.get(position, position) if column == 0 else "")
            axis.tick_params(axis="x", rotation=35)
            axis.tick_params(axis="y", rotation=0)
    colorbar_axis.set_ylabel("Absolute cosine", rotation=270, labelpad=18)
    figure.suptitle(
        f"{MODEL_LABELS.get(model, model)} — similarity among fitting methods",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(left=0.13, right=0.89, bottom=0.12, top=0.88, wspace=0.35, hspace=0.35)
    return figure


def plot_cross_position_cosines(
    similarities: pd.DataFrame,
    *,
    model: str,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> Figure:
    """Plot adjective-versus-final cosine for each method at comparison boundaries."""

    subset = similarities[
        (similarities["model"] == model)
        & (similarities["position_a"] == "adjective")
        & (similarities["position_b"] == "final")
        & (similarities["method_a"] == similarities["method_b"])
        & similarities["method_a"].isin(methods)
    ].copy()
    if subset.empty:
        raise ValueError(f"No adjective-versus-final similarities are available for {model}")
    subset["method_label"] = subset["method_a"].map(METHOD_LABELS)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=subset,
        x="layer",
        y="absolute_cosine",
        hue="method_label",
        palette={METHOD_LABELS[method]: METHOD_COLORS[method] for method in methods},
        marker="o",
        linewidth=2.4,
        markersize=8,
        ax=axis,
    )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Residual boundary")
    axis.set_ylabel("Absolute cosine")
    axis.set_title(
        f"{MODEL_LABELS.get(model, model)} — adjective vs. final-token directions",
        fontweight="bold",
    )
    axis.legend(title="Fitting method", frameon=True)
    figure.tight_layout()
    return figure


def plot_logit_difference_grid(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    models: tuple[str, ...] = MODEL_ORDER,
    positions: tuple[str, ...] = POSITION_ORDER,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> Figure:
    """Plot a model-by-position grid of logit-difference recovery across layers."""

    subset = metrics[
        (metrics["dataset"] == dataset)
        & metrics["model"].isin(models)
        & metrics["fit_position"].isin(positions)
        & metrics["method"].isin(methods)
    ].copy()
    if subset.empty:
        raise ValueError(f"No logit-difference results are available for {dataset}")
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(
        len(models),
        len(positions),
        figsize=(13, 8.5),
        sharey=True,
        squeeze=False,
    )
    for row, model in enumerate(models):
        for column, position in enumerate(positions):
            axis = axes[row, column]
            panel = subset[
                (subset["model"] == model) & (subset["fit_position"] == position)
            ].copy()
            expected_methods = set(methods)
            if set(panel["method"]) != expected_methods:
                raise ValueError(f"Incomplete methods for {dataset}/{model}/{position}")
            sns.lineplot(
                data=panel,
                x="layer",
                y="logit_difference_percent",
                hue="method",
                hue_order=methods,
                palette=METHOD_COLORS,
                linewidth=2.3,
                marker="o",
                markersize=5,
                legend=row == 0 and column == 0,
                ax=axis,
            )
            for method in methods:
                method_rows = panel[panel["method"] == method]
                best = method_rows.loc[method_rows["logit_difference_percent"].idxmax()]
                axis.scatter(
                    [best["layer"]],
                    [best["logit_difference_percent"]],
                    marker="*",
                    s=130,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.axhline(100, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)
            axis.set_title(POSITION_LABELS.get(position, position), fontweight="bold")
            axis.set_xlabel("Residual boundary" if row == len(models) - 1 else "")
            axis.set_ylabel(
                f"{MODEL_LABELS.get(model, model)}\nLogit difference (%)" if column == 0 else ""
            )
            if row == 0 and column == 0:
                handles, labels = axis.get_legend_handles_labels()
                axis.get_legend().remove()
    figure.legend(
        handles,
        [METHOD_LABELS.get(label, label) for label in labels],
        title="Fitting method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(methods),
        frameon=False,
    )
    figure.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)} — directional-patching recovery",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(top=0.84, hspace=0.28, wspace=0.12)
    return figure


__all__ = [
    "DATASET_ORDER",
    "MODEL_ORDER",
    "SentimentPositionReportData",
    "figure4_style_table",
    "plot_cosine_similarity_grid",
    "plot_cross_position_cosines",
    "plot_logit_difference_grid",
]

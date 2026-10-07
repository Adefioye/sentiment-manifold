"""Plots reconstructed from fixed-layer selectivity result tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..persistence import RunArtifactStore

SELECTIVITY_METHODS = ("mean_diff", "logistic_regression", "das", "mlp1")
SELECTIVITY_MODELS = ("gpt2-small", "qwen-0.6b")
SELECTIVITY_TABLES = (
    "metrics",
    "predictions",
    "causal_metrics",
    "patching_records",
    "random_label_assignments",
    "random_pair_manifest",
    "sample_manifest",
    "pair_manifest",
    "fit_metadata",
    "tuning_trials",
    "training_history",
    "runtime",
    "selectivity_summary",
)


@dataclass(frozen=True)
class FixedLayerSelectivityReport:
    tuning_trials: pd.DataFrame
    performance: pd.DataFrame
    training_memorization: pd.DataFrame
    native_midpoint_comparison: pd.DataFrame
    selectivity: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    das_causal_metrics: pd.DataFrame
    training_history: pd.DataFrame


def _read_optional_csv(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _mean_std_summary(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    available = [value for value in values if value in frame.columns]
    grouped = frame.groupby(keys, dropna=False)[available].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    result = grouped.reset_index()
    result["n_seeds"] = (
        frame.groupby(keys, dropna=False)["seed"].nunique().to_numpy()
    )
    std_columns = [column for column in result if column.endswith("_std")]
    result[std_columns] = result[std_columns].fillna(0.0)
    return result


def load_fixed_layer_selectivity_report(
    run_dir: str | Path,
) -> FixedLayerSelectivityReport:
    """Load concise common and method-specific tables for one completed run."""

    root = Path(run_dir)
    metrics = pd.read_csv(root / "metrics.csv")
    keys = ["model", "dataset", "layer", "method", "task", "split"]
    performance = _mean_std_summary(
        metrics,
        keys=keys,
        values=[
            "native_accuracy",
            "native_balanced_accuracy",
            "midpoint_accuracy",
            "midpoint_balanced_accuracy",
            "prediction_agreement",
        ],
    )
    training_memorization = performance[performance["split"] == "train"].reset_index(
        drop=True
    )

    comparisons = metrics.copy()
    comparisons["accuracy_gap_midpoint_minus_native"] = (
        comparisons["midpoint_accuracy"] - comparisons["native_accuracy"]
    )
    comparisons["balanced_gap_midpoint_minus_native"] = (
        comparisons["midpoint_balanced_accuracy"]
        - comparisons["native_balanced_accuracy"]
    )
    native_midpoint = _mean_std_summary(
        comparisons,
        keys=keys,
        values=[
            "accuracy_gap_midpoint_minus_native",
            "balanced_gap_midpoint_minus_native",
            "prediction_agreement",
        ],
    )

    training_history = _read_optional_csv(root, "training_history.csv")
    fit_metadata = _read_optional_csv(root, "fit_metadata.csv")
    diagnostic_rows: list[dict] = []
    for row in fit_metadata.to_dict(orient="records"):
        diagnostics = json.loads(row.pop("diagnostics"))
        diagnostic_rows.append(
            {
                **row,
                "n_parameters": diagnostics.get("n_parameters"),
                "best_epoch": diagnostics.get(
                    "best_epoch", diagnostics.get("selected_epoch")
                ),
                "epochs_completed": len(
                    diagnostics.get("loss_history", diagnostics.get("history", []))
                )
                or None,
                "best_validation_loss": diagnostics.get("best_validation_loss"),
                "best_training_loss": diagnostics.get("best_train_loss"),
            }
        )

    fit_diagnostics = pd.DataFrame(diagnostic_rows)
    if not training_history.empty and not fit_diagnostics.empty:
        fit_keys = ["model", "dataset", "layer", "method", "task", "seed"]
        epochs = (
            training_history.groupby(fit_keys, dropna=False)["epoch"]
            .max()
            .add(1)
            .rename("epochs_completed_from_history")
            .reset_index()
        )
        fit_diagnostics = fit_diagnostics.merge(epochs, on=fit_keys, how="left")

    causal = _read_optional_csv(root, "causal_metrics.csv")
    causal_summary = _mean_std_summary(
        causal,
        keys=keys,
        values=[
            "iia",
            "patched_accuracy",
            "corrupted_accuracy",
            "clean_accuracy",
            "recovery",
            "recovery_percent",
            "logit_flip",
            "logit_flip_percent",
            "sign_flip_rate",
        ],
    )
    return FixedLayerSelectivityReport(
        tuning_trials=_read_optional_csv(root, "tuning_trials.csv"),
        performance=performance,
        training_memorization=training_memorization,
        native_midpoint_comparison=native_midpoint,
        selectivity=pd.read_csv(root / "selectivity_summary.csv"),
        fit_diagnostics=fit_diagnostics,
        das_causal_metrics=causal_summary,
        training_history=training_history,
    )


def plot_fixed_layer_run_diagnostics(run_dir: str | Path) -> tuple[Path, ...]:
    """Create per-method plots intended for immediate notebook inspection."""

    root = Path(run_dir)
    metrics = pd.read_csv(root / "metrics.csv")
    summary = pd.read_csv(root / "selectivity_summary.csv")
    history = _read_optional_csv(root, "training_history.csv")
    causal = _read_optional_csv(root, "causal_metrics.csv")
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths: list[Path] = []

    accuracy = metrics.melt(
        id_vars=["dataset", "task", "seed", "split"],
        value_vars=["native_balanced_accuracy", "midpoint_balanced_accuracy"],
        var_name="decision_rule",
        value_name="balanced_accuracy",
    )
    grid = sns.catplot(
        data=accuracy,
        x="task",
        y="balanced_accuracy",
        hue="decision_rule",
        col="dataset",
        row="split",
        kind="bar",
        errorbar="sd",
        height=2.7,
        aspect=1.25,
    )
    grid.set(ylim=(0, 1))
    grid.figure.suptitle("Real versus random-label performance", y=1.02)
    path = figure_dir / "run_real_vs_random_accuracy.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)

    selected = summary[
        summary["metric"].isin(
            ["native_balanced_accuracy", "midpoint_balanced_accuracy"]
        )
    ]
    grid = sns.catplot(
        data=selected,
        x="split",
        y="selectivity_mean",
        hue="metric",
        col="dataset",
        kind="bar",
        height=3.0,
        aspect=1.3,
    )
    grid.refline(y=0.0, color="black", linewidth=1)
    grid.figure.suptitle("Paired selectivity across splits", y=1.02)
    path = figure_dir / "run_selectivity.png"
    grid.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.figure)
    paths.append(path)

    if not history.empty:
        loss_columns = [
            column
            for column in (
                "training_loss",
                "validation_loss",
                "post_epoch_train_loss",
            )
            if column in history and history[column].notna().any()
        ]
        losses = history.melt(
            id_vars=["dataset", "task", "seed", "epoch"],
            value_vars=loss_columns,
            var_name="loss",
            value_name="value",
        ).dropna(subset=["value"])
        grid = sns.relplot(
            data=losses,
            x="epoch",
            y="value",
            hue="loss",
            col="dataset",
            row="task",
            kind="line",
            estimator="mean",
            errorbar="sd",
            facet_kws={"sharex": False, "sharey": False},
            height=2.8,
            aspect=1.35,
        )
        grid.figure.suptitle("Training curves across seeds", y=1.02)
        path = figure_dir / "run_training_curves.png"
        grid.figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(grid.figure)
        paths.append(path)

    if not causal.empty:
        causal_long = causal.melt(
            id_vars=["dataset", "task", "seed", "split"],
            value_vars=["iia", "recovery", "logit_flip", "sign_flip_rate"],
            var_name="causal_metric",
            value_name="value",
        )
        grid = sns.catplot(
            data=causal_long,
            x="split",
            y="value",
            hue="task",
            col="dataset",
            row="causal_metric",
            kind="bar",
            errorbar="sd",
            sharey=False,
            height=2.3,
            aspect=1.35,
        )
        grid.figure.suptitle("DAS causal metrics", y=1.01)
        path = figure_dir / "run_das_causal_metrics.png"
        grid.figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(grid.figure)
        paths.append(path)
    return tuple(paths)


def selectivity_run_directories(
    run_root: str | Path,
    *,
    methods: tuple[str, ...] = SELECTIVITY_METHODS,
    models: tuple[str, ...] = SELECTIVITY_MODELS,
) -> dict[tuple[str, str], Path]:
    """Resolve completed per-method/per-model directories under one shared run."""

    root = Path(run_root)
    directories = {
        (method, model): root / "methods" / method / model
        for method in methods
        for model in models
    }
    missing = [
        str(path)
        for path in directories.values()
        if not (path / "metrics.csv").is_file()
        or not (path / "selectivity_summary.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "All method/model runs must finish before aggregation. Missing: "
            + ", ".join(missing)
        )
    return directories


def combine_fixed_layer_selectivity_runs(
    run_root: str | Path,
    *,
    methods: tuple[str, ...] = SELECTIVITY_METHODS,
    models: tuple[str, ...] = SELECTIVITY_MODELS,
) -> Path:
    """Combine all completed method/model tables, then return the combined directory."""

    directories = selectivity_run_directories(
        run_root, methods=methods, models=models
    )
    combined = Path(run_root) / "combined"
    store = RunArtifactStore(combined)
    for table in SELECTIVITY_TABLES:
        frames: list[pd.DataFrame] = []
        for (method, model), directory in directories.items():
            path = directory / f"{table}.csv"
            if not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["source_method_run"] = method
            frame["source_model_run"] = model
            frames.append(frame)
        if frames:
            rows = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates()
                .to_dict(orient="records")
            )
            store.write_rows(f"{table}.csv", rows)
    return combined


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


__all__ = [
    "FixedLayerSelectivityReport",
    "SELECTIVITY_METHODS",
    "SELECTIVITY_MODELS",
    "combine_fixed_layer_selectivity_runs",
    "load_fixed_layer_selectivity_report",
    "plot_fixed_layer_selectivity",
    "plot_fixed_layer_run_diagnostics",
    "selectivity_run_directories",
]

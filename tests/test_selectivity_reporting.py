import json
from pathlib import Path

import pandas as pd

from sentiment_geometry.reporting import (
    load_fixed_layer_selectivity_report,
    plot_fixed_layer_run_diagnostics,
)


def _write_synthetic_run(root: Path) -> None:
    metric_rows = []
    causal_rows = []
    for seed in (11, 22):
        for task, offset in (("real", 0.3), ("random", 0.0)):
            for split in ("train", "validation", "test"):
                native = 0.5 + offset
                midpoint = native - 0.02
                metric_rows.append(
                    {
                        "model": "gpt2-small",
                        "dataset": "toy_movie_review",
                        "layer": 10,
                        "method": "das",
                        "task": task,
                        "seed": seed,
                        "split": split,
                        "native_accuracy": native,
                        "native_balanced_accuracy": native,
                        "midpoint_accuracy": midpoint,
                        "midpoint_balanced_accuracy": midpoint,
                        "prediction_agreement": None,
                    }
                )
                causal_rows.append(
                    {
                        "model": "gpt2-small",
                        "dataset": "toy_movie_review",
                        "layer": 10,
                        "method": "das",
                        "task": task,
                        "seed": seed,
                        "split": split,
                        "iia": native,
                        "patched_accuracy": native,
                        "corrupted_accuracy": 0.4,
                        "clean_accuracy": 0.9,
                        "recovery": offset,
                        "recovery_percent": 100 * offset,
                        "logit_flip": offset,
                        "logit_flip_percent": 100 * offset,
                        "sign_flip_rate": offset,
                    }
                )
    pd.DataFrame(metric_rows).to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(causal_rows).to_csv(root / "causal_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "dataset": "toy_movie_review",
                "layer": 10,
                "method": "das",
                "split": split,
                "metric": metric,
                "selectivity_mean": 0.3,
                "selectivity_std": 0.0,
                "selectivity_sem": 0.0,
                "selectivity_ci95_low": 0.3,
                "selectivity_ci95_high": 0.3,
                "n_seeds": 2,
            }
            for split in ("train", "validation", "test")
            for metric in (
                "native_balanced_accuracy",
                "midpoint_balanced_accuracy",
            )
        ]
    ).to_csv(root / "selectivity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "dataset": "toy_movie_review",
                "layer": 10,
                "method": "das",
                "task": task,
                "seed": seed,
                "hyperparameters": "{}",
                "diagnostics": json.dumps(
                    {"selected_epoch": 1, "best_train_loss": 0.2}
                ),
            }
            for task in ("real", "random")
            for seed in (11, 22)
        ]
    ).to_csv(root / "fit_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "dataset": "toy_movie_review",
                "layer": 10,
                "method": "das",
                "task": task,
                "seed": seed,
                "epoch": epoch,
                "train_loss": 1.0 / (epoch + 1),
                "post_epoch_train_loss": 0.8 / (epoch + 1),
                "validation_loss": 0.7 / (epoch + 1) if task == "real" else None,
                "selected_epoch": epoch == 1,
            }
            for task in ("real", "random")
            for seed in (11, 22)
            for epoch in (0, 1)
        ]
    ).to_csv(root / "training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "dataset": "toy_movie_review",
                "layer": 10,
                "method": "das",
                "selected": True,
            }
        ]
    ).to_csv(root / "tuning_trials.csv", index=False)


def test_selectivity_report_surfaces_essential_tables_and_plots(tmp_path):
    _write_synthetic_run(tmp_path)

    report = load_fixed_layer_selectivity_report(tmp_path)

    assert set(report.training_memorization["task"]) == {"real", "random"}
    assert "balanced_gap_midpoint_minus_native_mean" in report.native_midpoint_comparison
    assert "iia_mean" in report.das_causal_metrics
    assert set(report.fit_diagnostics["epochs_completed_from_history"]) == {2}
    assert not report.training_history.empty

    paths = plot_fixed_layer_run_diagnostics(tmp_path)

    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

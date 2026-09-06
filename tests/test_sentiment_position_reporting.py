from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sentiment_geometry.reporting import (
    DATASET_ORDER,
    MODEL_ORDER,
    SentimentPositionReportData,
    figure4_style_table,
    plot_cosine_similarity_grid,
    plot_cross_position_cosines,
    plot_logit_difference_grid,
)

POSITIONS = ("adjective", "final")
METHODS = ("mean_diff", "logistic_regression", "das")
LAYERS = (1, 2, 3)


def _tables() -> dict[str, pd.DataFrame]:
    metrics = []
    best = []
    similarities = []
    summary = []
    for model_offset, model in enumerate(MODEL_ORDER):
        summary.append(
            {
                "model": model,
                "dataset": "toy_train",
                "role": "direction_fitting",
                "n_examples": 55,
                "n_directed_cases": 48,
            }
        )
        for dataset_offset, dataset in enumerate(DATASET_ORDER):
            summary.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "role": "causal_evaluation",
                    "n_examples": 30,
                    "n_directed_cases": 30,
                }
            )
            for position_offset, position in enumerate(POSITIONS):
                for method_offset, method in enumerate(METHODS):
                    for layer in LAYERS:
                        metrics.append(
                            {
                                "model": model,
                                "method": method,
                                "fit_position": position,
                                "layer": layer,
                                "dataset": dataset,
                                "logit_difference_percent": (
                                    10 * layer
                                    + model_offset
                                    + dataset_offset
                                    + position_offset
                                    + method_offset
                                ),
                                "logit_flip_percent": (
                                    40 - 5 * layer + dataset_offset + method_offset
                                ),
                            }
                        )
                    best.extend(
                        [
                            {
                                "model": model,
                                "method": method,
                                "fit_position": position,
                                "dataset": dataset,
                                "metric": "logit_difference",
                                "layer": 3,
                                "value_percent": (
                                    30
                                    + model_offset
                                    + dataset_offset
                                    + position_offset
                                    + method_offset
                                ),
                            },
                            {
                                "model": model,
                                "method": method,
                                "fit_position": position,
                                "dataset": dataset,
                                "metric": "logit_flip",
                                "layer": 1,
                                "value_percent": 35 + dataset_offset + method_offset,
                            },
                        ]
                    )
        for layer in LAYERS:
            for position_a in POSITIONS:
                for method_a_offset, method_a in enumerate(METHODS):
                    for position_b in POSITIONS:
                        for method_b_offset, method_b in enumerate(METHODS):
                            signed = 1.0 - 0.08 * abs(method_a_offset - method_b_offset)
                            if position_a != position_b:
                                signed -= 0.1
                            similarities.append(
                                {
                                    "model": model,
                                    "layer": layer,
                                    "position_a": position_a,
                                    "method_a": method_a,
                                    "position_b": position_b,
                                    "method_b": method_b,
                                    "signed_cosine": signed,
                                    "absolute_cosine": abs(signed),
                                }
                            )
    return {
        "metrics.csv": pd.DataFrame(metrics),
        "best_layers.csv": pd.DataFrame(best),
        "direction_similarities.csv": pd.DataFrame(similarities),
        "dataset_summary.csv": pd.DataFrame(summary),
    }


def _write_results(root: Path) -> None:
    tables = _tables()
    for model in MODEL_ORDER:
        model_dir = root / model
        model_dir.mkdir(parents=True)
        for filename, table in tables.items():
            table[table["model"] == model].to_csv(model_dir / filename, index=False)


def test_report_loader_reads_complete_evaluation_results_without_training_metrics(tmp_path):
    _write_results(tmp_path)
    files_before = set(tmp_path.rglob("*"))

    report = SentimentPositionReportData.load(tmp_path)

    assert report.evaluation_datasets == DATASET_ORDER
    assert not report.has_training_causal_metrics
    assert set(report.dataset_summary["dataset"]) == {"toy_train", *DATASET_ORDER}
    assert set(tmp_path.rglob("*")) == files_before


def test_figure4_style_table_has_methods_and_evaluation_metric_columns(tmp_path):
    _write_results(tmp_path)
    report = SentimentPositionReportData.load(tmp_path)

    table = figure4_style_table(
        report.best_layers,
        model="gpt2-small",
        fit_position="adjective",
    )

    assert table.shape == (3, 8)
    assert table.loc["Mean difference", "Toy adjectives\nLogit difference"] == "30.0%\n(L03)"
    assert table.loc["DAS", "SST\nLogit flip"] == "40.0%\n(L01)"


def test_read_only_plotters_return_expected_figure_layouts(tmp_path):
    _write_results(tmp_path)
    report = SentimentPositionReportData.load(tmp_path)
    files_before = set(tmp_path.rglob("*"))

    cosine = plot_cosine_similarity_grid(
        report.direction_similarities,
        model="gpt2-small",
    )
    cross_position = plot_cross_position_cosines(
        report.direction_similarities,
        model="gpt2-small",
    )
    layers = plot_logit_difference_grid(
        report.metrics,
        dataset="toy_adjectives",
    )

    assert len(cosine.axes) == 7  # Six heatmaps and one shared color bar.
    assert len(cross_position.axes) == 1
    assert len(layers.axes) == 4
    assert set(tmp_path.rglob("*")) == files_before
    plt.close(cosine)
    plt.close(cross_position)
    plt.close(layers)

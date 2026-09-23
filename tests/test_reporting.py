from __future__ import annotations

import pandas as pd
import pytest

from sentiment_geometry.reporting import (
    load_ait_valence_report_data,
    select_table1_best_layers,
    table1_cell_text,
)


def _layer_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "layer": 0,
                "toy_logit_diff_percent": 10.0,
                "toy_logit_flip_percent": 80.0,
                "sst_logit_diff_percent": 30.0,
                "sst_logit_flip_percent": 40.0,
            },
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "layer": 1,
                "toy_logit_diff_percent": 90.0,
                "toy_logit_flip_percent": 20.0,
                "sst_logit_diff_percent": 70.0,
                "sst_logit_flip_percent": 10.0,
            },
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "layer": 2,
                "toy_logit_diff_percent": 50.0,
                "toy_logit_flip_percent": 40.0,
                "sst_logit_diff_percent": 20.0,
                "sst_logit_flip_percent": 100.0,
            },
            {
                "model": "gpt2-small",
                "method": "pca",
                "layer": 0,
                "toy_logit_diff_percent": 50.0,
                "toy_logit_flip_percent": 50.0,
                "sst_logit_diff_percent": 50.0,
                "sst_logit_flip_percent": 50.0,
            },
            {
                "model": "gpt2-small",
                "method": "pca",
                "layer": 1,
                "toy_logit_diff_percent": 50.0,
                "toy_logit_flip_percent": 50.0,
                "sst_logit_diff_percent": 50.0,
                "sst_logit_flip_percent": 50.0,
            },
        ]
    )


def test_table1_metrics_select_their_best_layers_independently():
    best = select_table1_best_layers(_layer_metrics())
    mean_diff = best[best.method == "mean_diff"].set_index(["dataset", "metric"])

    assert len(mean_diff) == 4
    assert mean_diff.loc[("toy_movie_review", "logit_difference"), "layer"] == 1
    assert mean_diff.loc[("toy_movie_review", "logit_flip"), "layer"] == 0
    assert mean_diff.loc[("sst", "logit_difference"), "layer"] == 1
    assert mean_diff.loc[("sst", "logit_flip"), "layer"] == 2
    assert mean_diff.loc[("sst", "logit_flip"), "value_percent"] == 100.0


def test_table1_selection_breaks_ties_at_the_lower_layer():
    best = select_table1_best_layers(_layer_metrics())
    assert set(best.loc[best.method == "pca", "layer"]) == {0}


def test_table1_formatter_includes_values_and_selected_layers():
    table = table1_cell_text(select_table1_best_layers(_layer_metrics()))
    assert table.shape == (2, 4)
    assert table.loc["mean_diff", "ToyMovieReview\nlogit difference"] == "90.0%\n(layer 1)"
    assert table.loc["mean_diff", "SST\nlogit flip"] == "100.0%\n(layer 2)"


def test_table1_selection_rejects_a_missing_metric_instead_of_silently_omitting_it():
    metrics = _layer_metrics()
    metrics.loc[metrics.method == "pca", "sst_logit_flip_percent"] = float("nan")

    with pytest.raises(ValueError, match="sst_logit_flip_percent"):
        select_table1_best_layers(metrics)


def test_ait_report_loader_combines_explicit_per_model_tables(tmp_path):
    table_names = (
        "dataset_summary",
        "metrics",
        "final_metrics",
        "direction_metadata",
        "direction_similarities",
        "layer_selection",
    )
    for index, model_name in enumerate(("gpt2-small", "qwen-0.6b")):
        model_dir = tmp_path / model_name
        model_dir.mkdir()
        for table_name in table_names:
            pd.DataFrame(
                [{"model": model_name, "table": table_name, "order": index}]
            ).to_csv(model_dir / f"{table_name}.csv", index=False)

    report = load_ait_valence_report_data(
        tmp_path,
        model_names=["gpt2-small", "qwen-0.6b"],
    )

    assert report.dataset_summary["model"].tolist() == ["gpt2-small", "qwen-0.6b"]
    assert report.final_metrics["model"].tolist() == ["gpt2-small", "qwen-0.6b"]
    assert report.direction_similarities["model"].tolist() == [
        "gpt2-small",
        "qwen-0.6b",
    ]


def test_ait_report_loader_rejects_mislabeled_model_table(tmp_path):
    model_dir = tmp_path / "gpt2-small"
    model_dir.mkdir()
    pd.DataFrame([{"model": "qwen-0.6b"}]).to_csv(
        model_dir / "dataset_summary.csv", index=False
    )

    with pytest.raises(ValueError, match="expected only 'gpt2-small'"):
        load_ait_valence_report_data(tmp_path, model_names=["gpt2-small"])

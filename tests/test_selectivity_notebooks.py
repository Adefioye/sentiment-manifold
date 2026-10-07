import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment_geometry.reporting import combine_fixed_layer_selectivity_runs

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOKS = {
    "mean_diff": "12_colab_fixed_layer_mean_difference_selectivity.ipynb",
    "logistic_regression": "13_colab_fixed_layer_logistic_regression_selectivity.ipynb",
    "das": "14_colab_fixed_layer_das_selectivity.ipynb",
    "mlp1": "15_colab_fixed_layer_mlp1_selectivity.ipynb",
}


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize(("method", "filename"), NOTEBOOKS.items())
def test_selectivity_colab_notebook_is_clean_thin_and_uses_shared_run(
    method, filename
):
    source = _source(PROJECT_ROOT / "notebooks" / filename)

    assert f'METHOD = "{method}"' in source
    assert 'RUN_ID = "fixed-layer-selectivity-v1"' in source
    assert '"gpt2-small": {"toy_movie_review": 10, "full_ait": 11}' in source
    assert '"qwen-0.6b": {"toy_movie_review": 26, "full_ait": 26}' in source
    assert 'run_and_display("gpt2-small")' in source
    assert 'run_and_display("qwen-0.6b")' in source
    assert "run_fixed_layer_selectivity(config)" in source
    assert "SHOW_PROGRESS = True" in source
    assert "load_fixed_layer_selectivity_report(run_dir)" in source
    assert "report.training_memorization" in source
    assert "report.native_midpoint_comparison" in source
    assert "report.fit_diagnostics" in source
    assert "report.das_causal_metrics" in source
    assert "plot_fixed_layer_run_diagnostics(run_dir)" in source
    assert "combine_fixed_layer_selectivity_runs(RUN_ROOT)" in source
    assert "GridSearchCV" not in source
    assert "ParameterGrid" not in source


def test_combiner_requires_all_methods_and_models_then_combines(tmp_path):
    root = tmp_path / "shared-run"
    methods = tuple(NOTEBOOKS)
    models = ("gpt2-small", "qwen-0.6b")
    for method in methods:
        for model in models:
            directory = root / "methods" / method / model
            directory.mkdir(parents=True)
            row = {
                "model": model,
                "dataset": "toy_movie_review",
                "layer": 10,
                "method": method,
                "seed": 11,
                "split": "test",
            }
            pd.DataFrame([row]).to_csv(directory / "metrics.csv", index=False)
            pd.DataFrame([{**row, "metric": "native_accuracy"}]).to_csv(
                directory / "selectivity_summary.csv", index=False
            )

    missing = root / "methods" / "mlp1" / "qwen-0.6b" / "metrics.csv"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="All method/model runs"):
        combine_fixed_layer_selectivity_runs(root)

    pd.DataFrame(
        [
            {
                "model": "qwen-0.6b",
                "dataset": "toy_movie_review",
                "layer": 26,
                "method": "mlp1",
                "seed": 11,
                "split": "test",
            }
        ]
    ).to_csv(missing, index=False)
    combined = combine_fixed_layer_selectivity_runs(root)

    assert combined == root / "combined"
    metrics = pd.read_csv(combined / "metrics.csv")
    assert len(metrics) == 8
    assert set(metrics["method"]) == set(methods)

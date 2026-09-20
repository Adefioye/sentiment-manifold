import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/03_colab_sentiment_position_comparison.ipynb"
EXPLORATION_NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks/04_colab_explore_sentiment_position_results.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _exploration_notebook():
    return json.loads(EXPLORATION_NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_sentiment_position_notebook_is_valid_and_code_cells_compile():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")


def test_sentiment_position_notebook_imports_package_and_plots_all_metrics():
    source = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])

    assert 'pip", "install", "-e"' in source
    assert "import sentiment_geometry" in source
    assert "pkgutil.walk_packages" in source
    assert 'expected_final_evaluations = ["toy_adverbs", "toy_adjectives", "sst"]' in source
    assert '"logit_difference_percent": "Logit difference (%)"' in source
    assert '"logit_flip_percent": "Logit flip (%)"' in source
    assert '"sign_flip_percent": "Literal sign flip (%)"' in source
    assert 'data=model_metrics,\n            x="fit_position"' in source
    assert 'row="method"' in source
    assert 'col="dataset"' in source
    assert 'figure_path = model_figure_dir / f"{metric_column}_by_fit_position.png"' in source


def test_sentiment_position_exploration_notebook_is_valid_and_code_cells_compile():
    notebook = _exploration_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]),
                f"{EXPLORATION_NOTEBOOK_PATH.name}:cell-{index}",
                "exec",
            )


def test_exploration_notebook_fixes_run_position_order_and_layer_reporting():
    source = "\n".join(
        "".join(cell["source"]) for cell in _exploration_notebook()["cells"]
    )

    assert 'RUN_ID = "2026-09-18_20-22_CDT"' in source
    assert 'POSITION_ORDER = ("adjective", "verb", "summary", "final")' in source
    assert 'POSITION_LABELS = ("ADJ", "VRB", "SUM", "END")' in source
    assert '.sort_values("position_index", kind="stable")' in source
    assert 'observed_order != POSITION_ORDER' in source
    assert 'axis.plot(' in source
    assert '"logit_difference_percent": "Logit difference (%)"' in source
    assert '"logit_flip_percent": "Logit flip (%)"' in source
    assert '"sign_flip_percent": "Literal sign flip (%)"' in source
    assert '"mean_diff": "Mean Difference"' in source
    assert '"logistic_regression": "Logistic Regression"' in source
    assert '"toy_adverbs": "ToyMovieReview(ADVRB)"' in source
    assert '"toy_adjectives": "ToyMovieReview(ADJ)"' in source
    assert 'axis.set_xlabel("Activation position")' in source
    assert 'set(selection["selection_dataset"]) != {"toy_adverbs"}' in source
    assert 'set(selection["selection_metric"]) != {"logit_flip_percent"}' in source
    assert 'mismatched_sst' in source
    assert 'report.selected_metrics["dataset"] == "sst"' in source
    assert '"logit_flip_percent": "SST Logit Flip (%)"' in source
    assert '"sign_flip_percent": "SST Literal Sign Flip (%)"' in source
    assert 'row[metric_column]' in source
    assert 'SST is not used to reselect the layer' in source

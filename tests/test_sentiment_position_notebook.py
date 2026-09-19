import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/03_colab_sentiment_position_comparison.ipynb"


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


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

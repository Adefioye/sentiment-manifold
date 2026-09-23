import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/06_colab_train_ait_last_token_directions.ipynb"


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_ait_last_token_notebook_is_clean_and_all_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_ait_last_token_notebook_locks_training_and_drive_artifacts():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        '"--last-token"',
        '"--all-non-embedding-layers"',
        'METHODS = ["mean_diff", "logistic_regression", "das"]',
        'experiment_name="ait-last-token-directions"',
        "all_models_direction_similarities.csv",
        "snapshot_similarity_summary.csv",
        "plot_ait_valence_run",
        "RUN_LAYOUT.directions_dir",
        "RUN_LAYOUT.figures_dir",
    ):
        assert required in source
    assert 'MODEL_NAMES = [\n    "gpt2-small",\n    "qwen-0.6b",\n]' in source
    assert '"gemma-2b"' not in source
    assert '"pythia-1.4b"' not in source


def test_ait_last_token_notebook_passes_then_clears_hf_token():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    assert 'child_environment["HF_TOKEN"] = get_runtime_secret("HF_TOKEN")' in source
    assert 'child_environment.pop("HF_TOKEN", None)' in source
    assert "clear_hf_credentials()" in source
    assert 'if os.environ.get("HF_TOKEN") is not None' in source
    assert 'if "HF_TOKEN" in _RUNTIME_SECRETS' in source

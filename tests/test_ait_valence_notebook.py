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
        "all_models_final_metrics.csv",
        "all_models_final_patching_records.csv",
        'values="logit_flip_percent"',
        'values="sign_flip_percent"',
        '"Locked-test logit flip percent"',
        '"Locked-test sign flip percent"',
        "snapshot_similarity_summary.csv",
        "plot_ait_valence_run",
        "RUN_LAYOUT.directions_dir",
        "RUN_LAYOUT.figures_dir",
        '"gpt2-small": "gpt2_small_matched_pairs"',
        '"qwen-0.6b": "qwen_0_6b_matched_pairs"',
        'dataset_summary.groupby("model")["dataset_config"]',
        "required_model_files",
        '"layer_selection_role": "eval"',
        '"final_evaluation_role": "test"',
        '"test_is_unbiased_final_evaluation": True',
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


def test_ait_last_token_notebook_streams_and_persists_cli_failures():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    assert 'training_log_path = RUN_LAYOUT.root / "training.log"' in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'print(line, end="", flush=True)' in source
    assert '"training_log": str(training_log_path)' in source
    assert 'failure_metadata["exit_code"] = error.returncode' in source

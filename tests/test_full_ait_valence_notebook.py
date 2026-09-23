import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks/07_colab_train_full_ait_last_token_directions.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_full_ait_last_token_notebook_is_clean_and_all_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_full_ait_last_token_notebook_locks_full_data_contract():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "configs/full_ait_valence_directions.yaml",
        '"--last-token"',
        '"--all-non-embedding-layers"',
        'METHODS = ["mean_diff", "logistic_regression", "das"]',
        'experiment_name="full-ait-last-token-directions"',
        "base_config.sampling.uses_all_available",
        '"layer_selection_role": "eval"',
        '"final_evaluation_role": "test"',
        '"test_is_unbiased_final_evaluation": True',
        '"n_matched_pairs"',
        'values="n_examples"',
        'values="n_directed_cases"',
        '"gpt2-small": "gpt2_small_matched_pairs"',
        '"qwen-0.6b": "qwen_0_6b_matched_pairs"',
        '"gemma-2b": "gemma_2b_matched_pairs"',
        '"pythia-1.4b": "pythia_1_4b_matched_pairs"',
    ):
        assert required in source


def test_full_ait_last_token_notebook_reports_test_metrics_and_snapshot_cosines():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "all_models_final_metrics.csv",
        "all_models_final_patching_records.csv",
        'values="logit_flip_percent"',
        'values="sign_flip_percent"',
        '"Locked-test logit flip percent"',
        '"Locked-test sign flip percent"',
        "all_models_direction_similarities.csv",
        "snapshot_similarity_summary.csv",
        '["first", "middle", "last"]',
        "plot_ait_valence_run",
        "figure_manifest.csv",
    ):
        assert required in source


def test_full_ait_last_token_notebook_persists_and_audits_artifacts():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "RUN_LAYOUT.results_dir",
        "RUN_LAYOUT.directions_dir",
        "RUN_LAYOUT.figures_dir",
        'training_log_path = RUN_LAYOUT.root / "training.log"',
        "stdout=subprocess.PIPE",
        "stderr=subprocess.STDOUT",
        "required_result_files",
        "required_model_files",
        'rglob("*.npz")',
        'child_environment["HF_TOKEN"] = get_runtime_secret("HF_TOKEN")',
        'child_environment.pop("HF_TOKEN", None)',
        "clear_hf_credentials()",
    ):
        assert required in source

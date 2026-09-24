import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/08_colab_evaluate_frozen_ait_valence_directions.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_ait_transfer_notebook_is_clean_and_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_ait_transfer_notebook_locks_selection_and_ood_contract():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "configs/ait_valence_transfer_evaluation.yaml",
        'SOURCE_RUN_ID = "2026-09-23_09-10_CDT"',
        'ALL_MODEL_NAMES = ["gpt2-small", "qwen-0.6b"]',
        'EVALUATION_MODEL_NAMES = ["gpt2-small", "qwen-0.6b"]',
        "AITValenceTransferConfig.load",
        "load_frozen_direction_selections",
        "run_frozen_sentiment_direction_evaluation(config)",
        '{"ait_eval"}',
        'plan.source_evaluation_dataset',
        'plan.method_patch_positions',
        'DATASET_ORDER = [dataset.name for dataset in plan.datasets]',
        '"logit_flip_percent": "Logit Flip Percent"',
        '"sign_flip_percent": "Literal Sign Flip Percent"',
        'RUN_LAYOUT.update_manifest(\n    status="completed"',
    ):
        assert required in source


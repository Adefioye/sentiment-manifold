import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/11_colab_evaluate_frozen_partial_ait_valence_directions.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_partial_ait_transfer_notebook_is_clean_and_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_partial_ait_transfer_notebook_preserves_source_provenance():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "configs/partial_ait_valence_transfer_evaluation.yaml",
        'SOURCE_RUN_ID = "2026-09-23_02-47_CDT"',
        'plan.source_experiment_name != "ait-last-token-directions"',
        'plan.methods != ["mean_diff", "das"]',
        '{"ait_test"}',
        '"selection_provenance_caveat": "best layers selected on ait_test"',
        "load_frozen_direction_selections",
        "run_frozen_sentiment_direction_evaluation(config)",
        '"logit_flip_percent": "Logit Flip Percent"',
        '"sign_flip_percent": "Literal Sign Flip Percent"',
        '"reuse": ["gpt2-small"]',
        'status="completed" if FINAL_COMBINED_RUN else "partial"',
    ):
        assert required in source

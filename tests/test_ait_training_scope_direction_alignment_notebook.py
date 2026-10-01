import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/10_colab_explore_ait_training_scope_direction_alignment.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_ait_training_scope_alignment_notebook_is_clean_and_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_notebook_is_limited_to_full_vs_partial_absolute_alignment():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "configs/ait_training_scope_direction_alignment.yaml",
        'UPDATE_EXISTING_CHECKOUT = True',
        '["git", "fetch", "origin", PROJECT_BRANCH]',
        '["git", "merge", "--ff-only", f"origin/{PROJECT_BRANCH}"]',
        "2026-09-23_02-47_CDT",
        "2026-09-23_09-10_CDT",
        "PairwiseDirectionAlignmentConfig.load",
        "run_pairwise_direction_alignment_analysis",
        "plot_pairwise_absolute_direction_alignment",
        "Cosine alignment",
        '["mean_diff", "das"]',
        'row_label="Full AIT"',
        'column_label="Partial AIT"',
        '"partial_ait": "ait_test"',
        '"full_ait": "ait_eval"',
        "Stale alignment configuration in the Colab checkout",
        'status="completed"',
    ):
        assert required in source
    assert "logistic_regression" not in source
    assert "signed_cosine" not in source
    assert "selection_value_percent" not in source
    assert "Absolute-cosine alignment" not in source

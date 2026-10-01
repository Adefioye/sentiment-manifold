import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks/09_colab_explore_sentiment_valence_direction_alignment.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_direction_alignment_notebook_is_clean_and_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_direction_alignment_notebook_uses_absolute_cosine_for_three_views():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for required in (
        "configs/sentiment_valence_direction_alignment.yaml",
        'RESUME_RUN_ID = "2026-09-24_09-15_CDT"',
        "RUN_ANALYSIS = False",
        'UPDATE_EXISTING_CHECKOUT = True',
        '["git", "fetch", "origin", PROJECT_BRANCH]',
        '["git", "merge", "--ff-only", f"origin/{PROJECT_BRANCH}"]',
        "2026-09-18_20-22_CDT",
        "2026-09-23_09-10_CDT",
        "DirectionAlignmentConfig.load",
        "load_direction_alignment_result",
        "run_direction_alignment_analysis",
        "open_timestamped_run",
        "plot_direction_alignment",
        'REPRESENTATION_ORDER = ["sentiment", "valence"]',
        'COMPARISON_LABELS = {',
        '"within_sentiment"',
        '"within_valence"',
        '"valence_vs_sentiment"',
        '"absolute_cosine"',
        "Cosine alignment",
        '"figure_manifest.csv"',
        'status="completed"',
    ):
        assert required in source
    assert '"signed_cosine"' not in source
    assert '"row_layer"' not in source
    assert '"column_layer"' not in source

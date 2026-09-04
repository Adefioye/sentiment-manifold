import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/02_colab_preprocess_publish_explore_rq2.ipynb"


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_rq2_colab_notebook_is_valid_and_code_cells_compile():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "T4"

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:cell-{index}", "exec")


def test_rq2_colab_notebook_covers_preprocessing_private_publish_and_exploration():
    source = "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
    )

    for command in (
        "preprocess-ait",
        "preprocess-sst",
        "preprocess-imdb",
        "preprocess-dynasent",
    ):
        assert command in source
    assert 'FILTER_MODEL = "pythia-2.8b"' in source
    assert "MAX_PAIRING_PROMPT_TOKENS = 1000" in source
    assert '"--max-pairing-prompt-tokens", str(MAX_PAIRING_PROMPT_TOKENS)' in source
    assert '"--push-to-hub", "--private"' in source
    assert 'getpass(f"Enter {name} (input hidden): ")' in source
    assert '_RUNTIME_SECRETS[name] = value' in source
    assert 'child_env["HF_TOKEN"] = get_runtime_secret("HF_TOKEN")' in source
    assert 'child_env.pop("HF_TOKEN", None)' in source
    assert 'delete_runtime_secret("HF_TOKEN")' in source
    assert "AIT_HUB_UPLOAD_PERMISSION_ACKNOWLEDGED" in source
    assert "hf_hub_download" in source
    assert "load_processed" in source
    assert "GATE_TABLE" in source
    assert "PAIR_TABLE" in source
    assert "INVARIANT_TABLE" in source
    assert 'for split in ("train", "validation")' in source

    # A tracked notebook must never contain a pasted Hugging Face access token.
    assert re.search(r"hf_[A-Za-z0-9]{20,}", source) is None


def test_rq2_colab_notebook_mounts_drive_and_uses_configured_ait_source():
    source = "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
    )

    assert 'DRIVE_MOUNT_ROOT = CONTENT_ROOT / "drive"' in source
    assert (
        'AIT_SOURCE = DRIVE_MOUNT_ROOT / "MyDrive/sentiment-manifold/data/ait/V-oc"'
        in source
    )
    assert "from google.colab import drive" in source
    assert 'drive.mount(str(DRIVE_MOUNT_ROOT), force_remount=False)' in source
    assert "uploaded = files.upload()" not in source
    assert "from google.colab import files" not in source


def test_rq2_exploration_is_standalone_and_configuration_isolated():
    notebook = _notebook()
    part_two_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "Part II: standalone local exploration" in "".join(cell.get("source", []))
    )
    exploration_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"][part_two_index:]
        if cell["cell_type"] == "code"
    )

    for dependency in (
        "import matplotlib.pyplot as plt",
        "import pandas as pd",
        "import seaborn as sns",
        "from datasets import load_dataset",
        "from huggingface_hub import HfApi, get_token, hf_hub_download",
    ):
        assert dependency in exploration_code
    for colab_only_name in (
        "CONTENT_ROOT",
        "PROJECT_ROOT",
        "OUTPUTS",
        "SHOULD_PUSH",
        "load_from_disk",
        "git clone",
    ):
        assert colab_only_name not in exploration_code
    assert 'load_dataset("parquet", data_files=data_files)' in exploration_code
    assert 'SUMMARY_DIR = Path.cwd() / "rq2-data-exploration"' in exploration_code
    assert "def get_exploration_token():" in exploration_code
    assert "token=get_exploration_token()" in exploration_code
    assert "token=_EXPLORATION_TOKEN" not in exploration_code

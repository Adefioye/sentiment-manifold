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
    assert '"--push-to-hub", "--private"' in source
    assert 'getpass(f"Enter {name} (input hidden): ")' in source
    assert '_RUNTIME_SECRETS[name] = value' in source
    assert 'child_env["HF_TOKEN"] = get_runtime_secret("HF_TOKEN")' in source
    assert 'child_env.pop("HF_TOKEN", None)' in source
    assert 'delete_runtime_secret("HF_TOKEN")' in source
    assert 'os.environ.pop("HF_TOKEN", None)' in source
    assert "AIT_HUB_UPLOAD_PERMISSION_ACKNOWLEDGED" in source
    assert "get_dataset_config_names" in source
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

import json
from pathlib import Path

from sentiment_geometry.experiments import AITValenceTransferConfig


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/ait_valence_transfer_evaluation.yaml"


def test_ait_transfer_config_locks_source_selection_and_patch_protocol():
    config = AITValenceTransferConfig.load(CONFIG_PATH)

    assert config.source_experiment_name == "full-ait-last-token-directions"
    assert config.source_run_id == "2026-09-23_09-10_CDT"
    assert config.selection_dataset == "ait_eval"
    assert config.source_evaluation_dataset == "ait_test"
    assert config.source_direction_domain == "ait_valence"
    assert config.source_activation_representation == "last_token"
    assert config.method_patch_positions == {
        "mean_diff": "final",
        "das": "all",
    }
    assert config.methods == ["mean_diff", "das"]
    assert [dataset.name for dataset in config.datasets] == [
        "sst",
        "imdb",
        "dynasent_r1",
        "dynasent_r2",
    ]
    assert all(dataset.revision for dataset in config.datasets)


def test_ait_transfer_config_builds_frozen_evaluation(tmp_path):
    plan = AITValenceTransferConfig.load(CONFIG_PATH)
    source_root = plan.source_run_root(tmp_path)
    source_root.mkdir(parents=True)
    (source_root / "run_manifest.json").write_text(
        json.dumps({"run_id": plan.source_run_id, "status": "completed"}),
        encoding="utf-8",
    )

    config = plan.build_evaluation_config(
        storage_root=tmp_path,
        output_dir=tmp_path / "output",
        evaluated_model_names=["qwen-0.6b"],
        reuse_completed_models=["gpt2-small"],
        device="cuda",
        dtype="auto",
        batch_sizes={"qwen-0.6b": 8},
    )

    assert config.source_run_root == str(source_root)
    assert config.source_evaluation_dataset == "ait_test"
    assert config.source_direction_domain == "ait_valence"
    assert config.source_activation_representation == "last_token"
    assert config.models[0].name == "qwen-0.6b"
    assert config.models[0].device == "cuda"
    assert config.models[0].batch_size == 8
    assert config.reuse_completed_models == ["gpt2-small"]
    assert config.method_patch_positions["das"] == "all"


def test_ait_transfer_config_supports_a_single_model_first_stage(tmp_path):
    plan = AITValenceTransferConfig.load(CONFIG_PATH)
    source_root = plan.source_run_root(tmp_path)
    source_root.mkdir(parents=True)
    (source_root / "run_manifest.json").write_text(
        json.dumps({"run_id": plan.source_run_id, "status": "completed"}),
        encoding="utf-8",
    )

    config = plan.build_evaluation_config(
        storage_root=tmp_path,
        output_dir=tmp_path / "output",
        evaluated_model_names=["gpt2-small"],
        batch_sizes={"gpt2-small": 16},
    )

    assert [model.name for model in config.models] == ["gpt2-small"]
    assert config.models[0].batch_size == 16
    assert config.reuse_completed_models == []

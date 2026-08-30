from pathlib import Path

from sentiment_manifold.config import ReproductionConfig
from sentiment_manifold.storage import (
    checkpoint_variant_dir,
    resolve_checkpoint_dir,
    resolve_output_dir,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_default_storage_roots_are_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SENTIMENT_MANIFOLD_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SENTIMENT_MANIFOLD_CHECKPOINT_DIR", raising=False)
    assert resolve_output_dir() == tmp_path / "outputs/results"
    assert resolve_checkpoint_dir() == tmp_path / "checkpoints"


def test_config_storage_environment_overrides_are_independent(tmp_path, monkeypatch):
    results = tmp_path / "lightweight-results"
    checkpoints = tmp_path / "persistent-checkpoints"
    monkeypatch.setenv("SENTIMENT_MANIFOLD_OUTPUT_DIR", str(results))
    monkeypatch.setenv("SENTIMENT_MANIFOLD_CHECKPOINT_DIR", str(checkpoints))
    config = ReproductionConfig.load(PROJECT_ROOT / "configs/reproduction.yaml")
    assert config.experiment.output_dir == str(results)
    assert config.experiment.checkpoint_dir == str(checkpoints)


def test_checkpoint_variants_separate_models_and_configurations(tmp_path):
    common = {
        "artifact_schema_version": 4,
        "model_revision": "revision-a",
        "fit_hyperparameters": {"learning_rate": 0.001},
    }
    first = checkpoint_variant_dir(
        tmp_path,
        model_name="gpt2-small",
        phase="reproduction",
        method="das",
        fingerprint_payload=common,
    )
    repeated = checkpoint_variant_dir(
        tmp_path,
        model_name="gpt2-small",
        phase="reproduction",
        method="das",
        fingerprint_payload=common,
    )
    other_model = checkpoint_variant_dir(
        tmp_path,
        model_name="EleutherAI/pythia-1.4b",
        phase="reproduction",
        method="das",
        fingerprint_payload=common,
    )
    other_settings = checkpoint_variant_dir(
        tmp_path,
        model_name="gpt2-small",
        phase="reproduction",
        method="das",
        fingerprint_payload={**common, "fit_hyperparameters": {"learning_rate": 0.0003}},
    )
    assert first == repeated
    assert first.parts[-4:-1] == ("gpt2-small", "reproduction", "das")
    assert other_model.parts[-4] == "EleutherAI-pythia-1.4b"
    assert other_model != first
    assert other_settings != first

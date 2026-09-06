import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentiment_geometry.experiments.reproduction import ReproductionConfig
from sentiment_geometry.persistence import (
    checkpoint_variant_dir,
    prepare_timestamped_run,
    resolve_checkpoint_dir,
    resolve_output_dir,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_default_storage_roots_are_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SENTIMENT_GEOMETRY_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SENTIMENT_GEOMETRY_CHECKPOINT_DIR", raising=False)
    assert resolve_output_dir() == tmp_path / "outputs/results"
    assert resolve_checkpoint_dir() == tmp_path / "checkpoints"


def test_config_storage_environment_overrides_are_independent(tmp_path, monkeypatch):
    results = tmp_path / "lightweight-results"
    checkpoints = tmp_path / "persistent-checkpoints"
    monkeypatch.setenv("SENTIMENT_GEOMETRY_OUTPUT_DIR", str(results))
    monkeypatch.setenv("SENTIMENT_GEOMETRY_CHECKPOINT_DIR", str(checkpoints))
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


def test_timestamped_run_creates_readable_isolated_layout(tmp_path):
    layout = prepare_timestamped_run(
        tmp_path,
        experiment_name="sentiment-position-comparison",
        timezone_name="America/Chicago",
        now=datetime(2026, 9, 5, 23, 42, tzinfo=timezone.utc),
    )

    assert layout.run_id == "2026-09-05_18-42_CDT"
    assert layout.root == (
        tmp_path
        / "sentiment-position-comparison"
        / "runs"
        / "2026-09-05_18-42_CDT"
    )
    assert layout.results_dir.is_dir()
    assert layout.directions_dir.is_dir()
    assert layout.figures_dir.is_dir()
    manifest = json.loads(layout.manifest_path.read_text())
    assert manifest["started_at"] == "2026-09-05T18:42-05:00"
    assert manifest["started_at_utc"] == "2026-09-05T23:42+00:00"
    assert manifest["status"] == "initialized"


def test_timestamped_run_refuses_implicit_overwrite_and_supports_explicit_resume(tmp_path):
    now = datetime(2026, 1, 4, 16, 7, tzinfo=timezone.utc)
    first = prepare_timestamped_run(
        tmp_path,
        experiment_name="sentiment-position-comparison",
        now=now,
    )
    with pytest.raises(FileExistsError, match="resume_run_id"):
        prepare_timestamped_run(
            tmp_path,
            experiment_name="sentiment-position-comparison",
            now=now,
        )

    resumed = prepare_timestamped_run(
        tmp_path,
        experiment_name="sentiment-position-comparison",
        timezone_name="America/Chicago",
        resume_run_id=first.run_id,
    )
    assert resumed.root == first.root
    assert resumed.resumed is True
    assert resumed.timezone_name == "UTC"
    assert json.loads(resumed.manifest_path.read_text())["status"] == "resumed"


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", ".", "with spaces"])
def test_timestamped_run_rejects_unsafe_resume_identifiers(tmp_path, run_id):
    with pytest.raises(ValueError, match="filesystem-safe"):
        prepare_timestamped_run(
            tmp_path,
            experiment_name="sentiment-position-comparison",
            resume_run_id=run_id,
        )

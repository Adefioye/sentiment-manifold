import json
from pathlib import Path

import pandas as pd

from sentiment_geometry.datasets.types import CounterfactualPair, TextExample
from sentiment_geometry.experiments import (
    FrozenDirectionEvaluationConfig,
    FrozenEvaluationDataset,
    FrozenSentimentDirectionEvaluation,
    load_frozen_direction_selections,
)
from sentiment_geometry.experiments.sentiment_position.evaluation import (
    DirectionEvaluationResult,
)
from sentiment_geometry.fitting_methods import DirectionArtifact
from sentiment_geometry.models import ModelConfig


def _source_run(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "sentiment-position-comparison" / "runs" / "prior"
    model_results = source / "results" / "gpt2-small"
    checkpoint = (
        source
        / "directions"
        / "gpt2-small"
        / "sentiment-position-comparison"
        / "final-mean_diff"
        / "fingerprint"
        / "layer03.npz"
    )
    model_results.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    (source / "run_manifest.json").write_text(
        json.dumps({"run_id": "prior", "status": "completed"}), encoding="utf-8"
    )
    DirectionArtifact(
        method="mean_diff",
        model_name="gpt2",
        layer=3,
        vector=[3.0, 4.0],
        metadata={
            "artifact_schema_version": 2,
            "fit_position": "final",
            "model_revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
        },
    ).save(checkpoint)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "fit_position": "final",
                "selected_layer": 3,
                "selection_dataset": "toy_adverbs",
                "selection_metric": "logit_flip_percent",
                "selection_value_percent": 82.5,
                "direction_checkpoint": "/stale/colab/path/layer03.npz",
            }
        ]
    ).to_csv(model_results / "layer_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "fit_position": "final",
                "layer": 3,
                "selected_layer": True,
                "artifact_path": "/stale/colab/path/layer03.npz",
                "artifact_relative_path": str(checkpoint.relative_to(source / "directions")),
            }
        ]
    ).to_csv(model_results / "direction_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "mean_diff",
                "fit_position": "final",
                "dataset": "sst",
                "layer": 3,
                "selected_layer": True,
                "logit_flip_percent": 40.0,
            }
        ]
    ).to_csv(model_results / "selected_metrics.csv", index=False)
    return source, checkpoint


def _config(tmp_path: Path, source: Path) -> FrozenDirectionEvaluationConfig:
    return FrozenDirectionEvaluationConfig(
        source_run_root=str(source),
        output_dir=str(tmp_path / "results"),
        models=[
            ModelConfig(
                name="gpt2-small",
                revision="model-commit",
                device="cpu",
                dtype="float32",
                batch_size=2,
            )
        ],
        datasets=[
            FrozenEvaluationDataset(
                name="sst",
                repo_id="owner/private-sst",
                revision="dataset-commit",
                configs={"gpt2-small": "tigges_gpt2_small_directed_pairs"},
            )
        ],
        methods=["mean_diff"],
        fit_position="final",
        selection_dataset="toy_adverbs",
        selection_metric="logit_flip_percent",
    )


def test_frozen_selection_relocates_checkpoint_and_preserves_adverb_selection(tmp_path):
    source, checkpoint = _source_run(tmp_path)
    config = _config(tmp_path, source)

    selections = load_frozen_direction_selections(config, config.models[0])

    assert len(selections) == 1
    selected = selections[0]
    assert selected.selected_layer == 3
    assert selected.selection_dataset == "toy_adverbs"
    assert selected.selection_metric == "logit_flip_percent"
    assert selected.selection_value_percent == 82.5
    assert selected.checkpoint_path == checkpoint.resolve()
    assert selected.artifact.metadata["fit_position"] == "final"


def test_frozen_evaluation_writes_metrics_records_and_provenance(
    monkeypatch, tmp_path
):
    source, _ = _source_run(tmp_path)
    config = _config(tmp_path, source)
    clean = TextExample(
        text="Review Text: good Review Sentiment:",
        label=1,
        example_id="positive",
        metadata={
            "pairing_model": "gpt2-small",
            "resolved_dataset_revision": "dataset-commit",
        },
    )
    corrupted = TextExample(
        text="Review Text: bad Review Sentiment:",
        label=0,
        example_id="negative",
        metadata={
            "pairing_model": "gpt2-small",
            "resolved_dataset_revision": "dataset-commit",
        },
    )
    monkeypatch.setattr(
        "sentiment_geometry.experiments.sentiment_position.frozen_evaluation."
        "load_hf_directed_pairs",
        lambda *args, **kwargs: [CounterfactualPair(clean=clean, corrupted=corrupted)],
    )

    class FakeAdapter:
        def provenance(self):
            return {
                "resolved_model_revision": "model-commit",
                "resolved_tokenizer_revision": "tokenizer-commit",
            }

    def fake_evaluate(self, fitted, **kwargs):
        identity = {
            "model": self.model_name,
            "method": kwargs["method"],
            "fit_position": kwargs["fit_position"],
            "layer": kwargs["layer"],
            "phase": kwargs["phase"],
            "selected_layer": kwargs["selected_layer"],
            "dataset": "sst",
            "patch_position": "all",
        }
        return DirectionEvaluationResult(
            metrics=(
                {
                    **identity,
                    "n_directed_cases": 1,
                    "logit_difference_percent": 70.0,
                    "logit_flip_percent": 60.0,
                    "sign_flip_percent": 50.0,
                },
            ),
            patching_records=(
                {
                    **identity,
                    "case_id": "case-1",
                    "logit_flipped": True,
                    "sign_flipped": True,
                },
            ),
        )

    monkeypatch.setattr(
        "sentiment_geometry.experiments.sentiment_position.frozen_evaluation."
        "DirectionEvaluator.evaluate",
        fake_evaluate,
    )
    experiment = FrozenSentimentDirectionEvaluation(
        config,
        adapter_factory=lambda *args, **kwargs: FakeAdapter(),
    )

    output = experiment.run()

    metrics = pd.read_csv(output / "all_models_metrics.csv")
    selection = pd.read_csv(output / "all_models_layer_selection.csv")
    summary = pd.read_csv(output / "all_models_dataset_summary.csv")
    records = pd.read_csv(output / "gpt2-small" / "patching_records.csv")
    resolved = json.loads(
        (output / "gpt2-small" / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert metrics.loc[0, "logit_flip_percent"] == 60.0
    assert metrics.loc[0, "selection_dataset"] == "toy_adverbs"
    assert metrics.loc[0, "source_run_id"] == "prior"
    assert selection.loc[0, "selected_layer"] == 3
    assert summary.loc[0, "resolved_revision"] == "dataset-commit"
    assert len(records) == 1
    assert resolved["dataset_provenance"]["sst"]["config_name"] == (
        "tigges_gpt2_small_directed_pairs"
    )


def test_frozen_selection_rejects_non_adverb_provenance(tmp_path):
    source, _ = _source_run(tmp_path)
    selection_path = source / "results" / "gpt2-small" / "layer_selection.csv"
    selection = pd.read_csv(selection_path)
    selection["selection_dataset"] = "sst"
    selection.to_csv(selection_path, index=False)
    config = _config(tmp_path, source)

    try:
        load_frozen_direction_selections(config, config.models[0])
    except RuntimeError as error:
        assert "not selected exclusively on 'toy_adverbs'" in str(error)
    else:
        raise AssertionError("Expected non-ADVERB selection provenance to be rejected")


def test_frozen_selection_rejects_parent_sst_layer_mismatch(tmp_path):
    source, _ = _source_run(tmp_path)
    selected_metrics_path = source / "results" / "gpt2-small" / "selected_metrics.csv"
    selected_metrics = pd.read_csv(selected_metrics_path)
    selected_metrics["layer"] = 2
    selected_metrics.to_csv(selected_metrics_path, index=False)
    config = _config(tmp_path, source)

    try:
        load_frozen_direction_selections(config, config.models[0])
    except RuntimeError as error:
        assert "parent SST result does not use the frozen layer" in str(error)
    else:
        raise AssertionError("Expected a parent SST layer mismatch to be rejected")


def test_frozen_evaluation_reuses_completed_model_rows(monkeypatch, tmp_path):
    source, _ = _source_run(tmp_path)
    output_root = tmp_path / "results"
    completed_root = output_root / "gpt2-small"
    completed_root.mkdir(parents=True)
    completed_metrics = [
        {
            "model": "gpt2-small",
            "method": "mean_diff",
            "fit_position": "final",
            "dataset": "sst",
            "logit_flip_percent": 60.0,
        }
    ]
    completed_selections = [
        {
            "model": "gpt2-small",
            "method": "mean_diff",
            "fit_position": "final",
            "selection_dataset": "toy_adverbs",
            "selection_metric": "logit_flip_percent",
        }
    ]
    completed_summaries = [
        {"model": "gpt2-small", "dataset": "sst", "n_directed_cases": 1}
    ]
    pd.DataFrame(completed_metrics).to_csv(completed_root / "metrics.csv", index=False)
    pd.DataFrame(completed_selections).to_csv(
        completed_root / "layer_selection.csv", index=False
    )
    pd.DataFrame(completed_summaries).to_csv(
        completed_root / "dataset_summary.csv", index=False
    )

    config = FrozenDirectionEvaluationConfig(
        source_run_root=str(source),
        output_dir=str(output_root),
        models=[
            ModelConfig(
                name="qwen-0.6b",
                revision="model-commit",
                device="cpu",
                dtype="float32",
                batch_size=8,
            )
        ],
        datasets=[
            FrozenEvaluationDataset(
                name="sst",
                repo_id="owner/private-sst",
                revision="dataset-commit",
                configs={"qwen-0.6b": "tigges_qwen_0_6b_directed_pairs"},
            )
        ],
        methods=["mean_diff"],
        fit_position="final",
        selection_dataset="toy_adverbs",
        selection_metric="logit_flip_percent",
        reuse_completed_models=["gpt2-small"],
    )
    qwen_metrics = [
        {
            "model": "qwen-0.6b",
            "method": "mean_diff",
            "fit_position": "final",
            "dataset": "sst",
            "logit_flip_percent": 55.0,
        }
    ]
    qwen_selections = [
        {
            "model": "qwen-0.6b",
            "method": "mean_diff",
            "fit_position": "final",
            "selection_dataset": "toy_adverbs",
            "selection_metric": "logit_flip_percent",
        }
    ]
    qwen_summaries = [
        {"model": "qwen-0.6b", "dataset": "sst", "n_directed_cases": 1}
    ]
    monkeypatch.setattr(
        FrozenSentimentDirectionEvaluation,
        "_run_model",
        lambda self, model, root: (qwen_metrics, qwen_selections, qwen_summaries),
    )

    FrozenSentimentDirectionEvaluation(config).run()

    combined = pd.read_csv(output_root / "all_models_metrics.csv")
    assert combined["model"].tolist() == ["gpt2-small", "qwen-0.6b"]
    manifest = json.loads(
        (output_root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["models"] == ["gpt2-small", "qwen-0.6b"]
    assert manifest["evaluated_models"] == ["qwen-0.6b"]
    assert manifest["reused_completed_models"] == ["gpt2-small"]

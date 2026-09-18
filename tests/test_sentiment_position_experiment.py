import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

from sentiment_geometry.datasets.sst import load_hf_directed_pairs
from sentiment_geometry.datasets.toy_movie_review import (
    build_toy_evaluation_sets,
    load_toy_movie_review,
)
from sentiment_geometry.experiments import (
    SentimentPositionExperiment,
    SentimentPositionExperimentConfig,
)
from sentiment_geometry.experiments.sentiment_position import audit_direction_artifacts
from sentiment_geometry.experiments.sentiment_position.config import (
    SUPPORTED_TOY_EVALUATIONS,
    apply_config_overrides,
    comparison_boundaries,
)
from sentiment_geometry.experiments.sentiment_position.results import (
    ExperimentTables,
    select_layers_by_validation_metric,
)
from sentiment_geometry.models.huggingface import CausalLMAdapter, TokenizedBatch
from sentiment_geometry.persistence import RunArtifactStore

PROJECT_ROOT = Path(__file__).parents[1]


class _WordLengthTokenizer:
    """Minimal tokenizer implementing the length contracts used by data builders."""

    def __call__(self, text, *, add_special_tokens=False):
        if text.startswith(" "):
            word = text.strip()
            adverbs = {
                "amusingly",
                "blissfully",
                "buoyantly",
                "cheerfully",
                "comically",
                "delightedly",
                "euphorically",
                "gleefully",
                "humorously",
                "jokingly",
                "joyfully",
                "joyously",
                "lightheartedly",
                "merrily",
                "playfuly",
                "radiantly",
                "sportively",
                "triumphantly",
                "corruptly",
                "demonically",
                "devilishly",
                "disgustingly",
                "dreadfully",
                "fearfully",
                "fearingly",
                "frightenedly",
                "frightfully",
                "grimly",
                "gruesomely",
                "horrendously",
                "horrifically",
                "unjustly",
                "monstrously",
                "odiously",
                "obscenely",
                "outrageously",
                "perversely",
                "sadistically",
                "scandalously",
                "shamefully",
            }
            return {"input_ids": [1, 2] if word in adverbs else [1]}
        return {"input_ids": list(range(len(text.split()) + int(add_special_tokens)))}


def test_public_api_uses_domain_names():
    assert SentimentPositionExperiment.__name__ == "SentimentPositionExperiment"


def test_config_runs_all_positions_methods_and_non_embedding_layers():
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    assert [model.name for model in config.models] == ["gpt2-small", "qwen-0.6b"]
    assert config.sweep.methods == ["mean_diff", "logistic_regression", "das"]
    assert config.sweep.fit_positions == ["adjective", "verb", "summary", "final"]
    assert config.data.toy_evaluations == ["toy_adjectives", "toy_adverbs"]
    assert config.selection.das_checkpoint_dataset == "toy_adverbs"
    assert config.selection.layer_dataset == "toy_adverbs"
    assert config.selection.layer_metric == "logit_flip_percent"
    assert config.selection.final_evaluations == ["toy_adverbs", "toy_adjectives", "sst"]
    assert config.layers_for(4) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="boundary 0 is excluded"):
        config.sweep.layers = [0, 1]
        config.layers_for(4)


def test_checkpoint_and_layer_selection_reuse_the_same_dataset():
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    config.selection.layer_dataset = "toy_adjectives"
    with pytest.raises(ValueError, match="must reuse the same evaluation dataset"):
        config.validate()


def test_layer_selection_dataset_must_be_rerun_in_final_evaluations():
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    config.selection.final_evaluations = ["toy_adjectives", "sst"]
    with pytest.raises(ValueError, match="must also be included in final evaluations"):
        config.validate()


def test_selected_metrics_contains_only_post_selection_evaluations(tmp_path):
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    experiment = SentimentPositionExperiment(config)
    common = {
        "model": "gpt2-small",
        "method": "mean_diff",
        "fit_position": "adjective",
        "layer": 2,
        "selected_layer": True,
    }
    tables = ExperimentTables(
        metrics=[
            {**common, "dataset": "toy_adverbs", "phase": "layer_selection"},
            {**common, "dataset": "toy_adverbs", "phase": "selected_layer_evaluation"},
            {**common, "dataset": "toy_adjectives", "phase": "selected_layer_evaluation"},
            {**common, "dataset": "sst", "phase": "final_evaluation"},
        ]
    )
    experiment._finalize_tables(
        RunArtifactStore(tmp_path), tables, pd.DataFrame([{"selected_layer": 2}])
    )

    selected = pd.read_csv(tmp_path / "selected_metrics.csv")
    assert selected["dataset"].tolist() == ["toy_adverbs", "toy_adjectives", "sst"]
    assert "layer_selection" not in set(selected["phase"])


def test_explicit_full_run_flags_preserve_the_complete_grid():
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    args = Namespace(
        model=["gpt2-small", "qwen-0.6b"],
        device="cpu",
        dtype="float32",
        batch_size=2,
        output_dir=None,
        checkpoint_dir=None,
        seed=3,
        method=["mean_diff", "logistic_regression", "das"],
        fit_position=["adjective", "verb", "summary", "final"],
        all_non_embedding_layers=True,
        layer=None,
        logistic_c=0.5,
        logistic_max_iter=200,
        logistic_tol=1e-5,
        das_learning_rate=1e-3,
        das_weight_decay=0.0,
        das_epochs=2,
        das_batch_size=4,
        das_max_grad_norm=1.0,
        sst_repo_id=None,
        sst_revision=None,
        sst_max_directed_cases=8,
        hf_token_env="HF_TOKEN",
    )
    overridden = apply_config_overrides(config, args)
    assert overridden.sweep.layers == "all_non_embedding"
    assert overridden.sweep.methods == ["mean_diff", "logistic_regression", "das"]
    assert overridden.sweep.fit_positions == ["adjective", "verb", "summary", "final"]
    assert overridden.data.sst_max_directed_cases == 8
    assert all(model.device == "cpu" for model in overridden.models)


def test_full_run_script_passes_the_complete_experiment_grid_explicitly():
    script = (PROJECT_ROOT / "scripts/run_sentiment_position_comparison.sh").read_text()
    for required in (
        "compare-sentiment-positions",
        "--model gpt2-small",
        "--model qwen-0.6b",
        "--method mean_diff",
        "--method logistic_regression",
        "--method das",
        "--fit-position adjective",
        "--fit-position verb",
        "--fit-position summary",
        "--fit-position final",
        "--all-non-embedding-layers",
        "--batch-size 16",
        "--sst-repo-id",
        "--sst-revision",
        "--output-dir",
        "--checkpoint-dir",
    ):
        assert required in script


def test_toy_evaluation_builder_exposes_all_supported_panels():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    evaluations = build_toy_evaluation_sets(dataset, _WordLengthTokenizer(), prepend_bos=True)
    assert tuple(evaluations) == SUPPORTED_TOY_EVALUATIONS
    assert all(evaluation.pairs for evaluation in evaluations.values())
    assert len(evaluations["toy_verbs"].examples) == 8
    assert len(evaluations["toy_adverbs"].examples) == 40
    assert (
        sum(
            example.metadata["focus_word"] == "gleefully"
            for example in evaluations["toy_adverbs"].examples
        )
        == 1
    )


def test_toy_training_prompts_record_adj_vrb_and_second_movie_sum_spans():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    example = dataset.train[0]

    adjective_start, adjective_end = example.named_spans["adjective"]
    verb_start, verb_end = example.named_spans["verb"]
    summary_start, summary_end = example.named_spans["summary"]

    assert example.text[adjective_start:adjective_end] == example.metadata["adjective"]
    assert example.text[verb_start:verb_end] == example.metadata["verb"]
    assert example.text[summary_start:summary_end] == "movie"
    assert example.text[:summary_start].count("movie") == 1
    assert example.text.endswith("is")


def test_hf_directed_pair_loader_accepts_source_target_schema(monkeypatch, tmp_path):
    rows = [
        {
            "case_id": "case-1",
            "pair_id": "pair-1",
            "direction": "negative_to_positive",
            "source_example_id": "positive",
            "source_prompt": "Review Text: good Review Sentiment:",
            "source_text": "good",
            "source_label": 1,
            "target_example_id": "negative",
            "target_prompt": "Review Text: bad Review Sentiment:",
            "target_text": "bad",
            "target_label": 0,
            "split": "test",
            "pairing_model": "gpt2-small",
        }
    ]
    shard_dir = tmp_path / "commit" / "config"
    shard_dir.mkdir(parents=True)
    (shard_dir / "test-00000-of-00001.parquet").touch()
    monkeypatch.setattr(
        "sentiment_geometry.datasets.sst.snapshot_download",
        lambda *args, **kwargs: str(tmp_path / "commit"),
    )
    monkeypatch.setattr(
        "sentiment_geometry.datasets.sst.load_dataset", lambda *args, **kwargs: rows
    )
    pairs = load_hf_directed_pairs("owner/repo", config_name="config")
    assert len(pairs) == 1
    assert pairs[0].clean.example_id == "positive"
    assert pairs[0].clean.label == 1
    assert pairs[0].corrupted.example_id == "negative"
    assert pairs[0].corrupted.label == 0


def test_comparison_boundaries_are_first_floor_middle_last():
    assert comparison_boundaries(12) == (1, 6, 12)
    assert comparison_boundaries(3) == (1, 3)


def test_layers_are_selected_only_by_adverb_logit_flip():
    rows = []
    for fit_position in ("adjective", "final"):
        for layer, recovery, flip in (
            (1, 20.0, 80.0),
            (2, 90.0, 30.0),
            (3, 90.0, 20.0),
        ):
            rows.append(
                {
                    "model": "gpt2-small",
                    "method": "mean_diff",
                    "fit_position": fit_position,
                    "dataset": "toy_adverbs",
                    "phase": "layer_selection",
                    "layer": layer,
                    "logit_difference_percent": recovery,
                    "logit_flip_percent": flip,
                }
            )
    selected = select_layers_by_validation_metric(
        pd.DataFrame(rows), dataset="toy_adverbs", metric="logit_flip_percent"
    )
    assert set(selected["selected_layer"]) == {1}
    assert set(selected["selection_metric"]) == {"logit_flip_percent"}


def test_activation_position_api_supports_named_toy_positions_and_final():
    batch = TokenizedBatch(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        focus_positions=torch.tensor([1, 0]),
        named_positions={
            "adjective": torch.tensor([1, 0]),
            "verb": torch.tensor([2, 1]),
            "summary": torch.tensor([0, 1]),
        },
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "focus"), torch.tensor([1, 0])
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "final"), torch.tensor([2, 1])
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "adjective"), torch.tensor([1, 0])
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "verb"), torch.tensor([2, 1])
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "summary"), torch.tensor([0, 1])
    )


def test_direction_artifact_audit_checks_every_configured_combination(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "gpt2-small"
    direction_dir = tmp_path / "directions"
    model_dir.mkdir(parents=True)
    direction_dir.mkdir()
    (results_dir / "run_manifest.json").write_text(json.dumps({"models": ["gpt2-small"]}))
    (model_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "sweep": {
                    "methods": ["mean_diff", "das"],
                    "fit_positions": ["adjective", "verb", "summary", "final"],
                },
                "resolved_layers": [1, 2],
            }
        )
    )
    rows = []
    for position in ("adjective", "verb", "summary", "final"):
        for method in ("mean_diff", "das"):
            for layer in (1, 2):
                path = direction_dir / f"{position}-{method}-layer{layer:02d}.npz"
                path.touch()
                rows.append(
                    {
                        "fit_position": position,
                        "method": method,
                        "layer": layer,
                        "artifact_path": path,
                    }
                )
    pd.DataFrame(rows).to_csv(model_dir / "direction_metadata.csv", index=False)

    audit = audit_direction_artifacts(results_dir)

    audit.require_complete()
    assert audit.complete
    assert audit.expected_total == 16
    assert audit.summary.iloc[0].to_dict() == {
        "model": "gpt2-small",
        "expected_artifacts": 16,
        "recorded_artifacts": 16,
        "existing_artifacts": 16,
        "missing_combinations": 0,
        "unexpected_combinations": 0,
        "duplicate_records": 0,
        "missing_files": 0,
        "complete": True,
    }

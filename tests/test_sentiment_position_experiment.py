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
    REQUIRED_TOY_EVALUATIONS,
    apply_config_overrides,
    comparison_boundaries,
)
from sentiment_geometry.experiments.sentiment_position.results import select_best_layers
from sentiment_geometry.models.huggingface import CausalLMAdapter, TokenizedBatch

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


def test_config_runs_both_positions_methods_and_non_embedding_layers():
    config = SentimentPositionExperimentConfig.load(
        PROJECT_ROOT / "configs/sentiment_position_comparison.yaml"
    )
    assert [model.name for model in config.models] == ["gpt2-small", "qwen-0.6b"]
    assert config.sweep.methods == ["mean_diff", "logistic_regression", "das"]
    assert config.sweep.fit_positions == ["adjective", "final"]
    assert config.layers_for(4) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="boundary 0 is excluded"):
        config.sweep.layers = [0, 1]
        config.layers_for(4)


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
        fit_position=["adjective", "final"],
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
    assert overridden.sweep.fit_positions == ["adjective", "final"]
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
        "--fit-position final",
        "--all-non-embedding-layers",
        "--batch-size 16",
        "--sst-repo-id",
        "--sst-revision",
        "--output-dir",
        "--checkpoint-dir",
    ):
        assert required in script


def test_toy_evaluations_always_include_adjectives_verbs_and_adverbs():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    evaluations = build_toy_evaluation_sets(dataset, _WordLengthTokenizer(), prepend_bos=True)
    assert tuple(evaluations) == REQUIRED_TOY_EVALUATIONS
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


def test_best_layers_are_independent_by_position_dataset_and_metric():
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
                    "dataset": "toy_verbs",
                    "layer": layer,
                    "logit_difference_percent": recovery,
                    "logit_flip_percent": flip,
                }
            )
    best = select_best_layers(pd.DataFrame(rows))
    for _, group in best.groupby("fit_position"):
        recovery = group[group.metric == "logit_difference"].iloc[0]
        flip = group[group.metric == "logit_flip"].iloc[0]
        assert recovery.layer == 2
        assert flip.layer == 1


def test_activation_position_api_supports_focus_and_final():
    batch = TokenizedBatch(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        focus_positions=torch.tensor([1, 0]),
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "focus"), torch.tensor([1, 0])
    )
    torch.testing.assert_close(
        CausalLMAdapter.activation_positions(batch, "final"), torch.tensor([2, 1])
    )


def test_direction_artifact_audit_checks_every_configured_combination(tmp_path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "gpt2-small"
    direction_dir = tmp_path / "directions"
    model_dir.mkdir(parents=True)
    direction_dir.mkdir()
    (results_dir / "run_manifest.json").write_text(
        json.dumps({"models": ["gpt2-small"]})
    )
    (model_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "sweep": {
                    "methods": ["mean_diff", "das"],
                    "fit_positions": ["adjective", "final"],
                },
                "resolved_layers": [1, 2],
            }
        )
    )
    rows = []
    for position in ("adjective", "final"):
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
    assert audit.expected_total == 8
    assert audit.summary.iloc[0].to_dict() == {
        "model": "gpt2-small",
        "expected_artifacts": 8,
        "recorded_artifacts": 8,
        "existing_artifacts": 8,
        "missing_combinations": 0,
        "unexpected_combinations": 0,
        "duplicate_records": 0,
        "missing_files": 0,
        "complete": True,
    }

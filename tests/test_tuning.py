import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment_manifold.config import ReproductionConfig
from sentiment_manifold.data import load_toy_movie_review
from sentiment_manifold.tuning import (
    apply_selected_configuration,
    select_validation_configuration,
    split_toy_training_examples,
    tuning_grid,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_toy_validation_split_is_deterministic_stratified_and_train_only():
    toy = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    fit, validation = split_toy_training_examples(toy.train, validation_fraction=0.25, seed=7)
    repeated_fit, repeated_validation = split_toy_training_examples(
        toy.train, validation_fraction=0.25, seed=7
    )

    fit_ids = {example.example_id for example in fit}
    validation_ids = {example.example_id for example in validation}
    test_ids = {example.example_id for example in toy.test}
    assert fit_ids.isdisjoint(validation_ids)
    assert (fit_ids | validation_ids) == {example.example_id for example in toy.train}
    assert (fit_ids | validation_ids).isdisjoint(test_ids)
    assert [example.example_id for example in fit] == [
        example.example_id for example in repeated_fit
    ]
    assert [example.example_id for example in validation] == [
        example.example_id for example in repeated_validation
    ]
    assert {example.label for example in fit} == {0, 1}
    assert {example.label for example in validation} == {0, 1}


def test_tuning_grid_treats_seeds_as_repeats_not_selectable_hyperparameters():
    config = ReproductionConfig.load(PROJECT_ROOT / "configs/tuning.yaml")
    grid = tuning_grid(config, "logistic_regression")
    assert len(grid) == len(config.tuning.logistic_c) * len(config.tuning.seeds)
    assert {entry["seed"] for entry in grid} == set(config.tuning.seeds)
    assert {entry["c"] for entry in grid} == set(config.tuning.logistic_c)


def test_validation_selection_uses_mean_across_seeds():
    hp_a = json.dumps({"c": 0.1}, separators=(",", ":"))
    hp_b = json.dumps({"c": 1.0}, separators=(",", ":"))
    trials = pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "logistic_regression",
                "layer": 2,
                "hyperparameters": hp_a,
                "seed": 0,
                "toy_validation_logit_diff_percent": 80.0,
            },
            {
                "model": "gpt2-small",
                "method": "logistic_regression",
                "layer": 2,
                "hyperparameters": hp_a,
                "seed": 1,
                "toy_validation_logit_diff_percent": 90.0,
            },
            {
                "model": "gpt2-small",
                "method": "logistic_regression",
                "layer": 1,
                "hyperparameters": hp_b,
                "seed": 0,
                "toy_validation_logit_diff_percent": 84.0,
            },
            {
                "model": "gpt2-small",
                "method": "logistic_regression",
                "layer": 1,
                "hyperparameters": hp_b,
                "seed": 1,
                "toy_validation_logit_diff_percent": 84.0,
            },
        ]
    )
    selected = select_validation_configuration(
        trials, selection_metric="toy_validation_logit_diff_percent"
    )
    assert len(selected) == 1
    assert selected.iloc[0]["layer"] == 2
    assert selected.iloc[0]["hyperparameters"] == hp_a
    assert selected.iloc[0]["validation_mean"] == pytest.approx(85.0)
    assert selected.iloc[0]["n_seeds"] == 2


def test_confirmation_applies_exact_selected_configuration():
    config = ReproductionConfig.load(PROJECT_ROOT / "configs/reproduction.yaml")
    hyperparameters = {
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "epochs": 32,
        "batch_size": 64,
        "max_grad_norm": 0.5,
    }
    selected = pd.DataFrame(
        [
            {
                "model": "gpt2-small",
                "method": "das2d",
                "layer": 6,
                "hyperparameters": json.dumps(hyperparameters),
            }
        ]
    )
    frozen = apply_selected_configuration(config, selected, confirmation_seed=99)
    assert frozen.experiment.methods == ["das2d"]
    assert frozen.experiment.layers == [6]
    assert frozen.seed == 99
    assert frozen.das.learning_rate == pytest.approx(0.0003)
    assert frozen.das.weight_decay == pytest.approx(0.01)
    assert frozen.das.epochs == 32
    assert frozen.das.batch_size == 64
    assert frozen.das.max_grad_norm == pytest.approx(0.5)


def test_confirmation_rejects_a_selection_from_another_model():
    config = ReproductionConfig.load(PROJECT_ROOT / "configs/reproduction.yaml")
    selected = pd.DataFrame(
        [
            {
                "model": "qwen-0.6b",
                "method": "mean_diff",
                "layer": 6,
                "hyperparameters": "{}",
            }
        ]
    )
    with pytest.raises(ValueError, match="tuned for model"):
        apply_selected_configuration(config, selected)

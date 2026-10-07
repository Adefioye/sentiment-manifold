from pathlib import Path

import numpy as np
import pytest

from sentiment_geometry.experiments.selectivity import FixedLayerSelectivityConfig
from sentiment_geometry.experiments.selectivity.datasets import randomize_data
from sentiment_geometry.experiments.selectivity.datasets import PreparedSelectivityData
from sentiment_geometry.probes import (
    LogisticProbeConfig,
    MLP1ProbeConfig,
    balanced_label_permutation,
    evaluate_binary_probe,
    fit_logistic_probe,
    fit_mean_difference_probe,
    fit_mlp1_probe,
)
from sentiment_geometry.datasets import TextExample

PROJECT_ROOT = Path(__file__).parents[1]


def _separable_data():
    train_x = np.asarray(
        [[-3.0, 0.0], [-2.0, 0.1], [-1.0, -0.1], [1.0, 0.0], [2.0, 0.1], [3.0, -0.1]],
        dtype=np.float32,
    )
    train_y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    validation_x = np.asarray(
        [[-2.5, 0.2], [-0.5, -0.2], [0.5, 0.2], [2.5, -0.2]], dtype=np.float32
    )
    validation_y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return train_x, train_y, validation_x, validation_y


def test_balanced_label_permutation_is_reproducible_and_preserves_counts():
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1])

    first = balanced_label_permutation(labels, seed=17)
    second = balanced_label_permutation(labels, seed=17)
    different = balanced_label_permutation(labels, seed=18)

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(first, different)


def test_mean_difference_native_and_midpoint_rules_are_identical():
    train_x, train_y, validation_x, validation_y = _separable_data()
    probe = fit_mean_difference_probe(train_x, train_y)

    result = evaluate_binary_probe(
        training_midpoint_scores=probe.midpoint_scores(train_x),
        training_labels=train_y,
        native_scores=probe.native_scores(validation_x),
        midpoint_scores=probe.midpoint_scores(validation_x),
        labels=validation_y,
        native_threshold=probe.native_threshold,
    )

    assert result.native_accuracy == 1.0
    assert result.midpoint_accuracy == 1.0
    assert result.prediction_agreement == 1.0
    assert result.native_threshold == result.midpoint_threshold


def test_logistic_probe_retains_native_intercept_and_midpoint_rule():
    train_x, train_y, validation_x, validation_y = _separable_data()
    probe = fit_logistic_probe(
        train_x,
        train_y,
        config=LogisticProbeConfig(c=1.0),
        seed=3,
    )

    result = evaluate_binary_probe(
        training_midpoint_scores=probe.midpoint_scores(train_x),
        training_labels=train_y,
        native_scores=probe.native_scores(validation_x),
        midpoint_scores=probe.midpoint_scores(validation_x),
        labels=validation_y,
        native_threshold=probe.native_threshold,
    )

    assert result.native_accuracy == 1.0
    assert result.midpoint_accuracy == 1.0
    assert probe.state()["coefficient"].shape == (2,)
    assert "intercept" in probe.state()


def test_mlp1_is_reproducible_and_uses_training_only_standardization():
    train_x, train_y, validation_x, validation_y = _separable_data()
    config = MLP1ProbeConfig(
        hidden_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=3,
        max_epochs=100,
        patience=15,
    )

    first = fit_mlp1_probe(
        train_x, train_y, validation_x, validation_y, config=config, seed=9
    )
    second = fit_mlp1_probe(
        train_x, train_y, validation_x, validation_y, config=config, seed=9
    )

    assert np.allclose(first.native_scores(validation_x), second.native_scores(validation_x))
    assert np.allclose(first.state()["mean"], train_x.mean(axis=0))
    assert first.direction is None


def test_randomize_data_assigns_every_split_and_keeps_ids():
    examples = {
        role: tuple(
            TextExample(
                text=f"{role}-{index}",
                label=index % 2,
                example_id=f"{role}-{index}",
            )
            for index in range(6)
        )
        for role in ("train", "validation", "test")
    }
    prepared = PreparedSelectivityData(
        name="synthetic",
        examples=examples,
        pairs={role: tuple() for role in examples},
        answers={0: (" no",), 1: (" yes",)},
        sample_manifest=tuple(),
        pair_manifest=tuple(),
        provenance={},
    )

    randomized = randomize_data(prepared, seed=4)

    assert len(randomized.assignment_rows) == 18
    for role in examples:
        assert [row.example_id for row in randomized.examples[role]] == [
            row.example_id for row in examples[role]
        ]
        assert sorted(row.label for row in randomized.examples[role]) == [0, 0, 0, 1, 1, 1]
        assert len(randomized.pairs[role]) == 6


def test_selectivity_config_loads_modular_hyperparameters_and_requires_layers():
    config = FixedLayerSelectivityConfig.load(
        PROJECT_ROOT / "configs/selectivity/fixed_layer.yaml"
    )

    assert config.methods == ["mean_diff", "logistic_regression", "das", "mlp1"]
    assert config.random_labels.seeds == [11, 22, 33, 44, 55]
    assert len(config.logistic_regression.candidates()) == 5
    assert len(config.mlp1.candidates()) == 9
    assert config.das.objective == "answer_cross_entropy"
    assert config.das.intervention_position == "final"
    assert config.data.toy_config == str(PROJECT_ROOT / "data/toy_movie_review.yaml")

    with pytest.raises(ValueError, match="explicit residual boundary"):
        config.validate(require_layers=True)

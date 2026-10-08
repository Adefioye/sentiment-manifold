import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sentiment_geometry.datasets import TextExample
from sentiment_geometry.experiments.selectivity import (
    FixedLayerSelectivityConfig,
    run_fixed_layer_selectivity,
    run_fixed_layer_selectivity_with_frozen_hyperparameters,
    tune_fixed_layer_selectivity,
)
from sentiment_geometry.experiments.selectivity import experiment as selectivity_experiment
from sentiment_geometry.experiments.selectivity.config import (
    FixedLayerModelConfig,
    SelectivityDataConfig,
    SelectivityOutputConfig,
    SelectivityProgressConfig,
)
from sentiment_geometry.experiments.selectivity.datasets import (
    PreparedSelectivityData,
    randomize_data,
)
from sentiment_geometry.probes import (
    LogisticProbeConfig,
    MLP1ProbeConfig,
    balanced_label_permutation,
    evaluate_binary_probe,
    fit_logistic_probe,
    fit_mean_difference_probe,
    fit_mlp1_probe,
)

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

    first = fit_mlp1_probe(train_x, train_y, validation_x, validation_y, config=config, seed=9)
    second = fit_mlp1_probe(train_x, train_y, validation_x, validation_y, config=config, seed=9)

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


def test_selectivity_config_loads_fixed_dataset_layers_and_manual_trials():
    config = FixedLayerSelectivityConfig.load(PROJECT_ROOT / "configs/selectivity/fixed_layer.yaml")

    assert config.methods == ["mean_diff", "logistic_regression", "das", "mlp1"]
    assert config.random_labels.seeds == [11, 22, 33, 44, 55]
    assert len(config.logistic_regression.candidates()) == 3
    assert len(config.das.candidates()) == 3
    assert len(config.mlp1.candidates()) == 3
    assert config.das.objective == "answer_cross_entropy"
    assert config.das.intervention_position == "final"
    assert config.progress.enabled is True
    assert config.data.toy_config == str(PROJECT_ROOT / "data/toy_movie_review.yaml")
    layers = {
        model.name: {dataset: model.layer_for(dataset) for dataset in config.data.datasets}
        for model in config.models
    }
    assert layers == {
        "gpt2-small": {"toy_movie_review": 10, "full_ait": 11},
        "qwen-0.6b": {"toy_movie_review": 26, "full_ait": 26},
    }
    config.validate(require_layers=True)


def test_selectivity_run_clears_the_loaded_models_device_cache(tmp_path, monkeypatch):
    device = torch.device("cpu")
    adapter = SimpleNamespace(
        device_spec=SimpleNamespace(device=device),
        n_layers=12,
        provenance=dict,
    )
    config = FixedLayerSelectivityConfig(
        models=[FixedLayerModelConfig(name="gpt2-small", layer=10)],
        methods=["mean_diff"],
        data=SelectivityDataConfig(datasets=["toy_movie_review"]),
        output=SelectivityOutputConfig(
            output_dir=str(tmp_path),
            run_id="cache-cleanup",
            cache_activations=False,
            save_predictions=False,
        ),
        progress=SelectivityProgressConfig(enabled=False),
    )
    cleared_devices = []

    monkeypatch.setattr(
        selectivity_experiment.CausalLMAdapter,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: adapter),
    )
    monkeypatch.setattr(
        selectivity_experiment.FixedLayerSelectivityExperiment,
        "_prepare_data",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        selectivity_experiment.FixedLayerSelectivityExperiment,
        "_run_dataset",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        selectivity_experiment,
        "clear_device_cache",
        cleared_devices.append,
    )

    run_dir = run_fixed_layer_selectivity(config)

    assert run_dir == tmp_path / "cache-cleanup"
    assert cleared_devices == [device]


def test_final_stage_requires_frozen_validation_selection(tmp_path):
    config = FixedLayerSelectivityConfig(
        models=[FixedLayerModelConfig(name="gpt2-small", layer=10)],
        methods=["logistic_regression"],
        data=SelectivityDataConfig(datasets=["toy_movie_review"]),
        output=SelectivityOutputConfig(
            output_dir=str(tmp_path), run_id="staged", cache_activations=True
        ),
        progress=SelectivityProgressConfig(enabled=False),
    )

    with pytest.raises(FileNotFoundError, match="Run validation tuning first"):
        run_fixed_layer_selectivity_with_frozen_hyperparameters(config)


def test_tuning_stage_persists_selection_without_final_metrics(tmp_path, monkeypatch):
    device = torch.device("cpu")
    adapter = SimpleNamespace(
        device_spec=SimpleNamespace(device=device),
        n_layers=12,
        provenance=lambda: {"revision": "test"},
    )
    examples = {
        role: tuple(
            TextExample(text=f"{role}-{index}", label=index % 2, example_id=f"{role}-{index}")
            for index in range(4)
        )
        for role in ("train", "validation", "test")
    }
    prepared = PreparedSelectivityData(
        name="toy_movie_review",
        examples=examples,
        pairs={role: tuple() for role in examples},
        answers={0: (" no",), 1: (" yes",)},
        sample_manifest=tuple(),
        pair_manifest=tuple(),
        provenance={},
    )
    config = FixedLayerSelectivityConfig(
        models=[FixedLayerModelConfig(name="gpt2-small", layer=10)],
        methods=["mean_diff"],
        data=SelectivityDataConfig(datasets=["toy_movie_review"]),
        output=SelectivityOutputConfig(
            output_dir=str(tmp_path), run_id="staged", cache_activations=True
        ),
        progress=SelectivityProgressConfig(enabled=False),
    )
    monkeypatch.setattr(
        selectivity_experiment.CausalLMAdapter,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: adapter),
    )
    monkeypatch.setattr(
        selectivity_experiment.FixedLayerSelectivityExperiment,
        "_prepare_data",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        selectivity_experiment.FixedLayerSelectivityExperiment,
        "_collect_activations",
        lambda *args, **kwargs: {
            role: np.zeros((len(rows), 2), dtype=np.float32) for role, rows in examples.items()
        },
    )

    run_dir = tune_fixed_layer_selectivity(config)
    selections = json.loads((run_dir / "selected_hyperparameters.json").read_text())

    assert selections == {"gpt2-small": {"toy_movie_review": {"mean_diff": {}}}}
    assert (run_dir / "tuning_trials.csv").is_file()
    assert not (run_dir / "metrics.csv").exists()

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sentiment_geometry.activations import (
    extract_last_token_activations,
    extract_mean_pooled_activations,
)
from sentiment_geometry.datasets import HuggingFaceRows
from sentiment_geometry.datasets.types import CounterfactualPair, TextExample
from sentiment_geometry.evaluation import PatchingResult
from sentiment_geometry.experiments import AITValenceExperimentConfig
from sentiment_geometry.experiments.ait_valence.config import (
    AITDataConfig,
    AITSamplingConfig,
    AITValenceSweepConfig,
)
from sentiment_geometry.experiments.ait_valence.datasets import AITDatasetLoader
from sentiment_geometry.experiments.ait_valence.experiment import (
    AITValenceDirectionExperiment,
)
from sentiment_geometry.experiments.ait_valence.fitting import (
    AITDirectionFitRequest,
    AITDirectionFitService,
    FittedAITDirection,
)
from sentiment_geometry.fitting_methods import DirectionArtifact
from sentiment_geometry.fitting_methods.base import FitResult
from sentiment_geometry.fitting_methods.das import DASFitter, DASTrainingConfig
from sentiment_geometry.models import ModelConfig, TokenizedBatch
from sentiment_geometry.models.devices import DeviceSpec
from sentiment_geometry.reporting import plot_ait_valence_run

PROJECT_ROOT = Path(__file__).parents[1]


def _matched_rows(split: str, count: int) -> tuple[dict, ...]:
    return tuple(
        {
            "pair_id": f"ait-{split}-pair-{index:03d}",
            "split": split,
            "positive_example_id": f"ait-{split}-pos-{index:03d}",
            "positive_text": f"positive tweet {index}",
            "positive_prompt": f"Review Text: positive tweet {index} Review Sentiment:",
            "positive_label": 1,
            "positive_source_score": 1 + index % 3,
            "negative_example_id": f"ait-{split}-neg-{index:03d}",
            "negative_text": f"negative tweet {index}",
            "negative_prompt": f"Review Text: negative tweet {index} Review Sentiment:",
            "negative_label": 0,
            "negative_source_score": -(1 + index % 3),
        }
        for index in range(count)
    )


def _small_config(tmp_path: Path) -> AITValenceExperimentConfig:
    return AITValenceExperimentConfig(
        models=[ModelConfig(name="gpt2-small", device="cpu", dtype="float32")],
        data=AITDataConfig(revision="dataset-commit"),
        sampling=AITSamplingConfig(
            train_examples=5,
            eval_directed_cases=4,
            test_directed_cases=4,
        ),
        sweep=AITValenceSweepConfig(
            layers=[1],
            output_dir=str(tmp_path / "results"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )


def test_ait_config_loads_separate_reproducible_contract():
    config = AITValenceExperimentConfig.load(
        PROJECT_ROOT / "configs/ait_valence_directions.yaml"
    )

    assert config.sampling.train_examples == 55
    assert config.sampling.eval_directed_cases == 30
    assert config.sampling.test_directed_cases == 30
    assert config.data.matched_config == "common_matched_pairs"
    assert config.data.revision == "c53df7c117c2f433df904cfaeddb9062027f755f"
    assert config.sweep.methods == ["mean_diff", "logistic_regression", "das"]
    assert config.sweep.activation_representation == "mean_pool"
    assert config.selection.das_checkpoint_split == "eval"
    assert config.selection.layer_selection_split == "test"


def test_ait_loader_builds_deterministic_disjoint_roles(tmp_path):
    config = _small_config(tmp_path)
    rows = {
        "train": _matched_rows("train", 8),
        "validation": _matched_rows("validation", 5),
        "test": _matched_rows("test", 5),
    }

    def load_rows(repo_id, *, split, revision, **kwargs):
        assert repo_id == config.data.repo_id
        assert revision == "dataset-commit"
        return HuggingFaceRows(rows[split], revision, "resolved-dataset-commit")

    first = AITDatasetLoader(config, rows_loader=load_rows).load()
    second = AITDatasetLoader(config, rows_loader=load_rows).load()

    assert len(first.train_examples) == 5
    assert len(first.train_pairs) == 4
    assert len(first.eval_pairs) == 4
    assert len(first.test_pairs) == 4
    assert [row.example_id for row in first.train_examples] == [
        row.example_id for row in second.train_examples
    ]
    role_ids = {
        "train": {row.example_id for row in first.train_examples},
        "eval": {
            example.example_id
            for pair in first.eval_pairs
            for example in (pair.clean, pair.corrupted)
        },
        "test": {
            example.example_id
            for pair in first.test_pairs
            for example in (pair.clean, pair.corrupted)
        },
    }
    assert not (role_ids["train"] & role_ids["eval"])
    assert not (role_ids["train"] & role_ids["test"])
    assert not (role_ids["eval"] & role_ids["test"])
    assert {row["role"] for row in first.sample_manifest} == {"train", "eval", "test"}
    assert sum(row["used_by_das_training"] for row in first.sample_manifest) == 4
    assert sum(
        row["used_for_das_checkpoint_validation"] for row in first.sample_manifest
    ) == 4
    assert sum(row["used_for_layer_selection"] for row in first.sample_manifest) == 4
    assert {row["role"] for row in first.pair_manifest} == {"train", "eval", "test"}


def test_mean_pooling_excludes_padding_and_special_tokens():
    examples = [
        TextExample(text="one", label=1, example_id="one"),
        TextExample(text="two", label=0, example_id="two"),
    ]

    class FakeAdapter:
        device_spec = DeviceSpec(torch.device("cpu"), torch.float32)

        def tokenize(self, selected):
            assert len(selected) == 2
            return TokenizedBatch(
                input_ids=torch.tensor([[9, 1, 2, 0], [9, 3, 0, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                special_tokens_mask=torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]),
            )

        def boundary_activations(self, batch, layer):
            assert layer == 1
            return torch.tensor(
                [
                    [[100.0, 100.0], [1.0, 3.0], [3.0, 5.0], [0.0, 0.0]],
                    [[200.0, 200.0], [7.0, 9.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )

    pooled = extract_mean_pooled_activations(
        FakeAdapter(), examples, 1, batch_size=2, include_special_tokens=False
    )

    np.testing.assert_allclose(pooled, np.asarray([[2.0, 4.0], [7.0, 9.0]]))


def test_last_token_extraction_uses_each_prompt_final_non_padding_position():
    examples = [
        TextExample(text="one", label=1, example_id="one"),
        TextExample(text="two", label=0, example_id="two"),
    ]

    class FakeAdapter:
        device_spec = DeviceSpec(torch.device("cpu"), torch.float32)

        def tokenize(self, selected):
            assert len(selected) == 2
            return TokenizedBatch(
                input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
                attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
            )

        def boundary_activations(self, batch, layer):
            assert layer == 1
            return torch.tensor(
                [
                    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                    [[4.0, 40.0], [5.0, 50.0], [0.0, 0.0]],
                ]
            )

        @staticmethod
        def activation_positions(batch, position):
            assert position == "final"
            return batch.attention_mask.sum(dim=1).long() - 1

    extracted = extract_last_token_activations(
        FakeAdapter(), examples, 1, batch_size=2
    )

    np.testing.assert_allclose(extracted, np.asarray([[3.0, 30.0], [5.0, 50.0]]))


def test_last_token_das_trains_at_final_but_selects_checkpoint_on_all_tokens():
    request = AITDirectionFitRequest(
        examples=(),
        train_pairs=(),
        eval_pairs=(),
        activations=np.empty((0, 2)),
        labels=np.empty((0,)),
        method="das",
        layer=1,
        answers={1: (" Positive",), 0: (" Negative",)},
        activation_representation="last_token",
    )

    assert request.fit_position == "final"
    assert request.intervention_position == "final"
    assert request.checkpoint_validation_position == "all"


def test_linear_ait_direction_uses_portable_direction_artifact(tmp_path):
    config = _small_config(tmp_path)
    examples = tuple(
        TextExample(text=f"prompt {index}", label=index % 2, example_id=f"row-{index}")
        for index in range(4)
    )

    class FakeAdapter:
        hidden_size = 2

    service = AITDirectionFitService(
        config=config,
        adapter=FakeAdapter(),
        model=config.models[0],
        runtime={
            "resolved_model_revision": "model-commit",
            "resolved_tokenizer_revision": "tokenizer-commit",
        },
    )
    request = AITDirectionFitRequest(
        examples=examples,
        train_pairs=(),
        eval_pairs=(),
        activations=np.asarray(
            [[-2.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float32
        ),
        labels=np.asarray([0, 1, 0, 1]),
        method="mean_diff",
        layer=1,
        answers=config.data.answers,
    )

    fitted = service.fit(request)

    assert fitted.checkpoint_path.is_file()
    np.testing.assert_allclose(fitted.artifact.vector, np.asarray([1.0, 0.0]))
    assert fitted.artifact.metadata["domain"] == "ait_valence"
    assert fitted.artifact.metadata["representation"] == "masked_mean"
    assert fitted.artifact.metadata["orientation_reference"] == (
        "ait_train_class_mean_difference"
    )
    assert fitted.artifact.metadata["requested_dataset_revision"] == "dataset-commit"
    with np.load(fitted.checkpoint_path, allow_pickle=False) as saved:
        header = json.loads(str(saved["metadata"]))
    assert header["metadata"]["model_revision"] == "model-commit"


def test_ait_das_trains_at_last_token_but_validates_on_all_tokens(
    monkeypatch, tmp_path
):
    config = _small_config(tmp_path)
    config.sweep.activation_representation = "last_token"
    positive = TextExample(text="positive", label=1, example_id="positive")
    negative = TextExample(text="negative", label=0, example_id="negative")
    pair = CounterfactualPair(clean=positive, corrupted=negative)
    observed = {}

    class FakeAdapter:
        hidden_size = 2

    class FakeValidationEvaluator:
        def __init__(self, adapter, pairs, *, position, **kwargs):
            observed["validation_position"] = position

        @staticmethod
        def evaluate(direction):
            return PatchingResult(
                recovery=0.5,
                flip_rate=0.5,
                sign_flip_rate=0.5,
                corrupted_margin=-1.0,
                clean_margin=1.0,
                patched_margin=0.0,
                corrupted_accuracy=0.0,
                clean_accuracy=1.0,
                patched_accuracy=0.5,
                n_pairs=1,
                records=(),
            )

    class FakeDASFitter:
        def __init__(self, training_config):
            pass

        @staticmethod
        def fit(adapter, pairs, *, position, epoch_validator, **kwargs):
            observed["training_position"] = position
            validation = epoch_validator(np.asarray([1.0, 0.0]))
            observed["validation_loss"] = validation["validation_loss"]
            return FitResult(
                method="das",
                direction=np.asarray([1.0, 0.0]),
                diagnostics={"selected_epoch": 0, "loss_history": []},
            )

    monkeypatch.setattr(
        "sentiment_geometry.experiments.ait_valence.fitting."
        "DirectionalPatchingEvaluator",
        FakeValidationEvaluator,
    )
    monkeypatch.setattr(
        "sentiment_geometry.experiments.ait_valence.fitting.DASFitter",
        FakeDASFitter,
    )
    service = AITDirectionFitService(
        config=config,
        adapter=FakeAdapter(),
        model=config.models[0],
        runtime={
            "resolved_model_revision": "model-commit",
            "resolved_tokenizer_revision": "tokenizer-commit",
        },
    )
    service.fit(
        AITDirectionFitRequest(
            examples=(positive, negative),
            train_pairs=(pair,),
            eval_pairs=(pair,),
            activations=np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
            labels=np.asarray([1, 0]),
            method="das",
            layer=1,
            answers=config.data.answers,
            activation_representation="last_token",
        )
    )

    assert observed == {
        "validation_position": "all",
        "training_position": "final",
        "validation_loss": 0.5,
    }


def test_das_supports_all_non_padding_token_interventions():
    positive = TextExample(text="positive", label=1, example_id="positive")
    negative = TextExample(text="negative", label=0, example_id="negative")
    pairs = [
        CounterfactualPair(clean=positive, corrupted=negative),
        CounterfactualPair(clean=negative, corrupted=positive),
    ]

    class FakeModel:
        def __init__(self, adapter):
            self.adapter = adapter

        def eval(self):
            return self

        def __call__(self, *, input_ids, attention_mask, use_cache):
            hidden = self.adapter.hidden(input_ids)
            if self.adapter.editor is not None:
                hidden = self.adapter.editor(hidden)
            score = hidden[..., 0]
            return type("Output", (), {"logits": torch.stack((-score, score), dim=-1)})()

    class FakeAdapter:
        hidden_size = 2
        device_spec = DeviceSpec(torch.device("cpu"), torch.float32)

        def __init__(self):
            self.editor = None
            self.model = FakeModel(self)

        @staticmethod
        def hidden(input_ids):
            polarity = input_ids.float() * 2.0 - 1.0
            return torch.stack((polarity, torch.ones_like(polarity)), dim=-1)

        def tokenize(self, examples):
            labels = torch.tensor([[row.label, row.label] for row in examples])
            return TokenizedBatch(
                input_ids=labels,
                attention_mask=torch.ones_like(labels),
                special_tokens_mask=torch.zeros_like(labels),
            )

        def boundary_activations(self, batch, layer):
            assert layer == 1
            return self.hidden(batch.input_ids)

        @staticmethod
        def last_positions(attention_mask):
            return attention_mask.sum(dim=1).long() - 1

        @staticmethod
        def single_token_id(answer):
            return 1 if answer == " Positive" else 0

        @contextmanager
        def edit_boundary(self, layer, editor):
            assert layer == 1
            self.editor = editor
            try:
                yield
            finally:
                self.editor = None

    result = DASFitter(
        DASTrainingConfig(epochs=2, batch_size=2, learning_rate=0.01, seed=0)
    ).fit(
        FakeAdapter(),
        pairs,
        layer=1,
        answers={1: (" Positive",), 0: (" Negative",)},
        position="all",
    )

    assert result.direction.shape == (2,)
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.diagnostics["selected_epoch"] in {0, 1}


@pytest.mark.parametrize(
    (
        "activation_representation",
        "linear_patch_position",
        "expected_representations",
    ),
    [
        ("mean_pool", "all", {"masked_mean", "all_tokens"}),
        ("last_token", "final", {"last_token"}),
    ],
)
def test_ait_experiment_smoke_writes_and_selects_all_three_methods(
    monkeypatch,
    tmp_path,
    activation_representation,
    linear_patch_position,
    expected_representations,
):
    config = _small_config(tmp_path)
    config.sweep.layers = [1, 2]
    config.sweep.activation_representation = activation_representation
    rows = {
        "train": _matched_rows("train", 8),
        "validation": _matched_rows("validation", 5),
        "test": _matched_rows("test", 5),
    }

    def load_rows(repo_id, *, split, revision, **kwargs):
        return HuggingFaceRows(rows[split], revision, "resolved-dataset-commit")

    data_loader = AITDatasetLoader(config, rows_loader=load_rows)

    class FakeAdapter:
        n_layers = 2
        device_spec = DeviceSpec(torch.device("cpu"), torch.float32)

        @staticmethod
        def provenance():
            return {
                "resolved_model_revision": "model-commit",
                "resolved_tokenizer_revision": "tokenizer-commit",
            }

    extraction_calls = []

    def fake_extract(representation, examples, layer):
        extraction_calls.append((representation, layer))
        return np.asarray(
            [[float(row.label), float(layer)] for row in examples]
        )

    monkeypatch.setattr(
        "sentiment_geometry.experiments.ait_valence.experiment."
        "extract_mean_pooled_activations",
        lambda adapter, examples, layer, **kwargs: fake_extract(
            "mean_pool", examples, layer
        ),
    )
    monkeypatch.setattr(
        "sentiment_geometry.experiments.ait_valence.experiment."
        "extract_last_token_activations",
        lambda adapter, examples, layer, **kwargs: fake_extract(
            "last_token", examples, layer
        ),
    )

    def fake_fit(self, request):
        representation = request.representation
        path = (
            Path(self.config.sweep.checkpoint_dir)
            / self.model.name
            / request.method
            / f"layer{request.layer:02d}.npz"
        )
        artifact = DirectionArtifact(
            method=request.method,
            model_name=self.model.hub_name,
            layer=request.layer,
            vector=np.asarray([1.0, 0.0]),
            metadata={
                "representation": representation,
                "fit_position": request.fit_position,
                "activation_representation": request.activation_representation,
                "intervention_position": request.intervention_position,
                "training_intervention_position": request.intervention_position,
                "checkpoint_validation_position": (
                    request.checkpoint_validation_position
                ),
                "selected_epoch": 1 if request.method == "das" else None,
                "loss_history": [
                    {
                        "epoch": 1,
                        "validation_loss": 0.2,
                        "selected_epoch": True,
                    }
                ]
                if request.method == "das"
                else [],
            },
        )
        artifact.save(path)
        return FittedAITDirection(artifact, path)

    monkeypatch.setattr(AITDirectionFitService, "fit", fake_fit)

    evaluator_positions = []

    class FakeEvaluator:
        def __init__(self, adapter, pairs, *, layer, **kwargs):
            self.position = kwargs["position"]
            evaluator_positions.append((layer, self.position))
            self.layer = layer
            self.pairs = pairs

        def evaluate(self, direction):
            flip_rate = 0.8 if self.layer == 2 else 0.2
            return PatchingResult(
                recovery=flip_rate,
                flip_rate=flip_rate,
                sign_flip_rate=flip_rate / 2,
                corrupted_margin=-1.0,
                clean_margin=1.0,
                patched_margin=flip_rate,
                corrupted_accuracy=0.0,
                clean_accuracy=1.0,
                patched_accuracy=flip_rate,
                n_pairs=len(self.pairs),
                records=tuple(
                    {
                        "clean_id": pair.clean.example_id,
                        "corrupted_id": pair.corrupted.example_id,
                    }
                    for pair in self.pairs
                ),
            )

    monkeypatch.setattr(
        "sentiment_geometry.experiments.ait_valence.experiment."
        "DirectionalPatchingEvaluator",
        FakeEvaluator,
    )
    experiment = AITValenceDirectionExperiment(
        config,
        adapter_factory=lambda *args, **kwargs: FakeAdapter(),
        dataset_loader=data_loader,
    )

    output = experiment.run()

    metrics = pd.read_csv(output / "gpt2-small" / "metrics.csv")
    selection = pd.read_csv(output / "gpt2-small" / "layer_selection.csv")
    combined = pd.read_csv(output / "all_models_metrics.csv")
    similarities = pd.read_csv(output / "all_models_direction_similarities.csv")
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert len(metrics) == 6
    assert len(combined) == 6
    assert len(similarities) == 18
    assert set(similarities["layer"]) == {1, 2}
    assert np.allclose(similarities["absolute_cosine"], 1.0)
    assert set(metrics["representation"]) == expected_representations
    assert set(metrics.loc[metrics["method"] == "das", "patch_position"]) == {"all"}
    assert set(metrics.loc[metrics["method"] != "das", "patch_position"]) == {
        linear_patch_position
    }
    assert set(metrics["activation_representation"]) == {
        activation_representation
    }
    expected_fit_positions = (
        {"final"} if activation_representation == "last_token" else expected_representations
    )
    assert set(metrics["fit_position"]) == expected_fit_positions
    assert extraction_calls == [
        (activation_representation, 1),
        (activation_representation, 2),
    ]
    expected_evaluator_positions = (
        [(1, "all"), (2, "all")]
        if linear_patch_position == "all"
        else [(1, "final"), (1, "all"), (2, "final"), (2, "all")]
    )
    assert evaluator_positions == expected_evaluator_positions
    assert set(selection["method"]) == {"mean_diff", "logistic_regression", "das"}
    assert set(selection["selected_layer"]) == {2}
    assert not selection["selection_is_final_evaluation"].any()
    assert manifest["train_examples"] == 5
    assert manifest["eval_directed_cases"] == 4
    assert manifest["test_directed_cases"] == 4
    assert manifest["test_is_layer_selection_not_final_evaluation"] is True
    if activation_representation == "last_token":
        figures = plot_ait_valence_run(output, figure_dir=tmp_path / "figures")
        assert len(figures) == 5
        assert all(path.is_file() for path in figures)

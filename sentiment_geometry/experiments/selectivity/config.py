"""Configuration contracts for fixed-layer random-label selectivity experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...models import ModelConfig
from ...probes import LogisticProbeConfig, MLP1ProbeConfig

SUPPORTED_METHODS = ("mean_diff", "logistic_regression", "das", "mlp1")
SUPPORTED_DATASETS = ("toy_movie_review", "full_ait")


@dataclass
class FixedLayerModelConfig(ModelConfig):
    layer: int | None = None
    layers: dict[str, int] = field(default_factory=dict)

    def layer_for(self, dataset: str) -> int:
        selected = self.layers.get(dataset, self.layer)
        if selected is None:
            raise ValueError(
                f"Set an explicit residual boundary for {self.name}/{dataset}"
            )
        return int(selected)


@dataclass
class SelectivityDataConfig:
    datasets: list[str] = field(default_factory=lambda: list(SUPPORTED_DATASETS))
    toy_config: str = "data/toy_movie_review.yaml"
    toy_validation_fraction: float = 0.2
    ait_repo_id: str = "kokolamba/sentiment-manifold-ait-valence-binary"
    ait_revision: str | None = None
    ait_model_matched_configs: dict[str, str] = field(
        default_factory=lambda: {
            "gpt2-small": "gpt2_small_matched_pairs",
            "qwen-0.6b": "qwen_0_6b_matched_pairs",
        }
    )
    ait_train_split: str = "train"
    ait_validation_split: str = "validation"
    ait_test_split: str = "test"
    hf_token_env: str = "HF_TOKEN"
    positive_answers: list[str] = field(default_factory=lambda: [" Positive"])
    negative_answers: list[str] = field(default_factory=lambda: [" Negative"])


@dataclass
class RandomLabelConfig:
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    preserve_class_counts: bool = True
    generate_per_split: bool = True


@dataclass
class LogisticSearchConfig:
    trials: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"c": 0.1, "penalty": "l2", "class_weight": None},
            {"c": 1.0, "penalty": "l2", "class_weight": None},
            {"c": 10.0, "penalty": "l2", "class_weight": None},
        ]
    )
    solver: str = "liblinear"
    max_iter: int = 5000
    tol: float = 1e-4
    class_weight: str | None = None

    def candidates(self) -> tuple[LogisticProbeConfig, ...]:
        return tuple(
            LogisticProbeConfig(
                c=float(trial.get("c", 1.0)),
                penalty=str(trial.get("penalty", "l2")),
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=trial.get("class_weight", self.class_weight),
            )
            for trial in self.trials
        )


@dataclass
class DASSelectivityConfig:
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    max_grad_norm: float = 1.0
    objective: str = "answer_cross_entropy"
    intervention_position: str = "final"
    checkpoint_metric: str = "validation_loss"
    trials: list[dict[str, Any]] = field(default_factory=list)

    def candidates(self) -> tuple["DASSelectivityConfig", ...]:
        trials = self.trials or [{}]
        return tuple(
            DASSelectivityConfig(
                epochs=int(trial.get("epochs", self.epochs)),
                learning_rate=float(trial.get("learning_rate", self.learning_rate)),
                weight_decay=float(trial.get("weight_decay", self.weight_decay)),
                batch_size=self.batch_size,
                max_grad_norm=self.max_grad_norm,
                objective=self.objective,
                intervention_position=self.intervention_position,
                checkpoint_metric=self.checkpoint_metric,
                trials=[],
            )
            for trial in trials
        )


@dataclass
class MLP1SearchConfig:
    trials: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"hidden_size": 4, "learning_rate": 1e-3, "weight_decay": 1e-2},
            {"hidden_size": 10, "learning_rate": 1e-3, "weight_decay": 1e-2},
            {"hidden_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-2},
        ]
    )
    dropout: float = 0.0
    batch_size: int = 16
    max_epochs: int = 500
    patience: int = 20
    min_delta: float = 1e-4
    standardize_inputs: bool = True

    def candidates(self) -> tuple[MLP1ProbeConfig, ...]:
        return tuple(
            MLP1ProbeConfig(
                hidden_size=int(trial.get("hidden_size", 10)),
                learning_rate=float(trial.get("learning_rate", 1e-3)),
                weight_decay=float(trial.get("weight_decay", 1e-2)),
                dropout=self.dropout,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                min_delta=self.min_delta,
                standardize_inputs=self.standardize_inputs,
            )
            for trial in self.trials
        )


@dataclass
class SelectivityOutputConfig:
    output_dir: str = "outputs/fixed-layer-selectivity"
    run_id: str | None = None
    cache_activations: bool = True
    save_predictions: bool = True


@dataclass
class SelectivityProgressConfig:
    enabled: bool = True
    leave_completed: bool = False


@dataclass
class FixedLayerSelectivityConfig:
    seed: int = 0
    models: list[FixedLayerModelConfig] = field(default_factory=list)
    methods: list[str] = field(default_factory=lambda: list(SUPPORTED_METHODS))
    data: SelectivityDataConfig = field(default_factory=SelectivityDataConfig)
    random_labels: RandomLabelConfig = field(default_factory=RandomLabelConfig)
    logistic_regression: LogisticSearchConfig = field(default_factory=LogisticSearchConfig)
    das: DASSelectivityConfig = field(default_factory=DASSelectivityConfig)
    mlp1: MLP1SearchConfig = field(default_factory=MLP1SearchConfig)
    output: SelectivityOutputConfig = field(default_factory=SelectivityOutputConfig)
    progress: SelectivityProgressConfig = field(default_factory=SelectivityProgressConfig)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "FixedLayerSelectivityConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        method_files = raw.get("hyperparameter_files", {})

        def section(name: str) -> dict[str, Any]:
            inline = dict(raw.get(name, {}))
            referenced = method_files.get(name)
            if not referenced:
                return inline
            referenced_path = Path(referenced)
            if not referenced_path.is_absolute():
                referenced_path = source.parent / referenced_path
            loaded = yaml.safe_load(referenced_path.read_text(encoding="utf-8")) or {}
            return {**loaded, **inline}

        config = cls(
            seed=int(raw.get("seed", 0)),
            models=[FixedLayerModelConfig(**item) for item in raw.get("models", [])],
            methods=list(raw.get("methods", SUPPORTED_METHODS)),
            data=SelectivityDataConfig(**raw.get("data", {})),
            random_labels=RandomLabelConfig(**raw.get("random_labels", {})),
            logistic_regression=LogisticSearchConfig(**section("logistic_regression")),
            das=DASSelectivityConfig(**section("das")),
            mlp1=MLP1SearchConfig(**section("mlp1")),
            output=SelectivityOutputConfig(**raw.get("output", {})),
            progress=SelectivityProgressConfig(**raw.get("progress", {})),
            source_path=source,
        )
        project_root = source.parents[2]
        toy_path = Path(config.data.toy_config)
        if not toy_path.is_absolute():
            config.data.toy_config = str((project_root / toy_path).resolve())
        output_path = Path(config.output.output_dir)
        if not output_path.is_absolute():
            config.output.output_dir = str((project_root / output_path).resolve())
        config.validate(require_layers=False)
        return config

    def validate(self, *, require_layers: bool = True) -> None:
        if not self.models:
            raise ValueError("At least one model must be configured")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Configured model names must be unique")
        missing_layers = [
            f"{model.name}/{dataset}"
            for model in self.models
            for dataset in self.data.datasets
            if dataset not in model.layers and model.layer is None
        ]
        if require_layers and missing_layers:
            raise ValueError(
                "Set an explicit residual boundary for each model before running: "
                f"{missing_layers}"
            )
        configured_layers = [
            value
            for model in self.models
            for value in ([model.layer] if model.layer is not None else [])
            + list(model.layers.values())
        ]
        if any(layer < 1 for layer in configured_layers):
            raise ValueError("Fixed layers must exclude embedding boundary 0")
        unknown_layer_datasets = sorted(
            {
                dataset
                for model in self.models
                for dataset in model.layers
                if dataset not in SUPPORTED_DATASETS
            }
        )
        if unknown_layer_datasets:
            raise ValueError(
                f"Unsupported datasets in model layer mappings: {unknown_layer_datasets}"
            )
        unknown_methods = sorted(set(self.methods) - set(SUPPORTED_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported selectivity methods: {unknown_methods}")
        unknown_datasets = sorted(set(self.data.datasets) - set(SUPPORTED_DATASETS))
        if unknown_datasets:
            raise ValueError(f"Unsupported selectivity datasets: {unknown_datasets}")
        if not 0.0 < self.data.toy_validation_fraction < 0.5:
            raise ValueError("Toy validation fraction must be between 0 and 0.5")
        if len(self.random_labels.seeds) < 2:
            raise ValueError("Use at least two paired real/random-label seeds")
        if len(set(self.random_labels.seeds)) != len(self.random_labels.seeds):
            raise ValueError("Random-label seeds must be unique")
        if not self.random_labels.preserve_class_counts:
            raise ValueError("This experiment requires class-count-preserving permutations")
        if not self.random_labels.generate_per_split:
            raise ValueError("Random-label assignments must be explicit for every split")
        if self.das.intervention_position != "final":
            raise ValueError("Fixed-layer selectivity requires final-token DAS")
        if self.das.objective not in {"normalized_logit_difference", "answer_cross_entropy"}:
            raise ValueError("Unsupported DAS objective")
        if not self.logistic_regression.trials:
            raise ValueError("Logistic-regression trial list must not be empty")
        if not self.mlp1.candidates():
            raise ValueError("MLP-1 trial list must not be empty")
        self._validate_manual_trials()

    def _validate_manual_trials(self) -> None:
        specifications = (
            (
                "logistic_regression",
                self.logistic_regression.trials,
                {"c", "penalty", "class_weight"},
            ),
            (
                "das",
                self.das.trials,
                {"learning_rate", "weight_decay", "epochs"},
            ),
            (
                "mlp1",
                self.mlp1.trials,
                {"hidden_size", "learning_rate", "weight_decay"},
            ),
        )
        for name, trials, allowed in specifications:
            for trial in trials:
                unknown = sorted(set(trial) - allowed)
                if unknown:
                    raise ValueError(f"Unsupported {name} trial parameters: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path) if self.source_path else None
        return result


__all__ = [
    "DASSelectivityConfig",
    "FixedLayerModelConfig",
    "FixedLayerSelectivityConfig",
    "LogisticSearchConfig",
    "MLP1SearchConfig",
    "RandomLabelConfig",
    "SelectivityDataConfig",
    "SelectivityOutputConfig",
    "SelectivityProgressConfig",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
]

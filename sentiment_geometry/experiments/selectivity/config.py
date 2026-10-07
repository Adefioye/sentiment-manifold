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
    c: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    solver: str = "liblinear"
    max_iter: int = 5000
    tol: float = 1e-4
    class_weight: str | None = None

    def candidates(self) -> tuple[LogisticProbeConfig, ...]:
        return tuple(
            LogisticProbeConfig(
                c=value,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
            )
            for value in self.c
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


@dataclass
class MLP1SearchConfig:
    hidden_size: list[int] = field(default_factory=lambda: [2, 4, 10, 16, 32, 64])
    learning_rate: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    weight_decay: list[float] = field(default_factory=lambda: [0.0, 1e-3, 1e-2, 0.1])
    dropout: list[float] = field(default_factory=lambda: [0.0, 0.2])
    batch_size: int = 16
    max_epochs: int = 500
    patience: int = 20
    min_delta: float = 1e-4
    standardize_inputs: bool = True

    def candidates(self) -> tuple[MLP1ProbeConfig, ...]:
        return tuple(
            MLP1ProbeConfig(
                hidden_size=hidden,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                min_delta=self.min_delta,
                standardize_inputs=self.standardize_inputs,
            )
            for hidden in self.hidden_size
            for learning_rate in self.learning_rate
            for weight_decay in self.weight_decay
            for dropout in self.dropout
        )


@dataclass
class SelectivityOutputConfig:
    output_dir: str = "outputs/fixed-layer-selectivity"
    cache_activations: bool = True
    save_predictions: bool = True


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
        missing_layers = [model.name for model in self.models if model.layer is None]
        if require_layers and missing_layers:
            raise ValueError(
                "Set an explicit residual boundary for each model before running: "
                f"{missing_layers}"
            )
        if any(model.layer is not None and model.layer < 1 for model in self.models):
            raise ValueError("Fixed layers must exclude embedding boundary 0")
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
        if not self.logistic_regression.c:
            raise ValueError("Logistic-regression C grid must not be empty")
        if not self.mlp1.candidates():
            raise ValueError("MLP-1 grid must not be empty")

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
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
]

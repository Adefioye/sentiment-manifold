"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


MODEL_ALIASES = {
    "gpt2-small": "gpt2",
    "qwen-0.6b": "Qwen/Qwen3-0.6B-Base",
    "gemma-2b": "google/gemma-2b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


@dataclass
class ModelConfig:
    name: str = "gpt2-small"
    revision: str | None = None
    prepend_bos: bool = True
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16

    @property
    def hub_name(self) -> str:
        return MODEL_ALIASES.get(self.name, self.name)


@dataclass
class DataConfig:
    toy_config: str = "data/toy_movie_review.yaml"
    sst_root: str = "../eliciting-latent-sentiment/stanfordSentimentTreebank"
    sst_processed_dir: str = "data/processed/sst-pythia-1.4b"
    sst_dataset_config: str = "tigges_pythia_correct"
    sst_split: str = "test"
    sst_max_examples: int | None = None
    sst_max_pairs: int | None = None
    openwebtext_dataset: str = "stas/openwebtext-10k"
    openwebtext_split: str = "train"
    openwebtext_max_samples: int = 128
    openwebtext_sequence_length: int = 128
    openwebtext_random_seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class ExperimentConfig:
    layers: str | list[int] = "all"
    methods: list[str] = field(
        default_factory=lambda: [
            "mean_diff",
            "kmeans",
            "logistic_regression",
            "pca",
            "das",
            "das2d",
            "das3d",
            "random",
        ]
    )
    openwebtext_resample_ablation: bool = False
    # Compatibility with configurations created before the exploratory
    # diagnostic received its own explicit name.
    evaluate_openwebtext: bool = False
    output_dir: str = "outputs/results"
    checkpoint_dir: str = "checkpoints"
    resume: bool = True

    @property
    def resample_ablation_enabled(self) -> bool:
        return self.openwebtext_resample_ablation or self.evaluate_openwebtext


@dataclass
class FittingConfig:
    kmeans_n_init: int = 10
    logistic_c: float = 1.0
    logistic_solver: str = "liblinear"
    logistic_max_iter: int = 1000
    logistic_tol: float = 1e-4


@dataclass
class DASConfig:
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # Upstream fit_directions.py uses 128 for GPT-2 Small. Model extensions
    # should override this explicitly in their own configuration.
    batch_size: int = 128
    max_grad_norm: float = 1.0
    implementation: str = "tigges_rotation"


@dataclass
class TuningConfig:
    validation_fraction: float = 0.25
    selection_metric: str = "toy_validation_logit_diff_percent"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    kmeans_n_init: list[int] = field(default_factory=lambda: [10, 50])
    logistic_c: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    das_learning_rate: list[float] = field(default_factory=lambda: [3e-4, 1e-3])
    das_weight_decay: list[float] = field(default_factory=lambda: [0.0])
    das_epochs: list[int] = field(default_factory=lambda: [32, 64])
    das_batch_size: list[int] = field(default_factory=lambda: [128])
    das_max_grad_norm: list[float] = field(default_factory=lambda: [1.0])


@dataclass
class ReproductionConfig:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    das: DASConfig = field(default_factory=DASConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ReproductionConfig":
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        experiment = dict(raw.get("experiment", {}))
        # Configurations from the earlier single-metric selection pipeline may
        # still contain this key. Table 1 reporting now uses four fixed paper
        # metrics and intentionally ignores that legacy setting.
        experiment.pop("selection_metric", None)
        if os.environ.get("SENTIMENT_MANIFOLD_OUTPUT_DIR"):
            experiment["output_dir"] = os.environ["SENTIMENT_MANIFOLD_OUTPUT_DIR"]
        if os.environ.get("SENTIMENT_MANIFOLD_CHECKPOINT_DIR"):
            experiment["checkpoint_dir"] = os.environ["SENTIMENT_MANIFOLD_CHECKPOINT_DIR"]
        cfg = cls(
            seed=int(raw.get("seed", 0)),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            experiment=ExperimentConfig(**experiment),
            fitting=FittingConfig(**raw.get("fitting", {})),
            das=DASConfig(**raw.get("das", {})),
            tuning=TuningConfig(**raw.get("tuning", {})),
            source_path=path,
        )
        cfg._resolve_paths(path.parent.parent)
        return cfg

    def _resolve_paths(self, project_root: Path) -> None:
        for attribute in ("toy_config", "sst_root", "sst_processed_dir"):
            value = Path(getattr(self.data, attribute))
            if not value.is_absolute():
                setattr(self.data, attribute, str((project_root / value).resolve()))
        for attribute in ("output_dir", "checkpoint_dir"):
            value = Path(getattr(self.experiment, attribute))
            if not value.is_absolute():
                setattr(self.experiment, attribute, str((project_root / value).resolve()))

    def layers_for(self, n_layers: int) -> list[int]:
        if self.experiment.layers == "all":
            return list(range(n_layers + 1))
        layers = [int(layer) for layer in self.experiment.layers]
        invalid = [layer for layer in layers if layer < 0 or layer > n_layers]
        if invalid:
            raise ValueError(f"Layer boundaries must be in [0, {n_layers}], got {invalid}")
        return layers

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        result = asdict(self)
        result["source_path"] = str(self.source_path) if self.source_path else None
        return result

"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MODEL_ALIASES = {
    "gpt2-small": "gpt2",
    "qwen-0.6b": "Qwen/Qwen3-0.6B-Base",
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
    sst_split: str = "dev"
    sst_max_examples: int | None = None
    sst_max_pairs: int | None = 128
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
    selection_metric: str = "sst_patch_recovery"
    openwebtext_resample_ablation: bool = False
    # Compatibility with configurations created before the exploratory
    # diagnostic received its own explicit name.
    evaluate_openwebtext: bool = False
    output_dir: str = "outputs"
    resume: bool = True

    @property
    def resample_ablation_enabled(self) -> bool:
        return self.openwebtext_resample_ablation or self.evaluate_openwebtext


@dataclass
class DASConfig:
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    max_grad_norm: float = 1.0
    implementation: str = "tigges_rotation"


@dataclass
class ReproductionConfig:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    das: DASConfig = field(default_factory=DASConfig)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ReproductionConfig":
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        cfg = cls(
            seed=int(raw.get("seed", 0)),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            experiment=ExperimentConfig(**raw.get("experiment", {})),
            das=DASConfig(**raw.get("das", {})),
            source_path=path,
        )
        cfg._resolve_paths(path.parent.parent)
        return cfg

    def _resolve_paths(self, project_root: Path) -> None:
        for attribute in ("toy_config", "sst_root"):
            value = Path(getattr(self.data, attribute))
            if not value.is_absolute():
                setattr(self.data, attribute, str((project_root / value).resolve()))
        output = Path(self.experiment.output_dir)
        if not output.is_absolute():
            self.experiment.output_dir = str((project_root / output).resolve())

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

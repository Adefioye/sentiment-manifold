"""Configuration contracts for sentiment fitting-position comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...fitting_methods.config import DASConfig, FittingConfig
from ...models.config import ModelConfig

SUPPORTED_FITTING_METHODS = ("mean_diff", "logistic_regression", "das")
SUPPORTED_FITTING_POSITIONS = ("adjective", "final")
REQUIRED_TOY_EVALUATIONS = ("toy_adjectives", "toy_verbs", "toy_adverbs")


@dataclass
class EvaluationDataConfig:
    toy_config: str = "data/toy_movie_review.yaml"
    sst_repo_id: str = "kokolamba/sentiment-manifold-sst-pythia-2.8b"
    sst_revision: str | None = None
    sst_split: str = "test"
    sst_configs: dict[str, str] = field(
        default_factory=lambda: {
            "gpt2-small": "tigges_gpt2_small_directed_pairs",
            "qwen-0.6b": "tigges_qwen_0_6b_directed_pairs",
        }
    )
    sst_max_directed_cases: int | None = None
    hf_token_env: str = "HF_TOKEN"


@dataclass
class SweepConfig:
    layers: str | list[int] = "all_non_embedding"
    methods: list[str] = field(default_factory=lambda: list(SUPPORTED_FITTING_METHODS))
    fit_positions: list[str] = field(default_factory=lambda: list(SUPPORTED_FITTING_POSITIONS))
    output_dir: str = "outputs/sentiment-position-comparison"
    checkpoint_dir: str = "checkpoints/sentiment-position-comparison"
    resume: bool = True


@dataclass
class SentimentPositionExperimentConfig:
    seed: int = 0
    models: list[ModelConfig] = field(default_factory=list)
    data: EvaluationDataConfig = field(default_factory=EvaluationDataConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    das: DASConfig = field(default_factory=DASConfig)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> SentimentPositionExperimentConfig:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = [ModelConfig(**item) for item in raw.get("models", [])]
        if not models:
            raise ValueError("A sentiment-position experiment requires at least one model")
        config = cls(
            seed=int(raw.get("seed", 0)),
            models=models,
            data=EvaluationDataConfig(**raw.get("data", {})),
            sweep=SweepConfig(**raw.get("sweep", {})),
            fitting=FittingConfig(**raw.get("fitting", {})),
            das=DASConfig(**raw.get("das", {})),
            source_path=path,
        )
        config._resolve_paths(path.parent.parent)
        config.validate()
        return config

    def _resolve_paths(self, project_root: Path) -> None:
        toy_path = Path(self.data.toy_config)
        if not toy_path.is_absolute():
            self.data.toy_config = str((project_root / toy_path).resolve())
        for attribute in ("output_dir", "checkpoint_dir"):
            value = Path(getattr(self.sweep, attribute))
            if not value.is_absolute():
                setattr(self.sweep, attribute, str((project_root / value).resolve()))

    def validate(self) -> None:
        if not self.sweep.methods:
            raise ValueError("At least one fitting method is required")
        if not self.sweep.fit_positions:
            raise ValueError("At least one fitting position is required")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Configured model names must be unique")
        unknown_methods = sorted(set(self.sweep.methods) - set(SUPPORTED_FITTING_METHODS))
        if unknown_methods:
            raise ValueError(
                f"Fitting methods must be {SUPPORTED_FITTING_METHODS}; got {unknown_methods}"
            )
        unknown_positions = sorted(set(self.sweep.fit_positions) - set(SUPPORTED_FITTING_POSITIONS))
        if unknown_positions:
            raise ValueError(
                f"Fitting positions must be {SUPPORTED_FITTING_POSITIONS}; got {unknown_positions}"
            )
        missing_sst = [
            model.name for model in self.models if model.name not in self.data.sst_configs
        ]
        if missing_sst:
            raise ValueError(f"Missing SST directed-pair configurations for models: {missing_sst}")
        if self.data.sst_max_directed_cases is not None and self.data.sst_max_directed_cases < 1:
            raise ValueError("sst_max_directed_cases must be positive when provided")

    def layers_for(self, n_layers: int) -> list[int]:
        if self.sweep.layers == "all_non_embedding":
            return list(range(1, n_layers + 1))
        layers = list(dict.fromkeys(int(layer) for layer in self.sweep.layers))
        invalid = [layer for layer in layers if layer < 1 or layer > n_layers]
        if invalid:
            raise ValueError(
                f"Embedding boundary 0 is excluded; boundaries must be in [1, {n_layers}], "
                f"got {invalid}"
            )
        return layers

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_path"] = str(self.source_path) if self.source_path else None
        return result


def comparison_boundaries(n_layers: int) -> tuple[int, ...]:
    """Return first, floor-middle, and last non-embedding residual boundaries."""

    return tuple(dict.fromkeys((1, n_layers // 2, n_layers)))


def apply_config_overrides(
    config: SentimentPositionExperimentConfig, args: Any
) -> SentimentPositionExperimentConfig:
    """Apply command-line overrides while preserving model-specific revisions."""

    selected_models = getattr(args, "model", None)
    if selected_models:
        lookup = {model.name: model for model in config.models}
        missing = sorted(set(selected_models) - set(lookup))
        if missing:
            raise ValueError(f"Models are not defined in the config: {missing}")
        config.models = [lookup[name] for name in dict.fromkeys(selected_models)]
    for model in config.models:
        if getattr(args, "device", None):
            model.device = args.device
        if getattr(args, "dtype", None):
            model.dtype = args.dtype
        if getattr(args, "batch_size", None) is not None:
            model.batch_size = int(args.batch_size)
    if getattr(args, "output_dir", None):
        config.sweep.output_dir = str(Path(args.output_dir).resolve())
    if getattr(args, "checkpoint_dir", None):
        config.sweep.checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    if getattr(args, "seed", None) is not None:
        config.seed = int(args.seed)
    if getattr(args, "method", None):
        config.sweep.methods = list(dict.fromkeys(args.method))
    if getattr(args, "fit_position", None):
        config.sweep.fit_positions = list(dict.fromkeys(args.fit_position))
    if getattr(args, "all_non_embedding_layers", False):
        config.sweep.layers = "all_non_embedding"
    elif getattr(args, "layer", None):
        config.sweep.layers = list(dict.fromkeys(args.layer))
    for argument, attribute, cast in (
        ("logistic_c", "logistic_c", float),
        ("logistic_max_iter", "logistic_max_iter", int),
        ("logistic_tol", "logistic_tol", float),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config.fitting, attribute, cast(value))
    for argument, attribute, cast in (
        ("das_learning_rate", "learning_rate", float),
        ("das_weight_decay", "weight_decay", float),
        ("das_epochs", "epochs", int),
        ("das_batch_size", "batch_size", int),
        ("das_max_grad_norm", "max_grad_norm", float),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config.das, attribute, cast(value))
    if getattr(args, "sst_repo_id", None):
        config.data.sst_repo_id = args.sst_repo_id
    if getattr(args, "sst_revision", None):
        config.data.sst_revision = args.sst_revision
    if getattr(args, "sst_max_directed_cases", None) is not None:
        config.data.sst_max_directed_cases = int(args.sst_max_directed_cases)
    if getattr(args, "hf_token_env", None):
        config.data.hf_token_env = args.hf_token_env
    config.validate()
    return config


__all__ = [
    "REQUIRED_TOY_EVALUATIONS",
    "SUPPORTED_FITTING_METHODS",
    "SUPPORTED_FITTING_POSITIONS",
    "SentimentPositionExperimentConfig",
    "apply_config_overrides",
    "comparison_boundaries",
]

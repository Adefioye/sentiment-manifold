"""Configuration contracts for AIT valence-direction experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...fitting_methods.config import DASConfig, FittingConfig
from ...models.config import ModelConfig

SUPPORTED_AIT_METHODS = ("mean_diff", "logistic_regression", "das")
SUPPORTED_AIT_ACTIVATION_REPRESENTATIONS = ("mean_pool", "last_token")


@dataclass
class AITDataConfig:
    repo_id: str = "kokolamba/sentiment-manifold-ait-valence-binary"
    revision: str | None = None
    matched_config: str = "common_matched_pairs"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    hf_token_env: str = "HF_TOKEN"
    positive_answers: list[str] = field(default_factory=lambda: [" Positive"])
    negative_answers: list[str] = field(default_factory=lambda: [" Negative"])

    @property
    def answers(self) -> dict[int, tuple[str, ...]]:
        return {1: tuple(self.positive_answers), 0: tuple(self.negative_answers)}


@dataclass
class AITSamplingConfig:
    train_examples: int = 55
    eval_directed_cases: int = 30
    test_directed_cases: int = 30


@dataclass
class AITValenceSweepConfig:
    layers: str | list[int] = "all_non_embedding"
    methods: list[str] = field(default_factory=lambda: list(SUPPORTED_AIT_METHODS))
    activation_representation: str = "mean_pool"
    output_dir: str = "outputs/ait-valence-directions"
    checkpoint_dir: str = "checkpoints/ait-valence-directions"
    resume: bool = True
    include_special_tokens_in_mean_pool: bool = False


@dataclass
class AITValenceSelectionConfig:
    das_checkpoint_split: str = "eval"
    das_checkpoint_metric: str = "validation_loss"
    layer_selection_split: str = "test"
    layer_selection_metric: str = "logit_flip_percent"


@dataclass
class AITValenceExperimentConfig:
    seed: int = 0
    models: list[ModelConfig] = field(default_factory=list)
    data: AITDataConfig = field(default_factory=AITDataConfig)
    sampling: AITSamplingConfig = field(default_factory=AITSamplingConfig)
    sweep: AITValenceSweepConfig = field(default_factory=AITValenceSweepConfig)
    selection: AITValenceSelectionConfig = field(default_factory=AITValenceSelectionConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    das: DASConfig = field(default_factory=DASConfig)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> AITValenceExperimentConfig:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = cls(
            seed=int(raw.get("seed", 0)),
            models=[ModelConfig(**item) for item in raw.get("models", [])],
            data=AITDataConfig(**raw.get("data", {})),
            sampling=AITSamplingConfig(**raw.get("sampling", {})),
            sweep=AITValenceSweepConfig(**raw.get("sweep", {})),
            selection=AITValenceSelectionConfig(**raw.get("selection", {})),
            fitting=FittingConfig(**raw.get("fitting", {})),
            das=DASConfig(**raw.get("das", {})),
            source_path=path,
        )
        config._resolve_paths(path.parent.parent)
        config.validate()
        return config

    def _resolve_paths(self, project_root: Path) -> None:
        for attribute in ("output_dir", "checkpoint_dir"):
            value = Path(getattr(self.sweep, attribute))
            if not value.is_absolute():
                setattr(self.sweep, attribute, str((project_root / value).resolve()))

    def validate(self) -> None:
        if not self.models:
            raise ValueError("An AIT valence experiment requires at least one model")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Configured model names must be unique")
        if not self.data.repo_id or not self.data.matched_config:
            raise ValueError("AIT repository and matched-pair configuration are required")
        splits = [self.data.train_split, self.data.eval_split, self.data.test_split]
        if len(set(splits)) != 3:
            raise ValueError("AIT train, eval, and test source splits must be distinct")
        if not self.data.positive_answers or not self.data.negative_answers:
            raise ValueError("Positive and negative answer tokens must both be configured")
        if self.sampling.train_examples < 2:
            raise ValueError("AIT training requires at least two examples")
        for name in ("eval_directed_cases", "test_directed_cases"):
            value = int(getattr(self.sampling, name))
            if value < 2 or value % 2:
                raise ValueError(f"{name} must be a positive even number")
        if not self.sweep.methods:
            raise ValueError("At least one fitting method is required")
        if len(self.sweep.methods) != len(set(self.sweep.methods)):
            raise ValueError("Fitting methods must be unique")
        unknown = sorted(set(self.sweep.methods) - set(SUPPORTED_AIT_METHODS))
        if unknown:
            raise ValueError(f"Unsupported AIT fitting methods: {unknown}")
        if (
            self.sweep.activation_representation
            not in SUPPORTED_AIT_ACTIVATION_REPRESENTATIONS
        ):
            raise ValueError(
                "AIT activation_representation must be one of "
                f"{SUPPORTED_AIT_ACTIVATION_REPRESENTATIONS}"
            )
        if self.selection.das_checkpoint_split != "eval":
            raise ValueError("DAS checkpoints must be selected on the AIT eval role")
        if self.selection.das_checkpoint_metric != "validation_loss":
            raise ValueError("DAS checkpoint metric must be 'validation_loss'")
        if self.selection.layer_selection_split != "test":
            raise ValueError("AIT layers must be selected on the disjoint test role")
        if self.selection.layer_selection_metric != "logit_flip_percent":
            raise ValueError("AIT layer-selection metric must be 'logit_flip_percent'")
        if self.das.implementation != "tigges_rotation":
            raise ValueError("AIT DAS requires das.implementation=tigges_rotation")

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

    def representation_for(self, method: str) -> str:
        """Return the saved direction representation for one fitting method."""

        if method not in SUPPORTED_AIT_METHODS:
            raise ValueError(f"Unsupported AIT fitting method: {method!r}")
        if self.sweep.activation_representation == "last_token":
            return "last_token"
        return "all_tokens" if method == "das" else "masked_mean"

    def intervention_position(self) -> str:
        """Map the configured representation to the causal patching position."""

        return "final" if self.sweep.activation_representation == "last_token" else "all"


def apply_ait_valence_overrides(
    config: AITValenceExperimentConfig, args: Any
) -> AITValenceExperimentConfig:
    """Apply CLI/notebook overrides without weakening the split contract."""

    selected_models = getattr(args, "model", None)
    if selected_models:
        lookup = {model.name: model for model in config.models}
        missing = sorted(set(selected_models) - set(lookup))
        if missing:
            raise ValueError(f"Models are not defined in the AIT config: {missing}")
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
    if getattr(args, "hf_token_env", None):
        config.data.hf_token_env = str(args.hf_token_env)
    if getattr(args, "method", None):
        config.sweep.methods = list(dict.fromkeys(args.method))
    if getattr(args, "ait_activation_representation", None):
        config.sweep.activation_representation = args.ait_activation_representation
    if getattr(args, "all_non_embedding_layers", False):
        config.sweep.layers = "all_non_embedding"
    elif getattr(args, "layer", None):
        config.sweep.layers = list(dict.fromkeys(args.layer))
    config.validate()
    return config


__all__ = [
    "SUPPORTED_AIT_ACTIVATION_REPRESENTATIONS",
    "SUPPORTED_AIT_METHODS",
    "AITDataConfig",
    "AITSamplingConfig",
    "AITValenceExperimentConfig",
    "AITValenceSelectionConfig",
    "AITValenceSweepConfig",
    "apply_ait_valence_overrides",
]

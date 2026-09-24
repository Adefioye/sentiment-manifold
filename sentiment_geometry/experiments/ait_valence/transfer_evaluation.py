"""Configure locked cross-dataset evaluation of AIT valence directions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ...models import ModelConfig
from ..sentiment_position.config import (
    SUPPORTED_FITTING_METHODS,
    SUPPORTED_FITTING_POSITIONS,
)
from ..sentiment_position.frozen_evaluation import (
    FrozenDirectionEvaluationConfig,
    FrozenEvaluationDataset,
)


@dataclass
class AITValenceTransferConfig:
    """Reusable plan for evaluating validation-selected AIT directions OOD."""

    source_experiment_name: str
    source_run_id: str
    output_experiment_name: str
    models: list[ModelConfig]
    datasets: list[FrozenEvaluationDataset]
    methods: list[str] = field(
        default_factory=lambda: ["mean_diff", "das"]
    )
    fit_position: str = "final"
    selection_dataset: str = "ait_eval"
    selection_metric: str = "logit_flip_percent"
    source_evaluation_dataset: str = "ait_test"
    source_direction_domain: str = "ait_valence"
    source_activation_representation: str = "last_token"
    method_patch_positions: dict[str, str] = field(
        default_factory=lambda: {
            "mean_diff": "final",
            "das": "all",
        }
    )
    hf_token_env: str = "HF_TOKEN"
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> AITValenceTransferConfig:
        """Load and validate an AIT transfer-evaluation plan from YAML."""

        resolved_path = Path(path).resolve()
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        source = raw.get("source", {})
        evaluation = raw.get("evaluation", {})
        config = cls(
            source_experiment_name=str(source.get("experiment_name", "")),
            source_run_id=str(source.get("run_id", "")),
            output_experiment_name=str(evaluation.get("experiment_name", "")),
            models=[ModelConfig(**item) for item in raw.get("models", [])],
            datasets=[FrozenEvaluationDataset(**item) for item in raw.get("datasets", [])],
            methods=list(evaluation.get("methods", [])),
            fit_position=str(evaluation.get("fit_position", "")),
            selection_dataset=str(evaluation.get("selection_dataset", "")),
            selection_metric=str(evaluation.get("selection_metric", "")),
            source_evaluation_dataset=str(
                source.get("evaluation_dataset", "")
            ),
            source_direction_domain=str(source.get("direction_domain", "")),
            source_activation_representation=str(
                source.get("activation_representation", "")
            ),
            method_patch_positions=dict(
                evaluation.get("method_patch_positions", {})
            ),
            hf_token_env=str(evaluation.get("hf_token_env", "")),
            source_path=resolved_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate the static plan without requiring the source run to be mounted."""

        for name, value in (
            ("source experiment name", self.source_experiment_name),
            ("source run ID", self.source_run_id),
            ("output experiment name", self.output_experiment_name),
            ("source evaluation dataset", self.source_evaluation_dataset),
            ("source direction domain", self.source_direction_domain),
            (
                "source activation representation",
                self.source_activation_representation,
            ),
        ):
            if not value:
                raise ValueError(f"AIT transfer {name} cannot be empty")
        if not self.models:
            raise ValueError("AIT transfer evaluation requires at least one model")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("AIT transfer model names must be unique")
        if not self.datasets:
            raise ValueError("AIT transfer evaluation requires at least one dataset")
        dataset_names = [dataset.name for dataset in self.datasets]
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("AIT transfer dataset names must be unique")
        if set(self.method_patch_positions) != set(self.methods):
            raise ValueError(
                "AIT transfer patch positions must be explicit for every method"
            )
        unknown_methods = sorted(set(self.methods) - set(SUPPORTED_FITTING_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported AIT transfer methods: {unknown_methods}")
        if self.fit_position not in SUPPORTED_FITTING_POSITIONS:
            raise ValueError(
                f"Unsupported AIT transfer fit position: {self.fit_position!r}"
            )
        invalid_patch_positions = {
            method: position
            for method, position in self.method_patch_positions.items()
            if position not in {"all", "final"}
        }
        if invalid_patch_positions:
            raise ValueError(
                f"Unsupported AIT transfer patch positions: {invalid_patch_positions}"
            )
        for dataset in self.datasets:
            dataset.validate(model_names)

    def source_run_root(
        self,
        storage_root: str | Path,
        *,
        source_run_id: str | None = None,
    ) -> Path:
        """Resolve the completed full-AIT run under a storage root."""

        run_id = source_run_id or self.source_run_id
        if not run_id:
            raise ValueError("Source run ID cannot be empty")
        return (
            Path(storage_root)
            / self.source_experiment_name
            / "runs"
            / run_id
        )

    def build_evaluation_config(
        self,
        *,
        storage_root: str | Path,
        output_dir: str | Path,
        source_run_id: str | None = None,
        evaluated_model_names: Sequence[str] | None = None,
        reuse_completed_models: Sequence[str] = (),
        device: str | None = None,
        dtype: str | None = None,
        batch_sizes: Mapping[str, int] | None = None,
    ) -> FrozenDirectionEvaluationConfig:
        """Materialize the generic frozen evaluator without changing selection."""

        configured = {model.name: model for model in self.models}
        evaluated_names = list(evaluated_model_names or configured)
        reused_names = list(reuse_completed_models)
        requested_names = evaluated_names + reused_names
        unknown = sorted(set(requested_names) - set(configured))
        if unknown:
            raise ValueError(f"Unknown AIT transfer models: {unknown}")
        if set(evaluated_names) & set(reused_names):
            raise ValueError("A model cannot be both evaluated and reused")
        if set(requested_names) != set(configured):
            raise ValueError(
                "Every configured model must be either evaluated or reused"
            )
        if len(requested_names) != len(set(requested_names)):
            raise ValueError("Evaluated and reused model names must be unique")
        unknown_batch_sizes = sorted(set(batch_sizes or {}) - set(evaluated_names))
        if unknown_batch_sizes:
            raise ValueError(
                f"Batch sizes were provided for inactive models: {unknown_batch_sizes}"
            )

        active_models: list[ModelConfig] = []
        for name in evaluated_names:
            model = replace(configured[name])
            if device is not None:
                model.device = device
            if dtype is not None:
                model.dtype = dtype
            if batch_sizes and name in batch_sizes:
                model.batch_size = int(batch_sizes[name])
            active_models.append(model)

        config = FrozenDirectionEvaluationConfig(
            source_run_root=str(
                self.source_run_root(
                    storage_root, source_run_id=source_run_id
                )
            ),
            output_dir=str(Path(output_dir)),
            models=active_models,
            datasets=list(self.datasets),
            methods=list(self.methods),
            fit_position=self.fit_position,
            selection_dataset=self.selection_dataset,
            selection_metric=self.selection_metric,
            source_evaluation_dataset=self.source_evaluation_dataset,
            source_direction_domain=self.source_direction_domain,
            source_activation_representation=self.source_activation_representation,
            method_patch_positions=dict(self.method_patch_positions),
            hf_token_env=self.hf_token_env,
            reuse_completed_models=reused_names,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, environment-independent plan as plain values."""

        return {
            "source": {
                "experiment_name": self.source_experiment_name,
                "run_id": self.source_run_id,
                "evaluation_dataset": self.source_evaluation_dataset,
                "direction_domain": self.source_direction_domain,
                "activation_representation": self.source_activation_representation,
            },
            "evaluation": {
                "experiment_name": self.output_experiment_name,
                "methods": list(self.methods),
                "fit_position": self.fit_position,
                "selection_dataset": self.selection_dataset,
                "selection_metric": self.selection_metric,
                "method_patch_positions": dict(self.method_patch_positions),
                "hf_token_env": self.hf_token_env,
            },
            "models": [model.__dict__.copy() for model in self.models],
            "datasets": [dataset.__dict__.copy() for dataset in self.datasets],
            "source_path": str(self.source_path) if self.source_path else None,
        }


__all__ = ["AITValenceTransferConfig"]

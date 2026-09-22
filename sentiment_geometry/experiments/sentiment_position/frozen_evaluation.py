"""Evaluate previously selected sentiment directions on frozen OOD datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...datasets import load_hf_directed_pairs
from ...datasets.types import CounterfactualPair, TextExample
from ...fitting_methods import DirectionArtifact
from ...models import CausalLMAdapter, ModelConfig, clear_device_cache, resolve_device
from ...persistence import RunArtifactStore
from .config import SUPPORTED_FITTING_METHODS, SUPPORTED_FITTING_POSITIONS
from .datasets import CausalEvaluation, SST_ANSWERS
from .evaluation import DirectionEvaluator
from .fitting import FittedDirection

AdapterFactory = Callable[..., CausalLMAdapter]


@dataclass(frozen=True)
class FrozenEvaluationDataset:
    """One materialized directed-pair dataset used only for evaluation."""

    name: str
    repo_id: str
    configs: Mapping[str, str]
    revision: str | None = None
    split: str = "test"
    max_directed_cases: int | None = None

    def validate(self, model_names: Sequence[str]) -> None:
        if not self.name:
            raise ValueError("Evaluation dataset name cannot be empty")
        if not self.repo_id:
            raise ValueError(f"Evaluation dataset {self.name!r} requires a repo_id")
        missing = sorted(set(model_names) - set(self.configs))
        if missing:
            raise ValueError(
                f"Evaluation dataset {self.name!r} has no directed-pair configs for {missing}"
            )
        if self.max_directed_cases is not None and self.max_directed_cases < 1:
            raise ValueError("max_directed_cases must be positive when provided")


@dataclass
class FrozenDirectionEvaluationConfig:
    """Configuration for reusing a prior run's selected direction artifacts."""

    source_run_root: str
    output_dir: str
    models: list[ModelConfig]
    datasets: list[FrozenEvaluationDataset]
    methods: list[str] = field(
        default_factory=lambda: ["mean_diff", "logistic_regression", "das"]
    )
    fit_position: str = "final"
    selection_dataset: str = "toy_adverbs"
    selection_metric: str = "logit_flip_percent"
    hf_token_env: str = "HF_TOKEN"

    def validate(self) -> None:
        source_root = Path(self.source_run_root)
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source run does not exist: {source_root}")
        manifest_path = source_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source run manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise RuntimeError(
                f"Source run is not completed: {manifest.get('status')!r}"
            )
        if not self.models:
            raise ValueError("At least one model is required")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Configured model names must be unique")
        if not self.datasets:
            raise ValueError("At least one evaluation dataset is required")
        dataset_names = [dataset.name for dataset in self.datasets]
        if len(dataset_names) != len(set(dataset_names)):
            raise ValueError("Evaluation dataset names must be unique")
        if not self.methods:
            raise ValueError("At least one fitting method is required")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("Fitting methods must be unique")
        unknown_methods = sorted(set(self.methods) - set(SUPPORTED_FITTING_METHODS))
        if unknown_methods:
            raise ValueError(f"Unsupported fitting methods: {unknown_methods}")
        if self.fit_position not in SUPPORTED_FITTING_POSITIONS:
            raise ValueError(f"Unsupported fitting position: {self.fit_position!r}")
        if not self.selection_dataset or not self.selection_metric:
            raise ValueError("Selection dataset and metric must be explicit")
        if not self.hf_token_env:
            raise ValueError("Hugging Face token environment variable cannot be empty")
        for dataset in self.datasets:
            dataset.validate(model_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_root": self.source_run_root,
            "output_dir": self.output_dir,
            "models": [asdict(model) for model in self.models],
            "datasets": [asdict(dataset) for dataset in self.datasets],
            "methods": list(self.methods),
            "fit_position": self.fit_position,
            "selection_dataset": self.selection_dataset,
            "selection_metric": self.selection_metric,
            "hf_token_env": self.hf_token_env,
        }


@dataclass(frozen=True)
class FrozenDirectionSelection:
    """A frozen layer-selection row joined to its reusable direction checkpoint."""

    model: str
    method: str
    fit_position: str
    selected_layer: int
    selection_dataset: str
    selection_metric: str
    selection_value_percent: float
    checkpoint_path: Path
    artifact: DirectionArtifact


def _read_csv(path: Path, *, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table


def _checkpoint_candidates(
    *,
    source_root: Path,
    selection_row: pd.Series,
    metadata_row: pd.Series,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    relative = metadata_row.get("artifact_relative_path")
    if isinstance(relative, str) and relative.strip():
        candidates.append(source_root / "directions" / relative)
    for value in (
        selection_row.get("direction_checkpoint"),
        metadata_row.get("artifact_path"),
    ):
        if isinstance(value, str) and value.strip():
            candidates.append(Path(value))
    return tuple(dict.fromkeys(path.expanduser() for path in candidates))


def load_frozen_direction_selections(
    config: FrozenDirectionEvaluationConfig,
    model: ModelConfig,
) -> tuple[FrozenDirectionSelection, ...]:
    """Load and validate the source run's frozen selections for one model."""

    source_root = Path(config.source_run_root)
    model_results = source_root / "results" / model.name
    selection = _read_csv(
        model_results / "layer_selection.csv",
        required_columns={
            "method",
            "fit_position",
            "selected_layer",
            "selection_dataset",
            "selection_metric",
            "selection_value_percent",
        },
    )
    direction_metadata = _read_csv(
        model_results / "direction_metadata.csv",
        required_columns={"method", "fit_position", "layer", "artifact_path"},
    )
    selected_metrics = _read_csv(
        model_results / "selected_metrics.csv",
        required_columns={"method", "fit_position", "dataset", "layer"},
    )
    if "model" in selection.columns:
        selection = selection[selection["model"] == model.name]
    if "model" in direction_metadata.columns:
        direction_metadata = direction_metadata[direction_metadata["model"] == model.name]
    if "model" in selected_metrics.columns:
        selected_metrics = selected_metrics[selected_metrics["model"] == model.name]
    frozen = selection[
        (selection["fit_position"] == config.fit_position)
        & selection["method"].isin(config.methods)
    ].copy()
    if set(frozen["method"]) != set(config.methods) or len(frozen) != len(config.methods):
        raise RuntimeError(
            f"Expected one {config.fit_position!r} selection for each method in "
            f"{config.methods}; found {frozen[['method', 'fit_position']].to_dict('records')}"
        )
    if set(frozen["selection_dataset"]) != {config.selection_dataset}:
        raise RuntimeError(
            f"Frozen layers were not selected exclusively on {config.selection_dataset!r}"
        )
    if set(frozen["selection_metric"]) != {config.selection_metric}:
        raise RuntimeError(
            f"Frozen layers were not selected exclusively by {config.selection_metric!r}"
        )

    selections: list[FrozenDirectionSelection] = []
    for _, row in frozen.sort_values("method", kind="stable").iterrows():
        layer = int(row["selected_layer"])
        method = str(row["method"])
        sst_metrics = selected_metrics[
            (selected_metrics["method"] == method)
            & (selected_metrics["fit_position"] == config.fit_position)
            & (selected_metrics["dataset"] == "sst")
        ]
        if len(sst_metrics) != 1 or int(sst_metrics.iloc[0]["layer"]) != layer:
            raise RuntimeError(
                f"The parent SST result does not use the frozen layer for "
                f"{model.name}/{method}/{config.fit_position}/layer{layer}"
            )
        matches = direction_metadata[
            (direction_metadata["method"] == method)
            & (direction_metadata["fit_position"] == config.fit_position)
            & (direction_metadata["layer"].astype(int) == layer)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one direction metadata row for "
                f"{model.name}/{method}/{config.fit_position}/layer{layer}; found {len(matches)}"
            )
        metadata_row = matches.iloc[0]
        checkpoint_path = next(
            (
                candidate.resolve()
                for candidate in _checkpoint_candidates(
                    source_root=source_root,
                    selection_row=row,
                    metadata_row=metadata_row,
                )
                if candidate.is_file()
            ),
            None,
        )
        if checkpoint_path is None:
            candidates = _checkpoint_candidates(
                source_root=source_root,
                selection_row=row,
                metadata_row=metadata_row,
            )
            raise FileNotFoundError(
                f"No direction checkpoint exists for {model.name}/{method}/layer{layer}: "
                f"{[str(path) for path in candidates]}"
            )
        artifact = DirectionArtifact.load(checkpoint_path)
        if artifact.method != method or artifact.layer != layer:
            raise RuntimeError(
                f"Direction checkpoint identity mismatch at {checkpoint_path}: "
                f"method={artifact.method!r}, layer={artifact.layer}"
            )
        if artifact.model_name != model.hub_name:
            raise RuntimeError(
                f"Direction checkpoint model mismatch at {checkpoint_path}: "
                f"{artifact.model_name!r} != {model.hub_name!r}"
            )
        if artifact.metadata.get("fit_position") != config.fit_position:
            raise RuntimeError(
                f"Direction checkpoint fit-position mismatch at {checkpoint_path}"
            )
        if not np.isclose(float(np.linalg.norm(artifact.vector)), 1.0, atol=1e-5):
            raise RuntimeError(f"Direction is not unit norm: {checkpoint_path}")
        selections.append(
            FrozenDirectionSelection(
                model=model.name,
                method=method,
                fit_position=config.fit_position,
                selected_layer=layer,
                selection_dataset=str(row["selection_dataset"]),
                selection_metric=str(row["selection_metric"]),
                selection_value_percent=float(row["selection_value_percent"]),
                checkpoint_path=checkpoint_path,
                artifact=artifact,
            )
        )
    method_order = {method: index for index, method in enumerate(config.methods)}
    return tuple(sorted(selections, key=lambda item: method_order[item.method]))


def _hugging_face_token(variable_name: str) -> str | None:
    token = os.environ.get(variable_name)
    if token:
        return token
    token_path = os.environ.get("HF_TOKEN_PATH")
    if not token_path:
        return None
    path = Path(token_path).expanduser()
    if not path.is_file():
        raise ValueError(f"HF_TOKEN_PATH does not point to a readable file: {path}")
    return path.read_text(encoding="utf-8").strip() or None


def _examples_from_pairs(
    pairs: Sequence[CounterfactualPair],
) -> tuple[TextExample, ...]:
    return tuple(
        {
            example.example_id: example
            for pair in pairs
            for example in (pair.clean, pair.corrupted)
        }.values()
    )


class FrozenSentimentDirectionEvaluation:
    """Run confirmation/OOD evaluations without refitting or reselecting."""

    def __init__(
        self,
        config: FrozenDirectionEvaluationConfig,
        *,
        adapter_factory: AdapterFactory = CausalLMAdapter.from_pretrained,
    ) -> None:
        config.validate()
        self.config = config
        self.adapter_factory = adapter_factory

    def run(self) -> Path:
        output_root = Path(self.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        source_manifest = json.loads(
            (Path(self.config.source_run_root) / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        all_metrics: list[dict[str, Any]] = []
        all_selections: list[dict[str, Any]] = []
        all_summaries: list[dict[str, Any]] = []
        for model in self.config.models:
            metrics, selections, summaries = self._run_model(model, output_root)
            all_metrics.extend(metrics)
            all_selections.extend(selections)
            all_summaries.extend(summaries)
        store = RunArtifactStore(output_root)
        store.write_rows("all_models_metrics.csv", all_metrics)
        store.write_rows("all_models_layer_selection.csv", all_selections)
        store.write_rows("all_models_dataset_summary.csv", all_summaries)
        store.write_json(
            "evaluation_manifest.json",
            {
                "source_run_id": source_manifest.get("run_id"),
                "source_run_root": str(Path(self.config.source_run_root).resolve()),
                "source_run_status": source_manifest.get("status"),
                "selection_dataset": self.config.selection_dataset,
                "selection_metric": self.config.selection_metric,
                "fit_position": self.config.fit_position,
                "methods": list(self.config.methods),
                "models": [model.name for model in self.config.models],
                "datasets": [dataset.name for dataset in self.config.datasets],
            },
        )
        return output_root

    def _run_model(
        self,
        model: ModelConfig,
        output_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        selections = load_frozen_direction_selections(self.config, model)
        device_spec = resolve_device(model.device, model.dtype)
        adapter = self.adapter_factory(
            model.hub_name,
            device_spec,
            revision=model.revision,
            prepend_bos=model.prepend_bos,
        )
        evaluations, summaries, dataset_provenance = self._load_evaluations(model)
        evaluator = DirectionEvaluator(
            adapter=adapter,
            model_name=model.name,
            batch_size=model.batch_size,
            evaluations=evaluations,
        )
        source_run_id = Path(self.config.source_run_root).name
        metric_rows: list[dict[str, Any]] = []
        patching_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        direction_rows: list[dict[str, Any]] = []
        for selected in selections:
            fitted = FittedDirection(
                artifact=selected.artifact,
                checkpoint_path=selected.checkpoint_path,
            )
            result = evaluator.evaluate(
                fitted,
                fit_position=selected.fit_position,
                method=selected.method,
                layer=selected.selected_layer,
                evaluation_names=tuple(evaluations),
                phase="frozen_ood_evaluation",
                selected_layer=True,
            )
            provenance = {
                "source_run_id": source_run_id,
                "selection_dataset": selected.selection_dataset,
                "selection_metric": selected.selection_metric,
                "selection_value_percent": selected.selection_value_percent,
                "source_direction_checkpoint": str(selected.checkpoint_path),
            }
            metric_rows.extend({**row, **provenance} for row in result.metrics)
            patching_rows.extend({**row, **provenance} for row in result.patching_records)
            selection_rows.append(
                {
                    "model": selected.model,
                    "method": selected.method,
                    "fit_position": selected.fit_position,
                    "selected_layer": selected.selected_layer,
                    "selection_dataset": selected.selection_dataset,
                    "selection_metric": selected.selection_metric,
                    "selection_value_percent": selected.selection_value_percent,
                    "source_run_id": source_run_id,
                    "source_direction_checkpoint": str(selected.checkpoint_path),
                }
            )
            direction_rows.append(
                {
                    "model": selected.model,
                    "method": selected.method,
                    "fit_position": selected.fit_position,
                    "layer": selected.selected_layer,
                    "unit_norm": float(np.linalg.norm(selected.artifact.vector)),
                    "artifact_schema_version": selected.artifact.metadata.get(
                        "artifact_schema_version"
                    ),
                    "model_revision": selected.artifact.metadata.get("model_revision"),
                    "tokenizer_revision": selected.artifact.metadata.get(
                        "tokenizer_revision"
                    ),
                    "source_run_id": source_run_id,
                    "source_direction_checkpoint": str(selected.checkpoint_path),
                }
            )

        model_store = RunArtifactStore(output_root / model.name)
        model_store.write_rows("metrics.csv", metric_rows)
        model_store.write_rows("patching_records.csv", patching_rows)
        model_store.write_rows("layer_selection.csv", selection_rows)
        model_store.write_rows("dataset_summary.csv", summaries)
        model_store.write_rows("direction_metadata.csv", direction_rows)
        model_store.write_json(
            "resolved_config.json",
            {
                **self.config.to_dict(),
                "active_model": asdict(model),
                "runtime": dict(adapter.provenance()),
                "source_run_id": source_run_id,
                "dataset_provenance": dataset_provenance,
            },
        )
        del adapter
        clear_device_cache(device_spec.device)
        return metric_rows, selection_rows, summaries

    def _load_evaluations(
        self,
        model: ModelConfig,
    ) -> tuple[
        dict[str, CausalEvaluation],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        token = _hugging_face_token(self.config.hf_token_env)
        evaluations: dict[str, CausalEvaluation] = {}
        summaries: list[dict[str, Any]] = []
        provenance: dict[str, dict[str, Any]] = {}
        for dataset in self.config.datasets:
            config_name = dataset.configs[model.name]
            pairs = tuple(
                load_hf_directed_pairs(
                    dataset.repo_id,
                    config_name=config_name,
                    split=dataset.split,
                    revision=dataset.revision,
                    token=token,
                    max_pairs=dataset.max_directed_cases,
                )
            )
            pairing_models = {
                pair.clean.metadata.get("pairing_model")
                for pair in pairs
                if pair.clean.metadata.get("pairing_model") is not None
            }
            if pairing_models and pairing_models != {model.name}:
                raise RuntimeError(
                    f"Dataset {dataset.name!r} pairing-model mismatch: {pairing_models}"
                )
            resolved_revisions = {
                pair.clean.metadata.get("resolved_dataset_revision") for pair in pairs
            }
            if len(resolved_revisions) != 1:
                raise RuntimeError(
                    f"Dataset {dataset.name!r} resolved to multiple revisions: "
                    f"{resolved_revisions}"
                )
            examples = _examples_from_pairs(pairs)
            evaluations[dataset.name] = CausalEvaluation(
                name=dataset.name,
                pairs=pairs,
                answers=SST_ANSWERS,
                examples=examples,
            )
            resolved_revision = next(iter(resolved_revisions))
            summaries.append(
                {
                    "model": model.name,
                    "dataset": dataset.name,
                    "role": "frozen_ood_evaluation",
                    "repo_id": dataset.repo_id,
                    "config_name": config_name,
                    "split": dataset.split,
                    "requested_revision": dataset.revision,
                    "resolved_revision": resolved_revision,
                    "n_examples": len(examples),
                    "n_directed_cases": len(pairs),
                }
            )
            provenance[dataset.name] = {
                "repo_id": dataset.repo_id,
                "config_name": config_name,
                "split": dataset.split,
                "requested_revision": dataset.revision,
                "resolved_revision": resolved_revision,
                "n_directed_cases": len(pairs),
            }
        return evaluations, summaries, provenance


def run_frozen_sentiment_direction_evaluation(
    config: FrozenDirectionEvaluationConfig,
) -> Path:
    """Evaluate selected directions on frozen datasets without fitting or selection."""

    return FrozenSentimentDirectionEvaluation(config).run()


__all__ = [
    "FrozenDirectionEvaluationConfig",
    "FrozenDirectionSelection",
    "FrozenEvaluationDataset",
    "FrozenSentimentDirectionEvaluation",
    "load_frozen_direction_selections",
    "run_frozen_sentiment_direction_evaluation",
]

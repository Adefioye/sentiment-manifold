"""Train AIT valence directions and select layers on a disjoint AIT split."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...activations import extract_mean_pooled_activations
from ...evaluation import DirectionalPatchingEvaluator
from ...models import CausalLMAdapter, ModelConfig, clear_device_cache, resolve_device
from ...persistence import RunArtifactStore
from ..sentiment_position.results import select_layers_by_validation_metric
from .config import AITValenceExperimentConfig
from .datasets import AITDatasetLoader, PreparedAITData
from .fitting import AITDirectionFitRequest, AITDirectionFitService, FittedAITDirection

AdapterFactory = Callable[..., CausalLMAdapter]


def _metric_row(
    *,
    model: str,
    method: str,
    representation: str,
    layer: int,
    result,
) -> dict[str, Any]:
    return {
        "model": model,
        "method": method,
        "fit_position": representation,
        "representation": representation,
        "layer": layer,
        "phase": "layer_selection",
        "selected_layer": False,
        "dataset": "ait_test",
        "dataset_role": "layer_selection",
        "patch_position": "all",
        "n_directed_cases": result.n_pairs,
        "logit_difference_percent": result.recovery_percent,
        "logit_flip_percent": result.flip_percent,
        "sign_flip_percent": result.sign_flip_percent,
        "corrupted_margin": result.corrupted_margin,
        "clean_margin": result.clean_margin,
        "patched_margin": result.patched_margin,
        "corrupted_accuracy": result.corrupted_accuracy,
        "clean_accuracy": result.clean_accuracy,
        "patched_accuracy": result.patched_accuracy,
    }


class AITValenceDirectionExperiment:
    """Fit linear and DAS valence directions with explicit AIT data roles."""

    def __init__(
        self,
        config: AITValenceExperimentConfig,
        *,
        adapter_factory: AdapterFactory = CausalLMAdapter.from_pretrained,
        dataset_loader: AITDatasetLoader | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.adapter_factory = adapter_factory
        self.dataset_loader = dataset_loader or AITDatasetLoader(config)

    def run(self) -> Path:
        output_root = Path(self.config.sweep.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        store = RunArtifactStore(output_root)
        data = self.dataset_loader.load()
        store.write_json("requested_config.json", self.config.to_dict())
        store.write_rows("sample_manifest.csv", data.sample_manifest)
        store.write_rows("pair_manifest.csv", data.pair_manifest)
        store.write_rows("dataset_summary.csv", self._dataset_summary(data))

        combined: dict[str, list[dict[str, Any]]] = {
            "metrics": [],
            "patching_records": [],
            "direction_metadata": [],
            "das_epoch_metrics": [],
            "layer_selection": [],
            "selected_metrics": [],
        }
        model_runs: list[dict[str, Any]] = []
        for model in self.config.models:
            model_tables, runtime = self._run_model(model, data, output_root / model.name)
            for name, rows in model_tables.items():
                combined[name].extend(rows)
            model_runs.append({"model": model.name, "runtime": runtime})
        for name, rows in combined.items():
            if rows:
                store.write_rows(f"all_models_{name}.csv", rows)
        store.write_json(
            "experiment_manifest.json",
            {
                "experiment": "ait-valence-directions",
                "status": "completed",
                "dataset_repo_id": self.config.data.repo_id,
                "requested_dataset_revision": data.requested_revision,
                "resolved_dataset_revision": data.resolved_revision,
                "models": model_runs,
                "methods": list(self.config.sweep.methods),
                "train_examples": len(data.train_examples),
                "train_directed_cases": len(data.train_pairs),
                "eval_directed_cases": len(data.eval_pairs),
                "test_directed_cases": len(data.test_pairs),
                "das_checkpoint_role": "eval",
                "layer_selection_role": "test",
                "test_is_layer_selection_not_final_evaluation": True,
            },
        )
        return output_root

    def _run_model(
        self,
        model: ModelConfig,
        data: PreparedAITData,
        output_dir: Path,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        device_spec = resolve_device(model.device, model.dtype)
        adapter = self.adapter_factory(
            model.hub_name,
            device_spec,
            revision=model.revision,
            prepend_bos=model.prepend_bos,
        )
        runtime = dict(adapter.provenance())
        runtime.update(
            {
                "requested_dataset_revision": data.requested_revision,
                "resolved_dataset_revision": data.resolved_revision,
            }
        )
        layers = self.config.layers_for(adapter.n_layers)
        labels = np.asarray([example.label for example in data.train_examples])
        fit_service = AITDirectionFitService(
            config=self.config,
            adapter=adapter,
            model=model,
            runtime=runtime,
        )
        tables: dict[str, list[dict[str, Any]]] = {
            "metrics": [],
            "patching_records": [],
            "direction_metadata": [],
            "das_epoch_metrics": [],
            "layer_selection": [],
            "selected_metrics": [],
        }
        fitted: dict[tuple[int, str], FittedAITDirection] = {}
        store = RunArtifactStore(output_dir)
        for layer in tqdm(layers, desc=f"{model.name} AIT valence boundaries"):
            activations = extract_mean_pooled_activations(
                adapter,
                data.train_examples,
                layer,
                batch_size=model.batch_size,
                include_special_tokens=self.config.sweep.include_special_tokens_in_mean_pool,
            )
            evaluator = DirectionalPatchingEvaluator(
                adapter,
                list(data.test_pairs),
                layer=layer,
                answers=self.config.data.answers,
                position="all",
                batch_size=model.batch_size,
            )
            for method in self.config.sweep.methods:
                trained = fit_service.fit(
                    AITDirectionFitRequest(
                        examples=data.train_examples,
                        train_pairs=data.train_pairs,
                        eval_pairs=data.eval_pairs,
                        activations=activations,
                        labels=labels,
                        method=method,
                        layer=layer,
                        answers=self.config.data.answers,
                    )
                )
                fitted[(layer, method)] = trained
                representation = str(trained.artifact.metadata["representation"])
                result = evaluator.evaluate(trained.artifact.vector)
                tables["metrics"].append(
                    _metric_row(
                        model=model.name,
                        method=method,
                        representation=representation,
                        layer=layer,
                        result=result,
                    )
                )
                for record, pair in zip(result.records, data.test_pairs):
                    tables["patching_records"].append(
                        {
                            "model": model.name,
                            "method": method,
                            "fit_position": representation,
                            "representation": representation,
                            "layer": layer,
                            "phase": "layer_selection",
                            "dataset": "ait_test",
                            "dataset_role": "layer_selection",
                            "case_id": pair.clean.metadata.get("case_id"),
                            **record,
                        }
                    )
                metadata = trained.artifact.metadata
                tables["direction_metadata"].append(
                    {
                        "model": model.name,
                        "method": method,
                        "fit_position": representation,
                        "representation": representation,
                        "layer": layer,
                        "selected_layer": False,
                        "unit_norm": float(np.linalg.norm(trained.artifact.vector)),
                        "selected_epoch": metadata.get("selected_epoch"),
                        "best_checkpoint_metric": metadata.get("best_checkpoint_metric"),
                        "checkpoint_selection_metric": metadata.get(
                            "checkpoint_selection_metric"
                        ),
                        "artifact_path": str(trained.checkpoint_path),
                        "dataset_repo_id": metadata.get("dataset_repo_id"),
                        "dataset_config": metadata.get("dataset_config"),
                        "resolved_dataset_revision": metadata.get(
                            "resolved_dataset_revision"
                        ),
                        "model_revision": metadata.get("model_revision"),
                        "tokenizer_revision": metadata.get("tokenizer_revision"),
                    }
                )
                for epoch_row in metadata.get("loss_history", []):
                    tables["das_epoch_metrics"].append(
                        {
                            "model": model.name,
                            "method": method,
                            "fit_position": representation,
                            "representation": representation,
                            "layer": layer,
                            "validation_dataset": "ait_eval",
                            **epoch_row,
                        }
                    )
            self._flush(store, tables)
            clear_device_cache(device_spec.device)

        selection = select_layers_by_validation_metric(
            pd.DataFrame(tables["metrics"]),
            dataset="ait_test",
            metric=self.config.selection.layer_selection_metric,
        )
        for row in selection.to_dict(orient="records"):
            row["selection_role"] = "layer_selection"
            row["selection_is_final_evaluation"] = False
            tables["layer_selection"].append(row)
            layer = int(row["selected_layer"])
            method = str(row["method"])
            representation = str(row["fit_position"])
            checkpoint = fitted[(layer, method)].checkpoint_path
            row["direction_checkpoint"] = str(checkpoint)
            for metric in tables["metrics"]:
                if metric["method"] == method and metric["layer"] == layer:
                    metric["selected_layer"] = True
                    tables["selected_metrics"].append(dict(metric))
            for metadata in tables["direction_metadata"]:
                if metadata["method"] == method and metadata["layer"] == layer:
                    metadata["selected_layer"] = True
                    metadata["selection_value_percent"] = row["selection_value_percent"]
                    metadata["fit_position"] = representation
        self._flush(store, tables)
        store.write_json(
            "resolved_config.json",
            {
                **self.config.to_dict(),
                "active_model": vars(model),
                "runtime": runtime,
                "requested_dataset_revision": data.requested_revision,
                "resolved_dataset_revision": data.resolved_revision,
            },
        )
        del adapter
        clear_device_cache(device_spec.device)
        return tables, runtime

    @staticmethod
    def _flush(store: RunArtifactStore, tables: dict[str, list[dict[str, Any]]]) -> None:
        for name, rows in tables.items():
            if rows:
                store.write_rows(f"{name}.csv", rows)

    def _dataset_summary(self, data: PreparedAITData) -> list[dict[str, Any]]:
        return [
            {
                "dataset": "ait",
                "role": "train",
                "source_split": self.config.data.train_split,
                "n_examples": len(data.train_examples),
                "n_directed_cases": len(data.train_pairs),
            },
            {
                "dataset": "ait",
                "role": "das_checkpoint_validation",
                "source_split": self.config.data.eval_split,
                "n_examples": len(
                    {
                        example.example_id
                        for pair in data.eval_pairs
                        for example in (pair.clean, pair.corrupted)
                    }
                ),
                "n_directed_cases": len(data.eval_pairs),
            },
            {
                "dataset": "ait",
                "role": "layer_selection",
                "source_split": self.config.data.test_split,
                "n_examples": len(
                    {
                        example.example_id
                        for pair in data.test_pairs
                        for example in (pair.clean, pair.corrupted)
                    }
                ),
                "n_directed_cases": len(data.test_pairs),
            },
        ]


def run_ait_valence_direction_experiment(config: AITValenceExperimentConfig) -> Path:
    """Run the configured AIT valence-direction experiment."""

    return AITValenceDirectionExperiment(config).run()


__all__ = [
    "AITValenceDirectionExperiment",
    "run_ait_valence_direction_experiment",
]

"""Orchestrate sentiment-direction comparisons across fitting positions."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ...activations import extract_activations
from ...models import CausalLMAdapter, ModelConfig, clear_device_cache, resolve_device
from ...persistence import RunArtifactStore
from .config import SentimentPositionExperimentConfig, comparison_boundaries
from .datasets import PreparedSentimentData, SentimentDatasetLoader
from .evaluation import DirectionEvaluator
from .fitting import DirectionFitRequest, DirectionFitService, FittedDirection
from .manifests import answer_token_rows, pair_rows, prompt_rows, vocabulary_rows
from .results import (
    ExperimentTables,
    direction_similarity_rows,
    select_layers_by_validation_metric,
)

AdapterFactory = Callable[..., CausalLMAdapter]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


class SentimentPositionExperiment:
    """Run a reproducible causal comparison of direction-fitting positions."""

    def __init__(
        self,
        config: SentimentPositionExperimentConfig,
        *,
        adapter_factory: AdapterFactory = CausalLMAdapter.from_pretrained,
    ) -> None:
        config.validate()
        self.config = config
        self.adapter_factory = adapter_factory

    def run(self) -> Path:
        self._seed_everything()
        root = Path(self.config.sweep.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        completed_models: list[str] = []
        for model in self.config.models:
            self._run_model(model, root)
            completed_models.append(model.name)
        RunArtifactStore(root).write_json(
            "run_manifest.json",
            {
                "models": completed_models,
                "methods": list(self.config.sweep.methods),
                "fit_positions": list(self.config.sweep.fit_positions),
                "toy_evaluations": list(self.config.data.toy_evaluations),
                "selection": asdict(self.config.selection),
                "sst_repo_id": self.config.data.sst_repo_id,
                "sst_revision": self.config.data.sst_revision,
            },
        )
        return root

    def _seed_everything(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _run_model(self, model: ModelConfig, root: Path) -> Path:
        device_spec = resolve_device(model.device, model.dtype)
        adapter = self.adapter_factory(
            model.hub_name,
            device_spec,
            revision=model.revision,
            prepend_bos=model.prepend_bos,
        )
        runtime = adapter.provenance()
        store = RunArtifactStore(root / _slug(model.name))
        data = SentimentDatasetLoader(
            config=self.config,
            model=model,
            adapter=adapter,
        ).load()
        layers = self.config.layers_for(adapter.n_layers)
        snapshots = set(comparison_boundaries(adapter.n_layers))
        self._write_static_artifacts(store, adapter, model, data)
        store.write_json(
            "resolved_config.json",
            self._resolved_config(
                model=model,
                runtime=runtime,
                data=data,
                layers=layers,
                snapshots=snapshots,
                run_dir=store.run_dir,
            ),
        )

        fit_service = DirectionFitService(
            config=self.config,
            adapter=adapter,
            model=model,
            runtime=runtime,
        )
        evaluator = DirectionEvaluator(
            adapter=adapter,
            model_name=model.name,
            batch_size=model.batch_size,
            evaluations=data.evaluations,
        )
        tables = ExperimentTables()
        fitted_directions: dict[tuple[int, str, str], FittedDirection] = {}
        labels = np.asarray([row.label for row in data.train_examples])
        for layer in tqdm(layers, desc=f"{model.name} sentiment-position boundaries"):
            activations = self._extract_training_activations(adapter, model, data, layer)
            directions, fitted_at_layer = self._run_layer(
                adapter=adapter,
                model=model,
                data=data,
                fit_service=fit_service,
                evaluator=evaluator,
                labels=labels,
                activations=activations,
                layer=layer,
                tables=tables,
            )
            fitted_directions.update(
                {
                    (layer, fit_position, method): fitted
                    for (fit_position, method), fitted in fitted_at_layer.items()
                }
            )
            if layer in snapshots:
                tables.direction_similarities.extend(
                    direction_similarity_rows(
                        model=model.name,
                        layer=layer,
                        directions=directions,
                    )
                )
            self._flush_tables(store, tables)
        selection = select_layers_by_validation_metric(
            pd.DataFrame(tables.metrics),
            dataset=self.config.selection.layer_dataset,
            metric=self.config.selection.layer_metric,
        )
        self._record_layer_selection(selection, tables, fitted_directions)
        self._evaluate_selected_directions(
            evaluator=evaluator,
            selection=selection,
            fitted_directions=fitted_directions,
            tables=tables,
        )
        self._finalize_tables(store, tables, selection)
        del adapter
        clear_device_cache(device_spec.device)
        return store.run_dir

    @staticmethod
    def _flush_tables(store: RunArtifactStore, tables: ExperimentTables) -> None:
        store.write_rows("metrics.csv", tables.metrics)
        store.write_rows("patching_records.csv", tables.patching_records)
        store.write_rows("direction_metadata.csv", tables.direction_metadata)
        if tables.das_epoch_metrics:
            store.write_rows("das_epoch_metrics.csv", tables.das_epoch_metrics)
        if tables.direction_similarities:
            store.write_rows("direction_similarities.csv", tables.direction_similarities)

    def _finalize_tables(
        self,
        store: RunArtifactStore,
        tables: ExperimentTables,
        selection: pd.DataFrame,
    ) -> None:
        self._flush_tables(store, tables)
        store.write_rows("layer_selection.csv", selection.to_dict(orient="records"))
        selected_metrics = [
            row
            for row in tables.metrics
            if row.get("selected_layer")
            and row.get("phase") != "layer_selection"
            and row.get("dataset") in self.config.selection.final_evaluations
        ]
        store.write_rows("selected_metrics.csv", selected_metrics)

    def _extract_training_activations(
        self,
        adapter: CausalLMAdapter,
        model: ModelConfig,
        data: PreparedSentimentData,
        layer: int,
    ) -> dict[str, np.ndarray]:
        return {
            position: extract_activations(
                adapter,
                data.train_examples,
                layer,
                position=position,
                batch_size=model.batch_size,
            )
            for position in self.config.sweep.fit_positions
        }

    def _run_layer(
        self,
        *,
        adapter: CausalLMAdapter,
        model: ModelConfig,
        data: PreparedSentimentData,
        fit_service: DirectionFitService,
        evaluator: DirectionEvaluator,
        labels: np.ndarray,
        activations: Mapping[str, np.ndarray],
        layer: int,
        tables: ExperimentTables,
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], FittedDirection]]:
        directions: dict[tuple[str, str], np.ndarray] = {}
        fitted_at_layer: dict[tuple[str, str], FittedDirection] = {}
        validation = data.evaluations[self.config.selection.das_checkpoint_dataset]
        for fit_position in self.config.sweep.fit_positions:
            for method in self.config.sweep.methods:
                fitted = fit_service.fit(
                    DirectionFitRequest(
                        examples=data.train_examples,
                        pairs=data.train_pairs,
                        activations=activations[fit_position],
                        labels=labels,
                        method=method,
                        fit_position=fit_position,
                        layer=layer,
                        answers=data.filtered_toy.answers,
                        validation_dataset=validation.name,
                        validation_pairs=validation.pairs,
                        validation_answers=validation.answers,
                    )
                )
                directions[(fit_position, method)] = fitted.artifact.vector
                fitted_at_layer[(fit_position, method)] = fitted
                self._record_direction(tables, model, fitted, fit_position, method, layer)
                evaluation = evaluator.evaluate(
                    fitted=fitted,
                    fit_position=fit_position,
                    method=method,
                    layer=layer,
                    evaluation_names=(self.config.selection.layer_dataset,),
                    phase="layer_selection",
                )
                tables.metrics.extend(evaluation.metrics)
                tables.patching_records.extend(evaluation.patching_records)
                clear_device_cache(adapter.device_spec.device)
        return directions, fitted_at_layer

    def _record_layer_selection(
        self,
        selection: pd.DataFrame,
        tables: ExperimentTables,
        fitted_directions: Mapping[tuple[int, str, str], FittedDirection],
    ) -> None:
        for index, selected in selection.iterrows():
            layer = int(selected["selected_layer"])
            fit_position = str(selected["fit_position"])
            method = str(selected["method"])
            fitted = fitted_directions[(layer, fit_position, method)]
            selection.loc[index, "selected_epoch"] = fitted.artifact.metadata.get(
                "selected_epoch"
            )
            selection.loc[index, "direction_checkpoint"] = str(fitted.checkpoint_path)
            for row in tables.metrics:
                if (
                    row["method"] == method
                    and row["fit_position"] == fit_position
                    and row["layer"] == layer
                    and row["dataset"] == self.config.selection.layer_dataset
                ):
                    row["selected_layer"] = True
            for row in tables.patching_records:
                if (
                    row["method"] == method
                    and row["fit_position"] == fit_position
                    and row["layer"] == layer
                    and row["dataset"] == self.config.selection.layer_dataset
                ):
                    row["selected_layer"] = True
            for row in tables.direction_metadata:
                if (
                    row["method"] == method
                    and row["fit_position"] == fit_position
                    and row["layer"] == layer
                ):
                    row["selected_layer"] = True

    def _evaluate_selected_directions(
        self,
        *,
        evaluator: DirectionEvaluator,
        selection: pd.DataFrame,
        fitted_directions: Mapping[tuple[int, str, str], FittedDirection],
        tables: ExperimentTables,
    ) -> None:
        for selected in selection.itertuples(index=False):
            layer = int(selected.selected_layer)
            fit_position = str(selected.fit_position)
            method = str(selected.method)
            fitted = fitted_directions[(layer, fit_position, method)]
            for dataset in self.config.selection.final_evaluations:
                phase = "final_evaluation" if dataset == "sst" else "selected_layer_evaluation"
                evaluation = evaluator.evaluate(
                    fitted=fitted,
                    fit_position=fit_position,
                    method=method,
                    layer=layer,
                    evaluation_names=(dataset,),
                    phase=phase,
                    selected_layer=True,
                )
                tables.metrics.extend(evaluation.metrics)
                tables.patching_records.extend(evaluation.patching_records)

    def _record_direction(
        self,
        tables: ExperimentTables,
        model: ModelConfig,
        fitted: FittedDirection,
        fit_position: str,
        method: str,
        layer: int,
    ) -> None:
        artifact = fitted.artifact
        tables.direction_metadata.append(
            {
                "model": model.name,
                "method": method,
                "fit_position": fit_position,
                "layer": layer,
                "boundary_semantics": "post_block_pre_next_block_or_final_norm",
                "unit_norm": float(np.linalg.norm(artifact.vector)),
                "orientation_convention": artifact.metadata.get("orientation_convention"),
                "orientation_sign_flipped": artifact.metadata.get("orientation_sign_flipped"),
                "raw_orientation_dot": artifact.metadata.get("raw_orientation_dot"),
                "train_accuracy": artifact.metadata.get("train_accuracy"),
                "selected_epoch": artifact.metadata.get("selected_epoch"),
                "checkpoint_selection_metric": artifact.metadata.get(
                    "checkpoint_selection_metric"
                ),
                "best_checkpoint_metric": artifact.metadata.get("best_checkpoint_metric"),
                "selected_layer": False,
                "artifact_path": str(fitted.checkpoint_path),
                "artifact_relative_path": str(
                    fitted.checkpoint_path.relative_to(self.config.sweep.checkpoint_dir)
                ),
            }
        )
        tables.das_epoch_metrics.extend(
            {
                "model": model.name,
                "method": method,
                "fit_position": fit_position,
                "layer": layer,
                "seed": self.config.seed,
                "validation_dataset": self.config.selection.das_checkpoint_dataset,
                **loss,
            }
            for loss in artifact.metadata.get("loss_history", [])
        )

    def _write_static_artifacts(
        self,
        store: RunArtifactStore,
        adapter: CausalLMAdapter,
        model: ModelConfig,
        data: PreparedSentimentData,
    ) -> None:
        store.write_rows(
            "answer_tokens.csv",
            answer_token_rows(adapter, data.answers_by_dataset),
        )
        prompts = prompt_rows(adapter, "toy_train", "train", data.train_examples)
        for evaluation in data.evaluations.values():
            prompts.extend(
                prompt_rows(
                    adapter,
                    evaluation.name,
                    self.config.data.sst_split if evaluation.name == "sst" else "test",
                    evaluation.examples,
                )
            )
        store.write_rows("prompt_manifest.csv", prompts)
        store.write_rows(
            "toy_vocabulary.csv",
            vocabulary_rows(
                adapter,
                data.raw_toy,
                data.filtered_toy,
                data.toy_evaluations,
            ),
        )
        pairs = pair_rows(adapter, "toy_train", data.train_pairs)
        for evaluation in data.evaluations.values():
            pairs.extend(pair_rows(adapter, evaluation.name, evaluation.pairs))
        if not all(row["equal_token_length"] for row in pairs):
            raise RuntimeError("At least one all-token patching pair has unequal token lengths")
        store.write_rows("pair_manifest.csv", pairs)
        store.write_rows(
            "dataset_summary.csv",
            self._dataset_summary_rows(model, data),
        )

    def _dataset_summary_rows(
        self,
        model: ModelConfig,
        data: PreparedSentimentData,
    ) -> list[dict[str, Any]]:
        return [
            {
                "model": model.name,
                "dataset": "toy_train",
                "role": "direction_fitting",
                "n_examples": len(data.train_examples),
                "n_directed_cases": len(data.train_pairs),
            },
            *[
                {
                    "model": model.name,
                    "dataset": evaluation.name,
                    "role": self._dataset_role(evaluation.name),
                    "n_examples": len(evaluation.examples),
                    "n_directed_cases": len(evaluation.pairs),
                }
                for evaluation in data.evaluations.values()
            ],
        ]

    def _dataset_role(self, dataset: str) -> str:
        roles: list[str] = []
        if dataset == self.config.selection.das_checkpoint_dataset:
            roles.append("checkpoint_validation")
        if dataset == self.config.selection.layer_dataset:
            roles.append("layer_selection")
        if dataset in self.config.selection.final_evaluations:
            roles.append("final_ood_evaluation" if dataset == "sst" else "selected_layer_evaluation")
        return "+".join(roles) or "configured_evaluation"

    def _resolved_config(
        self,
        *,
        model: ModelConfig,
        runtime: Mapping[str, Any],
        data: PreparedSentimentData,
        layers: list[int],
        snapshots: set[int],
        run_dir: Path,
    ) -> dict[str, Any]:
        resolved = self.config.to_dict()
        resolved.update(
            runtime=dict(runtime),
            active_model=asdict(model),
            resolved_layers=layers,
            comparison_boundaries=sorted(snapshots),
            resolved_toy_evaluations=list(data.toy_evaluations),
            sst_directed_pair_config=data.sst_config,
            sst_resolved_revision=data.sst_resolved_revision,
            results_run_dir=str(run_dir),
        )
        return resolved


def run_sentiment_position_comparison(
    config: SentimentPositionExperimentConfig,
) -> Path:
    """Convenience API for running a complete fitting-position comparison."""

    return SentimentPositionExperiment(config).run()


__all__ = ["SentimentPositionExperiment", "run_sentiment_position_comparison"]

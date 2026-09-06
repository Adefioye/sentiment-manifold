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
from .config import (
    REQUIRED_TOY_EVALUATIONS,
    SentimentPositionExperimentConfig,
    comparison_boundaries,
)
from .datasets import PreparedSentimentData, SentimentDatasetLoader
from .evaluation import DirectionEvaluator
from .fitting import DirectionFitRequest, DirectionFitService, FittedDirection
from .manifests import answer_token_rows, pair_rows, prompt_rows, vocabulary_rows
from .results import ExperimentTables, direction_similarity_rows, select_best_layers

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
                "required_toy_evaluations": list(REQUIRED_TOY_EVALUATIONS),
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
        labels = np.asarray([row.label for row in data.train_examples])
        for layer in tqdm(layers, desc=f"{model.name} sentiment-position boundaries"):
            activations = self._extract_training_activations(adapter, model, data, layer)
            directions = self._run_layer(
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
            if layer in snapshots:
                tables.direction_similarities.extend(
                    direction_similarity_rows(
                        model=model.name,
                        layer=layer,
                        directions=directions,
                    )
                )
            self._flush_tables(store, tables)
        self._finalize_tables(store, tables)
        del adapter
        clear_device_cache(device_spec.device)
        return store.run_dir

    @staticmethod
    def _flush_tables(store: RunArtifactStore, tables: ExperimentTables) -> None:
        store.write_rows("metrics.csv", tables.metrics)
        store.write_rows("patching_records.csv", tables.patching_records)
        store.write_rows("direction_metadata.csv", tables.direction_metadata)
        if tables.das_losses:
            store.write_rows("das_losses.csv", tables.das_losses)
        if tables.direction_similarities:
            store.write_rows("direction_similarities.csv", tables.direction_similarities)

    @classmethod
    def _finalize_tables(cls, store: RunArtifactStore, tables: ExperimentTables) -> None:
        cls._flush_tables(store, tables)
        best = select_best_layers(pd.DataFrame(tables.metrics))
        store.write_rows("best_layers.csv", best.to_dict(orient="records"))

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
                position="focus" if position == "adjective" else "final",
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
    ) -> dict[tuple[str, str], np.ndarray]:
        directions: dict[tuple[str, str], np.ndarray] = {}
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
                    )
                )
                directions[(fit_position, method)] = fitted.artifact.vector
                self._record_direction(tables, model, fitted, fit_position, method, layer)
                evaluation = evaluator.evaluate(
                    fitted=fitted,
                    fit_position=fit_position,
                    method=method,
                    layer=layer,
                )
                tables.metrics.extend(evaluation.metrics)
                tables.patching_records.extend(evaluation.patching_records)
                clear_device_cache(adapter.device_spec.device)
        return directions

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
                "artifact_path": str(fitted.checkpoint_path),
                "artifact_relative_path": str(
                    fitted.checkpoint_path.relative_to(self.config.sweep.checkpoint_dir)
                ),
            }
        )
        tables.das_losses.extend(
            {
                "model": model.name,
                "method": method,
                "fit_position": fit_position,
                "layer": layer,
                "seed": self.config.seed,
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

    @staticmethod
    def _dataset_summary_rows(
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
                    "role": "causal_evaluation",
                    "n_examples": len(evaluation.examples),
                    "n_directed_cases": len(evaluation.pairs),
                }
                for evaluation in data.evaluations.values()
            ],
        ]

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
            required_toy_evaluations=list(data.toy_evaluations),
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

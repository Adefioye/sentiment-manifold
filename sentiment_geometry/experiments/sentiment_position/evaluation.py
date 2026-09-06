"""Evaluate fitted directions on a collection of causal datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...evaluation import evaluate_directional_patching
from ...models import CausalLMAdapter
from .datasets import CausalEvaluation
from .fitting import FittedDirection


@dataclass(frozen=True)
class DirectionEvaluationResult:
    metrics: tuple[dict[str, Any], ...]
    patching_records: tuple[dict[str, Any], ...]


class DirectionEvaluator:
    """Apply one fitted direction to every configured evaluation dataset."""

    def __init__(
        self,
        *,
        adapter: CausalLMAdapter,
        model_name: str,
        batch_size: int,
        evaluations: Mapping[str, CausalEvaluation],
    ) -> None:
        self.adapter = adapter
        self.model_name = model_name
        self.batch_size = batch_size
        self.evaluations = evaluations

    def evaluate(
        self,
        fitted: FittedDirection,
        *,
        fit_position: str,
        method: str,
        layer: int,
    ) -> DirectionEvaluationResult:
        metric_rows: list[dict[str, Any]] = []
        patching_rows: list[dict[str, Any]] = []
        identity = {
            "model": self.model_name,
            "method": method,
            "fit_position": fit_position,
            "layer": layer,
        }
        for evaluation in self.evaluations.values():
            result = evaluate_directional_patching(
                self.adapter,
                list(evaluation.pairs),
                fitted.artifact.vector,
                layer=layer,
                answers=evaluation.answers,
                position="all",
                batch_size=self.batch_size,
            )
            metric_rows.append(
                {
                    **identity,
                    "dataset": evaluation.name,
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
            )
            patching_rows.extend(
                {
                    **identity,
                    "dataset": evaluation.name,
                    "patch_position": "all",
                    **record,
                }
                for record in result.records
            )
        return DirectionEvaluationResult(tuple(metric_rows), tuple(patching_rows))


__all__ = ["DirectionEvaluationResult", "DirectionEvaluator"]

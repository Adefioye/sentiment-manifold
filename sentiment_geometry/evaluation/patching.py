"""Directional activation patching with Tigges-style recovery and flip metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..datasets.types import CounterfactualPair
from ..interventions import directional_replace
from ..models import CausalLMAdapter, TokenizedBatch


@dataclass(frozen=True)
class PatchingResult:
    recovery: float
    flip_rate: float
    sign_flip_rate: float
    corrupted_margin: float
    clean_margin: float
    patched_margin: float
    corrupted_accuracy: float
    clean_accuracy: float
    patched_accuracy: float
    n_pairs: int
    records: tuple[dict, ...]

    @property
    def recovery_percent(self) -> float:
        """Tigges-style logit-difference recovery on a percentage scale."""
        return 100.0 * self.recovery

    @property
    def flip_percent(self) -> float:
        """Tigges code's baseline-calibrated logit-flip score in percent."""
        return 100.0 * self.flip_rate

    @property
    def sign_flip_percent(self) -> float:
        """Literal percentage of corrupted logits flipped toward the clean target."""
        return 100.0 * self.sign_flip_rate


def _logit_differences(logits: Tensor, answer_ids: dict[int, int | Tensor]) -> Tensor:
    """Return the paper's positive-minus-negative next-token logit difference."""
    positive_ids = torch.as_tensor(answer_ids[1], device=logits.device, dtype=torch.long).reshape(
        -1
    )
    negative_ids = torch.as_tensor(answer_ids[0], device=logits.device, dtype=torch.long).reshape(
        -1
    )
    if len(positive_ids) != len(negative_ids):
        raise ValueError("Positive and negative answers must form aligned pairs")
    positive = logits.index_select(-1, positive_ids)
    negative = logits.index_select(-1, negative_ids)
    return (positive - negative).mean(dim=-1)


def _target_signed_margins(logit_differences: Tensor, target_labels: Tensor) -> Tensor:
    """Orient each logit difference toward its counterfactual target.

    Tigges et al. implement this orientation by ordering ``answer_tokens`` as
    ``[correct, incorrect]`` for every prompt.  Multiplying the fixed
    positive-minus-negative difference by the target polarity is equivalent and
    prevents positive-to-negative and negative-to-positive cases from cancelling
    in the aggregate recovery metric.
    """
    return torch.where(target_labels == 1, logit_differences, -logit_differences)


def _centered_target_signed_margins(logit_differences: Tensor, target_labels: Tensor) -> Tensor:
    """Match Tigges's binary-classifier bias centering before flip accuracy."""
    centered = logit_differences - logit_differences.mean()
    return _target_signed_margins(centered, target_labels)


def _target_directed_logit_flips(
    corrupted_logit_differences: Tensor,
    patched_logit_differences: Tensor,
    target_labels: Tensor,
) -> Tensor:
    """Identify strict pre/post sign inversions that end at the target label."""
    sign_changed = corrupted_logit_differences * patched_logit_differences < 0
    patched_target_margins = _target_signed_margins(patched_logit_differences, target_labels)
    return sign_changed & (patched_target_margins > 0)


def _directional_editor(
    adapter: CausalLMAdapter,
    *,
    clean: TokenizedBatch,
    corrupted: TokenizedBatch,
    clean_boundary: Tensor,
    vector: Tensor,
    position: str,
) -> Callable[[Tensor], Tensor]:
    """Create an editor whose batch state is fixed for one forward pass."""

    def editor(hidden: Tensor) -> Tensor:
        edited = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        if position == "focus":
            if clean.focus_positions is None or corrupted.focus_positions is None:
                raise ValueError("Focus patching requires focus spans")
            edited[rows, corrupted.focus_positions] = directional_replace(
                hidden[rows, corrupted.focus_positions],
                clean_boundary[rows, clean.focus_positions],
                vector,
            )
        elif position == "final":
            corrupted_pos = adapter.last_positions(corrupted.attention_mask)
            clean_pos = adapter.last_positions(clean.attention_mask)
            edited[rows, corrupted_pos] = directional_replace(
                hidden[rows, corrupted_pos], clean_boundary[rows, clean_pos], vector
            )
        elif position == "all":
            replacements = directional_replace(hidden, clean_boundary, vector)
            mask = corrupted.attention_mask.bool().unsqueeze(-1)
            edited = torch.where(mask, replacements, hidden)
        else:
            raise ValueError("position must be focus, final, or all")
        return edited

    return editor


@dataclass(frozen=True)
class _PreparedPatchingBatch:
    pairs: tuple[CounterfactualPair, ...]
    clean: TokenizedBatch
    corrupted: TokenizedBatch
    clean_boundary: Tensor
    target_labels: Tensor
    clean_logit_differences: Tensor
    corrupted_logit_differences: Tensor


class DirectionalPatchingEvaluator:
    """Cache direction-independent baselines for repeated patching evaluations."""

    def __init__(
        self,
        adapter: CausalLMAdapter,
        pairs: list[CounterfactualPair],
        *,
        layer: int,
        answers: dict[int, tuple[str, ...] | list[str]],
        position: str,
        batch_size: int = 16,
    ) -> None:
        if not pairs:
            raise ValueError("Directional patching requires non-empty pairs")
        self.adapter = adapter
        self.layer = layer
        self.position = position
        self.answer_ids = {
            label: torch.tensor(
                [adapter.single_token_id(answer) for answer in values],
                device=adapter.device_spec.device,
            )
            for label, values in answers.items()
        }
        if len(self.answer_ids[0]) != len(self.answer_ids[1]) or not len(self.answer_ids[0]):
            raise ValueError("Positive and negative answers must form non-empty aligned pairs")
        self.batches = self._prepare(pairs, batch_size)
        self.n_pairs = len(pairs)

    def _prepare(
        self, pairs: list[CounterfactualPair], batch_size: int
    ) -> tuple[_PreparedPatchingBatch, ...]:
        prepared: list[_PreparedPatchingBatch] = []
        device = self.adapter.device_spec.device
        for start in range(0, len(pairs), batch_size):
            selected = tuple(pairs[start : start + batch_size])
            clean = self.adapter.tokenize([pair.clean for pair in selected]).to(device)
            corrupted = self.adapter.tokenize([pair.corrupted for pair in selected]).to(device)
            if self.position == "all" and not torch.equal(
                clean.attention_mask.sum(1), corrupted.attention_mask.sum(1)
            ):
                raise ValueError("Patching pairs must have equal token lengths")
            targets = torch.tensor([pair.clean.label for pair in selected], device=device)
            rows = torch.arange(len(selected), device=device)
            with torch.inference_mode():
                clean_boundary = self.adapter.boundary_activations(clean, self.layer).detach()
                clean_output = self.adapter.model(
                    input_ids=clean.input_ids,
                    attention_mask=clean.attention_mask,
                    use_cache=False,
                )
                corrupted_output = self.adapter.model(
                    input_ids=corrupted.input_ids,
                    attention_mask=corrupted.attention_mask,
                    use_cache=False,
                )
            clean_positions = self.adapter.last_positions(clean.attention_mask)
            corrupted_positions = self.adapter.last_positions(corrupted.attention_mask)
            clean_logits = clean_output.logits[rows, clean_positions].float()
            corrupted_logits = corrupted_output.logits[rows, corrupted_positions].float()
            prepared.append(
                _PreparedPatchingBatch(
                    pairs=selected,
                    clean=clean,
                    corrupted=corrupted,
                    clean_boundary=clean_boundary,
                    target_labels=targets,
                    clean_logit_differences=_logit_differences(
                        clean_logits, self.answer_ids
                    ).detach(),
                    corrupted_logit_differences=_logit_differences(
                        corrupted_logits, self.answer_ids
                    ).detach(),
                )
            )
        return tuple(prepared)

    def evaluate(self, direction: np.ndarray) -> PatchingResult:
        vector = torch.as_tensor(
            direction, device=self.adapter.device_spec.device, dtype=torch.float32
        )
        corrupted_margins: list[Tensor] = []
        clean_margins: list[Tensor] = []
        patched_margins: list[Tensor] = []
        corrupted_logit_differences: list[Tensor] = []
        clean_logit_differences: list[Tensor] = []
        patched_logit_differences: list[Tensor] = []
        all_target_labels: list[Tensor] = []
        flips: list[Tensor] = []
        records: list[dict] = []

        for batch in self.batches:
            editor = _directional_editor(
                self.adapter,
                clean=batch.clean,
                corrupted=batch.corrupted,
                clean_boundary=batch.clean_boundary,
                vector=vector,
                position=self.position,
            )
            with torch.inference_mode(), self.adapter.edit_boundary(self.layer, editor):
                patched_output = self.adapter.model(
                    input_ids=batch.corrupted.input_ids,
                    attention_mask=batch.corrupted.attention_mask,
                    use_cache=False,
                )
            rows = torch.arange(
                len(batch.pairs), device=self.adapter.device_spec.device
            )
            positions = self.adapter.last_positions(batch.corrupted.attention_mask)
            patched_logits = patched_output.logits[rows, positions].float()
            patched_differences = _logit_differences(patched_logits, self.answer_ids)
            clean_differences = batch.clean_logit_differences
            corrupted_differences = batch.corrupted_logit_differences
            targets = batch.target_labels

            clean_logit_differences.append(clean_differences.cpu())
            corrupted_logit_differences.append(corrupted_differences.cpu())
            patched_logit_differences.append(patched_differences.cpu())
            all_target_labels.append(targets.cpu())
            clean_batch_margins = _target_signed_margins(clean_differences, targets).cpu()
            corrupted_batch_margins = _target_signed_margins(
                corrupted_differences, targets
            ).cpu()
            patched_batch_margins = _target_signed_margins(patched_differences, targets).cpu()
            clean_margins.append(clean_batch_margins)
            corrupted_margins.append(corrupted_batch_margins)
            patched_margins.append(patched_batch_margins)
            batch_flips = (
                _target_directed_logit_flips(
                    corrupted_differences,
                    patched_differences,
                    targets,
                )
                .float()
                .cpu()
            )
            flips.append(batch_flips)
            for index, pair in enumerate(batch.pairs):
                corrupted_value = float(corrupted_batch_margins[index])
                clean_value = float(clean_batch_margins[index])
                patched_value = float(patched_batch_margins[index])
                denominator = clean_value - corrupted_value
                records.append(
                    {
                        "clean_id": pair.clean.example_id,
                        "corrupted_id": pair.corrupted.example_id,
                        "clean_label": pair.clean.label,
                        "corrupted_label": pair.corrupted.label,
                        "clean_logit_diff": float(clean_differences[index]),
                        "corrupted_logit_diff": float(corrupted_differences[index]),
                        "patched_logit_diff": float(patched_differences[index]),
                        "clean_margin": clean_value,
                        "corrupted_margin": corrupted_value,
                        "patched_margin": patched_value,
                        "recovery": float("nan")
                        if abs(denominator) < 1e-8
                        else (patched_value - corrupted_value) / denominator,
                        "recovery_percent": float("nan")
                        if abs(denominator) < 1e-8
                        else 100.0 * (patched_value - corrupted_value) / denominator,
                        "flipped": float(batch_flips[index]),
                    }
                )

        return _summarize_patching(
            clean_margins=clean_margins,
            corrupted_margins=corrupted_margins,
            patched_margins=patched_margins,
            clean_logit_differences=clean_logit_differences,
            corrupted_logit_differences=corrupted_logit_differences,
            patched_logit_differences=patched_logit_differences,
            target_labels=all_target_labels,
            flips=flips,
            records=records,
            n_pairs=self.n_pairs,
        )


def _summarize_patching(
    *,
    clean_margins: list[Tensor],
    corrupted_margins: list[Tensor],
    patched_margins: list[Tensor],
    clean_logit_differences: list[Tensor],
    corrupted_logit_differences: list[Tensor],
    patched_logit_differences: list[Tensor],
    target_labels: list[Tensor],
    flips: list[Tensor],
    records: list[dict],
    n_pairs: int,
) -> PatchingResult:
    corrupted_margin = torch.cat(corrupted_margins).mean().item()
    clean_margin = torch.cat(clean_margins).mean().item()
    patched_margin = torch.cat(patched_margins).mean().item()
    denominator = clean_margin - corrupted_margin
    recovery = (
        float("nan")
        if abs(denominator) < 1e-8
        else (patched_margin - corrupted_margin) / denominator
    )
    targets = torch.cat(target_labels)
    centered_corrupted_margins = _centered_target_signed_margins(
        torch.cat(corrupted_logit_differences), targets
    )
    centered_clean_margins = _centered_target_signed_margins(
        torch.cat(clean_logit_differences), targets
    )
    centered_patched_margins = _centered_target_signed_margins(
        torch.cat(patched_logit_differences), targets
    )
    corrupted_accuracy = (centered_corrupted_margins > 0).float().mean().item()
    clean_accuracy = (centered_clean_margins > 0).float().mean().item()
    patched_accuracy = (centered_patched_margins > 0).float().mean().item()
    accuracy_denominator = clean_accuracy - corrupted_accuracy
    flip_rate = (
        float("nan")
        if abs(accuracy_denominator) < 1e-8
        else (patched_accuracy - corrupted_accuracy) / accuracy_denominator
    )
    return PatchingResult(
        recovery=float(recovery),
        flip_rate=float(flip_rate),
        sign_flip_rate=float(torch.cat(flips).mean()),
        corrupted_margin=float(corrupted_margin),
        clean_margin=float(clean_margin),
        patched_margin=float(patched_margin),
        corrupted_accuracy=float(corrupted_accuracy),
        clean_accuracy=float(clean_accuracy),
        patched_accuracy=float(patched_accuracy),
        n_pairs=n_pairs,
        records=tuple(records),
    )


def evaluate_directional_patching(
    adapter: CausalLMAdapter,
    pairs: list[CounterfactualPair],
    direction: np.ndarray,
    *,
    layer: int,
    answers: dict[int, tuple[str, ...] | list[str]],
    position: str,
    batch_size: int = 16,
) -> PatchingResult:
    evaluator = DirectionalPatchingEvaluator(
        adapter,
        pairs,
        layer=layer,
        answers=answers,
        position=position,
        batch_size=batch_size,
    )
    return evaluator.evaluate(direction)

"""Directional activation patching with Tigges-style recovery and flip metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..directions.das import directional_replace
from ..models import CausalLMAdapter
from ..types import CounterfactualPair


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


def _logit_differences(logits: Tensor, answer_ids: dict[int, int]) -> Tensor:
    """Return the paper's positive-minus-negative next-token logit difference."""
    positive = logits[:, answer_ids[1]]
    negative = logits[:, answer_ids[0]]
    return positive - negative


def _target_signed_margins(logit_differences: Tensor, target_labels: Tensor) -> Tensor:
    """Orient each logit difference toward its counterfactual target.

    Tigges et al. implement this orientation by ordering ``answer_tokens`` as
    ``[correct, incorrect]`` for every prompt.  Multiplying the fixed
    positive-minus-negative difference by the target polarity is equivalent and
    prevents positive-to-negative and negative-to-positive cases from cancelling
    in the aggregate recovery metric.
    """
    return torch.where(target_labels == 1, logit_differences, -logit_differences)


def _centered_target_signed_margins(
    logit_differences: Tensor, target_labels: Tensor
) -> Tensor:
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


def evaluate_directional_patching(
    adapter: CausalLMAdapter,
    pairs: list[CounterfactualPair],
    direction: np.ndarray,
    *,
    layer: int,
    answers: dict[int, str],
    position: str,
    batch_size: int = 16,
) -> PatchingResult:
    if not pairs:
        raise ValueError("Directional patching requires non-empty pairs")
    vector = torch.as_tensor(direction, device=adapter.device_spec.device, dtype=torch.float32)
    answer_ids = {label: adapter.single_token_id(answer) for label, answer in answers.items()}
    corrupted_margins: list[Tensor] = []
    clean_margins: list[Tensor] = []
    patched_margins: list[Tensor] = []
    corrupted_logit_differences: list[Tensor] = []
    clean_logit_differences: list[Tensor] = []
    patched_logit_differences: list[Tensor] = []
    all_target_labels: list[Tensor] = []
    flips: list[Tensor] = []
    records: list[dict] = []

    for start in range(0, len(pairs), batch_size):
        selected = pairs[start : start + batch_size]
        clean = adapter.tokenize([pair.clean for pair in selected]).to(adapter.device_spec.device)
        corrupted = adapter.tokenize([pair.corrupted for pair in selected]).to(
            adapter.device_spec.device
        )
        if position == "all" and not torch.equal(
            clean.attention_mask.sum(1), corrupted.attention_mask.sum(1)
        ):
            raise ValueError("Patching pairs must have equal token lengths")
        target_labels = torch.tensor(
            [pair.clean.label for pair in selected], device=adapter.device_spec.device
        )
        with torch.inference_mode():
            clean_boundary = adapter.boundary_activations(clean, layer)
            clean_output = adapter.model(
                input_ids=clean.input_ids, attention_mask=clean.attention_mask, use_cache=False
            )
            corrupted_output = adapter.model(
                input_ids=corrupted.input_ids,
                attention_mask=corrupted.attention_mask,
                use_cache=False,
            )

            def editor(hidden: Tensor) -> Tensor:
                edited = hidden.clone()
                rows = torch.arange(len(selected), device=hidden.device)
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

            with adapter.edit_boundary(layer, editor):
                patched_output = adapter.model(
                    input_ids=corrupted.input_ids,
                    attention_mask=corrupted.attention_mask,
                    use_cache=False,
                )

        rows = torch.arange(len(selected), device=adapter.device_spec.device)
        corrupted_pos = adapter.last_positions(corrupted.attention_mask)
        clean_pos = adapter.last_positions(clean.attention_mask)
        corrupted_logits = corrupted_output.logits[rows, corrupted_pos].float()
        clean_logits = clean_output.logits[rows, clean_pos].float()
        patched_logits = patched_output.logits[rows, corrupted_pos].float()
        batch_corrupted_logit_differences = _logit_differences(corrupted_logits, answer_ids)
        batch_clean_logit_differences = _logit_differences(clean_logits, answer_ids)
        batch_patched_logit_differences = _logit_differences(patched_logits, answer_ids)
        corrupted_logit_differences.append(batch_corrupted_logit_differences.cpu())
        clean_logit_differences.append(batch_clean_logit_differences.cpu())
        patched_logit_differences.append(batch_patched_logit_differences.cpu())
        all_target_labels.append(target_labels.cpu())
        batch_corrupted_margins = _target_signed_margins(
            batch_corrupted_logit_differences, target_labels
        ).cpu()
        batch_clean_margins = _target_signed_margins(
            batch_clean_logit_differences, target_labels
        ).cpu()
        batch_patched_margins = _target_signed_margins(
            batch_patched_logit_differences, target_labels
        ).cpu()
        corrupted_margins.append(batch_corrupted_margins)
        clean_margins.append(batch_clean_margins)
        patched_margins.append(batch_patched_margins)
        batch_flips = _target_directed_logit_flips(
            batch_corrupted_logit_differences,
            batch_patched_logit_differences,
            target_labels,
        ).float().cpu()
        flips.append(batch_flips)
        for index, pair in enumerate(selected):
            corrupted_value = float(batch_corrupted_margins[index])
            clean_value = float(batch_clean_margins[index])
            patched_value = float(batch_patched_margins[index])
            denominator = clean_value - corrupted_value
            records.append(
                {
                    "clean_id": pair.clean.example_id,
                    "corrupted_id": pair.corrupted.example_id,
                    "clean_label": pair.clean.label,
                    "corrupted_label": pair.corrupted.label,
                    "clean_logit_diff": float(batch_clean_logit_differences[index]),
                    "corrupted_logit_diff": float(batch_corrupted_logit_differences[index]),
                    "patched_logit_diff": float(batch_patched_logit_differences[index]),
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

    corrupted_margin = torch.cat(corrupted_margins).mean().item()
    clean_margin = torch.cat(clean_margins).mean().item()
    patched_margin = torch.cat(patched_margins).mean().item()
    denominator = clean_margin - corrupted_margin
    recovery = (
        float("nan")
        if abs(denominator) < 1e-8
        else (patched_margin - corrupted_margin) / denominator
    )
    targets = torch.cat(all_target_labels)
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
        n_pairs=len(pairs),
        records=tuple(records),
    )

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
    base_margin: float
    source_margin: float
    patched_margin: float
    n_pairs: int
    records: tuple[dict, ...]


def _margins(logits: Tensor, target_labels: Tensor, answer_ids: dict[int, int]) -> Tensor:
    positive = logits[:, answer_ids[1]]
    negative = logits[:, answer_ids[0]]
    signed = positive - negative
    return torch.where(target_labels == 1, signed, -signed)


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
    base_margins: list[Tensor] = []
    source_margins: list[Tensor] = []
    patched_margins: list[Tensor] = []
    flips: list[Tensor] = []
    records: list[dict] = []

    for start in range(0, len(pairs), batch_size):
        selected = pairs[start : start + batch_size]
        base = adapter.tokenize([pair.base for pair in selected]).to(adapter.device_spec.device)
        source = adapter.tokenize([pair.source for pair in selected]).to(adapter.device_spec.device)
        if position == "all" and not torch.equal(
            base.attention_mask.sum(1), source.attention_mask.sum(1)
        ):
            raise ValueError("Patching pairs must have equal token lengths")
        target_labels = torch.tensor(
            [pair.source.label for pair in selected], device=adapter.device_spec.device
        )
        with torch.inference_mode():
            source_boundary = adapter.boundary_activations(source, layer)
            base_output = adapter.model(
                input_ids=base.input_ids, attention_mask=base.attention_mask, use_cache=False
            )
            source_output = adapter.model(
                input_ids=source.input_ids, attention_mask=source.attention_mask, use_cache=False
            )

            def editor(hidden: Tensor) -> Tensor:
                edited = hidden.clone()
                rows = torch.arange(len(selected), device=hidden.device)
                if position == "focus":
                    if base.focus_positions is None or source.focus_positions is None:
                        raise ValueError("Focus patching requires focus spans")
                    edited[rows, base.focus_positions] = directional_replace(
                        hidden[rows, base.focus_positions],
                        source_boundary[rows, source.focus_positions],
                        vector,
                    )
                elif position == "final":
                    base_pos = adapter.last_positions(base.attention_mask)
                    source_pos = adapter.last_positions(source.attention_mask)
                    edited[rows, base_pos] = directional_replace(
                        hidden[rows, base_pos], source_boundary[rows, source_pos], vector
                    )
                elif position == "all":
                    replacements = directional_replace(hidden, source_boundary, vector)
                    mask = base.attention_mask.bool().unsqueeze(-1)
                    edited = torch.where(mask, replacements, hidden)
                else:
                    raise ValueError("position must be focus, final, or all")
                return edited

            with adapter.edit_boundary(layer, editor):
                patched_output = adapter.model(
                    input_ids=base.input_ids, attention_mask=base.attention_mask, use_cache=False
                )

        rows = torch.arange(len(selected), device=adapter.device_spec.device)
        base_pos = adapter.last_positions(base.attention_mask)
        source_pos = adapter.last_positions(source.attention_mask)
        base_logits = base_output.logits[rows, base_pos].float()
        source_logits = source_output.logits[rows, source_pos].float()
        patched_logits = patched_output.logits[rows, base_pos].float()
        batch_base_margins = _margins(base_logits, target_labels, answer_ids).cpu()
        batch_source_margins = _margins(source_logits, target_labels, answer_ids).cpu()
        batch_patched_margins = _margins(patched_logits, target_labels, answer_ids).cpu()
        base_margins.append(batch_base_margins)
        source_margins.append(batch_source_margins)
        patched_margins.append(batch_patched_margins)
        pair_logits = torch.stack(
            (patched_logits[:, answer_ids[0]], patched_logits[:, answer_ids[1]]), dim=1
        )
        batch_flips = (pair_logits.argmax(dim=1) == target_labels).float().cpu()
        flips.append(batch_flips)
        for index, pair in enumerate(selected):
            base_value = float(batch_base_margins[index])
            source_value = float(batch_source_margins[index])
            patched_value = float(batch_patched_margins[index])
            denominator = source_value - base_value
            records.append(
                {
                    "base_id": pair.base.example_id,
                    "source_id": pair.source.example_id,
                    "base_label": pair.base.label,
                    "source_label": pair.source.label,
                    "base_margin": base_value,
                    "source_margin": source_value,
                    "patched_margin": patched_value,
                    "recovery": float("nan")
                    if abs(denominator) < 1e-8
                    else (patched_value - base_value) / denominator,
                    "flipped": float(batch_flips[index]),
                }
            )

    base_margin = torch.cat(base_margins).mean().item()
    source_margin = torch.cat(source_margins).mean().item()
    patched_margin = torch.cat(patched_margins).mean().item()
    denominator = source_margin - base_margin
    recovery = (
        float("nan") if abs(denominator) < 1e-8 else (patched_margin - base_margin) / denominator
    )
    return PatchingResult(
        recovery=float(recovery),
        flip_rate=float(torch.cat(flips).mean()),
        base_margin=float(base_margin),
        source_margin=float(source_margin),
        patched_margin=float(patched_margin),
        n_pairs=len(pairs),
        records=tuple(records),
    )

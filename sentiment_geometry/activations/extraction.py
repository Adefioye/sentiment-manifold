"""Model-independent batching for residual-stream activation extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from ..datasets.types import TextExample
from ..models import CausalLMAdapter

ActivationPosition = Literal["focus", "adjective", "verb", "summary", "final"]


def extract_activations(
    adapter: CausalLMAdapter,
    examples: Sequence[TextExample],
    layer: int,
    *,
    position: ActivationPosition = "focus",
    batch_size: int = 16,
) -> np.ndarray:
    """Extract one residual-boundary activation per text example."""

    if not examples:
        raise ValueError("Activation extraction requires at least one example")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    chunks: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = adapter.tokenize(selected).to(adapter.device_spec.device)
        with torch.inference_mode():
            hidden = adapter.boundary_activations(batch, layer)
        positions = adapter.activation_positions(batch, position)
        rows = hidden[torch.arange(len(selected), device=hidden.device), positions]
        chunks.append(rows.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def extract_mean_pooled_activations(
    adapter: CausalLMAdapter,
    examples: Sequence[TextExample],
    layer: int,
    *,
    batch_size: int = 16,
    include_special_tokens: bool = False,
) -> np.ndarray:
    """Mean-pool one residual-boundary representation over each prompt's real tokens."""

    if not examples:
        raise ValueError("Activation extraction requires at least one example")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    chunks: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = adapter.tokenize(selected).to(adapter.device_spec.device)
        with torch.inference_mode():
            hidden = adapter.boundary_activations(batch, layer)
        mask = batch.attention_mask.bool()
        if not include_special_tokens:
            if batch.special_tokens_mask is None:
                raise RuntimeError("Tokenizer did not return a special-token mask")
            mask &= ~batch.special_tokens_mask.bool()
        counts = mask.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("Mean pooling found a prompt with no eligible tokens")
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        chunks.append(pooled.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def extract_last_token_activations(
    adapter: CausalLMAdapter,
    examples: Sequence[TextExample],
    layer: int,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract each prompt's final non-padding residual-stream activation."""

    return extract_activations(
        adapter,
        examples,
        layer,
        position="final",
        batch_size=batch_size,
    )

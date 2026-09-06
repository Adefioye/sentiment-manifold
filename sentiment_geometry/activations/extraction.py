"""Model-independent batching for residual-stream activation extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from ..datasets.types import TextExample
from ..models import CausalLMAdapter

ActivationPosition = Literal["focus", "final"]


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

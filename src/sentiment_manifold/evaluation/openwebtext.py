"""OpenWebText language-model loss under directional resample ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..directions.das import directional_replace
from ..models import CausalLMAdapter


@dataclass(frozen=True)
class OpenWebTextResult:
    baseline_loss: float
    ablated_loss: float
    loss_delta: float
    n_sequences: int


def _token_batches(
    adapter: CausalLMAdapter, texts: list[str], sequence_length: int, batch_size: int
):
    content_length = sequence_length - 1 if adapter.prepend_bos else sequence_length
    encoded = adapter.tokenizer(
        texts,
        truncation=True,
        max_length=content_length,
        padding="max_length",
        add_special_tokens=not adapter.prepend_bos,
        return_tensors="pt",
    )
    if adapter.prepend_bos:
        bos = torch.full(
            (encoded["input_ids"].shape[0], 1),
            int(adapter.tokenizer.bos_token_id),
            dtype=encoded["input_ids"].dtype,
        )
        encoded["input_ids"] = torch.cat((bos, encoded["input_ids"]), dim=1)
        encoded["attention_mask"] = torch.cat(
            (torch.ones_like(bos), encoded["attention_mask"]), dim=1
        )
    for start in range(0, len(texts), batch_size):
        yield {
            key: value[start : start + batch_size].to(adapter.device_spec.device)
            for key, value in encoded.items()
        }


def evaluate_openwebtext_ablation(
    adapter: CausalLMAdapter,
    texts: list[str],
    direction: np.ndarray,
    *,
    layer: int,
    sequence_length: int = 128,
    batch_size: int = 8,
    seed: int = 0,
) -> OpenWebTextResult:
    vector = torch.as_tensor(direction, device=adapter.device_spec.device, dtype=torch.float32)
    torch.manual_seed(seed)
    baseline_total = 0.0
    ablated_total = 0.0
    sequences = 0
    for batch in _token_batches(adapter, texts, sequence_length, batch_size):
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        with torch.inference_mode():
            baseline = adapter.model(**batch, labels=labels, use_cache=False).loss

            def editor(hidden: Tensor) -> Tensor:
                permutation = torch.randperm(hidden.shape[0], device=hidden.device)
                shuffled = hidden[permutation]
                replaced = directional_replace(hidden, shuffled, vector)
                return torch.where(batch["attention_mask"].bool().unsqueeze(-1), replaced, hidden)

            with adapter.edit_boundary(layer, editor):
                ablated = adapter.model(**batch, labels=labels, use_cache=False).loss
        current = len(batch["input_ids"])
        baseline_total += float(baseline) * current
        ablated_total += float(ablated) * current
        sequences += current
    baseline_loss = baseline_total / sequences
    ablated_loss = ablated_total / sequences
    return OpenWebTextResult(
        baseline_loss=baseline_loss,
        ablated_loss=ablated_loss,
        loss_delta=ablated_loss - baseline_loss,
        n_sequences=sequences,
    )

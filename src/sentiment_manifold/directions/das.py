"""One-dimensional Distributed Alignment Search through causal patching."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..models import CausalLMAdapter
from ..types import CounterfactualPair
from .base import FitResult


@dataclass(frozen=True)
class DASTrainingConfig:
    epochs: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    max_grad_norm: float = 1.0
    seed: int = 0


def directional_replace(corrupted: Tensor, clean: Tensor, direction: Tensor) -> Tensor:
    """Replace the corrupted activation's scalar projection with the clean one."""
    unit = F.normalize(direction.float(), dim=0).to(corrupted.dtype)
    delta = ((clean - corrupted) * unit).sum(dim=-1, keepdim=True)
    return corrupted + delta * unit


class DASFitter:
    name = "das"

    def __init__(self, config: DASTrainingConfig) -> None:
        self.config = config

    def fit(
        self,
        adapter: CausalLMAdapter,
        pairs: list[CounterfactualPair],
        *,
        layer: int,
        answers: dict[int, str],
    ) -> FitResult:
        if not pairs:
            raise ValueError("DAS requires at least one counterfactual pair")
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        raw_direction = torch.nn.Parameter(
            torch.randn(adapter.hidden_size, device=adapter.device_spec.device, dtype=torch.float32)
        )
        optimizer = torch.optim.Adam(
            [raw_direction],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        answer_ids = {label: adapter.single_token_id(answer) for label, answer in answers.items()}
        indices = list(range(len(pairs)))
        epoch_losses: list[float] = []
        adapter.model.eval()
        for _epoch in range(self.config.epochs):
            random.shuffle(indices)
            running_loss = 0.0
            batches = 0
            for start in range(0, len(indices), self.config.batch_size):
                selected = [
                    pairs[index] for index in indices[start : start + self.config.batch_size]
                ]
                corrupted_examples = [pair.corrupted for pair in selected]
                clean_examples = [pair.clean for pair in selected]
                corrupted = adapter.tokenize(corrupted_examples).to(adapter.device_spec.device)
                clean = adapter.tokenize(clean_examples).to(adapter.device_spec.device)
                if corrupted.focus_positions is None or clean.focus_positions is None:
                    raise ValueError("DAS training requires focus spans")
                with torch.no_grad():
                    clean_activations = adapter.boundary_activations(clean, layer)
                    clean_rows = clean_activations[
                        torch.arange(len(selected), device=clean_activations.device),
                        clean.focus_positions,
                    ].detach()

                def editor(hidden: Tensor) -> Tensor:
                    edited = hidden.clone()
                    row = torch.arange(len(selected), device=hidden.device)
                    positions = corrupted.focus_positions
                    edited[row, positions] = directional_replace(
                        hidden[row, positions], clean_rows.to(hidden.dtype), raw_direction
                    )
                    return edited

                optimizer.zero_grad(set_to_none=True)
                with adapter.edit_boundary(layer, editor):
                    outputs = adapter.model(
                        input_ids=corrupted.input_ids,
                        attention_mask=corrupted.attention_mask,
                        use_cache=False,
                    )
                final_positions = adapter.last_positions(corrupted.attention_mask)
                logits = outputs.logits[
                    torch.arange(len(selected), device=outputs.logits.device), final_positions
                ].float()
                targets = torch.tensor(
                    [answer_ids[pair.clean.label] for pair in selected],
                    device=logits.device,
                    dtype=torch.long,
                )
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([raw_direction], self.config.max_grad_norm)
                optimizer.step()
                running_loss += float(loss.detach())
                batches += 1
            epoch_losses.append(running_loss / max(1, batches))

        direction = F.normalize(raw_direction.detach(), dim=0).cpu().numpy()
        examples = [pair.corrupted for pair in pairs]
        activations = adapter.extract_activations(
            examples, layer, position="focus", batch_size=self.config.batch_size
        )
        labels = np.asarray([example.label for example in examples])
        mean_diff = activations[labels == 1].mean(0) - activations[labels == 0].mean(0)
        if float(direction @ mean_diff) < 0:
            direction = -direction
        return FitResult(
            self.name,
            direction,
            {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]},
        )

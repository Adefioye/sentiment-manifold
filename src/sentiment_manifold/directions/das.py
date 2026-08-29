"""Tigges-compatible Distributed Alignment Search through directional patching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from ..models import CausalLMAdapter, TokenizedBatch
from ..types import CounterfactualPair
from .base import FitResult


AnswerSpec = dict[int, tuple[str, ...] | list[str]]


@dataclass(frozen=True)
class DASTrainingConfig:
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    max_grad_norm: float = 1.0
    seed: int = 0


class _RotateLayer(nn.Module):
    """Match the upstream ``x @ weight`` orthogonal rotation parameterization."""

    def __init__(self, hidden_size: int, device: torch.device) -> None:
        super().__init__()
        weight = torch.empty(hidden_size, hidden_size, device=device, dtype=torch.float32)
        nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, values: Tensor) -> Tensor:
        return values @ self.weight


def _basis(direction: Tensor) -> Tensor:
    basis = direction.float()
    if basis.ndim == 1:
        basis = basis.unsqueeze(1)
    if basis.ndim != 2:
        raise ValueError("Direction must have shape [d_model] or [d_model, d_subspace]")
    norms = basis.norm(dim=0, keepdim=True)
    if (norms <= 1e-12).any():
        raise ValueError("Direction/subspace contains a zero basis vector")
    return basis / norms


def directional_replace(corrupted: Tensor, clean: Tensor, direction: Tensor) -> Tensor:
    """Replace the corrupted projection onto a direction or orthonormal subspace."""
    basis = _basis(direction).to(corrupted.dtype)
    coefficients = torch.einsum("...d,dk->...k", clean - corrupted, basis)
    replacement = torch.einsum("...k,dk->...d", coefficients, basis)
    return corrupted + replacement


def _answer_ids(adapter: CausalLMAdapter, answers: AnswerSpec) -> dict[int, Tensor]:
    if set(answers) != {0, 1}:
        raise ValueError("Answer specifications must contain labels 0 and 1")
    if len(answers[0]) != len(answers[1]) or not answers[0]:
        raise ValueError("Positive and negative answer lists must be non-empty and paired")
    return {
        label: torch.tensor(
            [adapter.single_token_id(answer) for answer in values],
            device=adapter.device_spec.device,
            dtype=torch.long,
        )
        for label, values in answers.items()
    }


def fixed_positive_minus_negative(logits: Tensor, answer_ids: dict[int, Tensor]) -> Tensor:
    """Average the Tigges positive/negative answer-pair logit differences."""
    positive = logits.index_select(-1, answer_ids[1])
    negative = logits.index_select(-1, answer_ids[0])
    return (positive - negative).mean(dim=-1)


def target_signed_margins(
    logits: Tensor, target_labels: Tensor, answer_ids: dict[int, Tensor]
) -> Tensor:
    differences = fixed_positive_minus_negative(logits, answer_ids)
    return torch.where(target_labels == 1, differences, -differences)


@dataclass
class _PreparedBatch:
    corrupted: TokenizedBatch
    clean_rows: Tensor
    corrupted_rows: Tensor
    target_labels: Tensor


class DASFitter:
    """Learn the first ``dimension`` columns of an orthogonal rotation, as upstream."""

    def __init__(
        self,
        config: DASTrainingConfig,
        *,
        dimension: int = 1,
        name: str | None = None,
    ) -> None:
        if dimension not in {1, 2, 3}:
            raise ValueError("Tigges parity supports DAS dimensions 1, 2, or 3")
        self.config = config
        self.dimension = dimension
        self.name = name or ("das" if dimension == 1 else f"das{dimension}d")
        self._layer = -1

    def _prepare(
        self,
        adapter: CausalLMAdapter,
        pairs: list[CounterfactualPair],
        layer: int,
        answer_ids: dict[int, Tensor],
    ) -> tuple[list[_PreparedBatch], float, float, np.ndarray, np.ndarray]:
        prepared: list[_PreparedBatch] = []
        clean_margins: list[Tensor] = []
        corrupted_margins: list[Tensor] = []
        activation_rows: list[Tensor] = []
        activation_labels: list[Tensor] = []
        device = adapter.device_spec.device
        for start in range(0, len(pairs), self.config.batch_size):
            selected = pairs[start : start + self.config.batch_size]
            corrupted = adapter.tokenize([pair.corrupted for pair in selected]).to(device)
            clean = adapter.tokenize([pair.clean for pair in selected]).to(device)
            if corrupted.focus_positions is None or clean.focus_positions is None:
                raise ValueError("DAS training requires focus spans")
            if not torch.equal(clean.attention_mask.sum(1), corrupted.attention_mask.sum(1)):
                raise ValueError("DAS clean/corrupted prompts must have equal token lengths")
            labels = torch.tensor([pair.clean.label for pair in selected], device=device)
            rows = torch.arange(len(selected), device=device)
            with torch.no_grad():
                clean_hidden = adapter.boundary_activations(clean, layer)
                corrupted_hidden = adapter.boundary_activations(corrupted, layer)
                clean_rows = clean_hidden[rows, clean.focus_positions].detach()
                corrupted_rows = corrupted_hidden[rows, corrupted.focus_positions].detach()
                clean_output = adapter.model(
                    input_ids=clean.input_ids,
                    attention_mask=clean.attention_mask,
                    use_cache=False,
                )
                corrupted_output = adapter.model(
                    input_ids=corrupted.input_ids,
                    attention_mask=corrupted.attention_mask,
                    use_cache=False,
                )
                clean_positions = adapter.last_positions(clean.attention_mask)
                corrupted_positions = adapter.last_positions(corrupted.attention_mask)
                clean_logits = clean_output.logits[rows, clean_positions].float()
                corrupted_logits = corrupted_output.logits[rows, corrupted_positions].float()
                clean_margins.append(target_signed_margins(clean_logits, labels, answer_ids).cpu())
                corrupted_margins.append(
                    target_signed_margins(corrupted_logits, labels, answer_ids).cpu()
                )
            prepared.append(
                _PreparedBatch(
                    corrupted=corrupted,
                    clean_rows=clean_rows,
                    corrupted_rows=corrupted_rows,
                    target_labels=labels,
                )
            )
            activation_rows.append(corrupted_rows.float().cpu())
            activation_labels.append(
                torch.tensor([pair.corrupted.label for pair in selected], dtype=torch.long)
            )
        return (
            prepared,
            float(torch.cat(clean_margins).mean()),
            float(torch.cat(corrupted_margins).mean()),
            torch.cat(activation_rows).numpy(),
            torch.cat(activation_labels).numpy(),
        )

    def _batch_loss(
        self,
        adapter: CausalLMAdapter,
        batch: _PreparedBatch,
        basis: Tensor,
        answer_ids: dict[int, Tensor],
        clean_baseline: float,
        corrupted_baseline: float,
    ) -> Tensor:
        corrupted = batch.corrupted

        def editor(hidden: Tensor) -> Tensor:
            edited = hidden.clone()
            rows = torch.arange(len(batch.target_labels), device=hidden.device)
            assert corrupted.focus_positions is not None
            edited[rows, corrupted.focus_positions] = directional_replace(
                batch.corrupted_rows.to(hidden.dtype),
                batch.clean_rows.to(hidden.dtype),
                basis,
            )
            return edited

        with adapter.edit_boundary(layer=self._layer, editor=editor):
            output = adapter.model(
                input_ids=corrupted.input_ids,
                attention_mask=corrupted.attention_mask,
                use_cache=False,
            )
        rows = torch.arange(len(batch.target_labels), device=output.logits.device)
        positions = adapter.last_positions(corrupted.attention_mask)
        logits = output.logits[rows, positions].float()
        patched = target_signed_margins(logits, batch.target_labels, answer_ids).mean()
        denominator = corrupted_baseline - clean_baseline
        if abs(denominator) < 1e-8:
            raise RuntimeError("Clean and corrupted DAS baselines are indistinguishable")
        # Mirror upstream's inverted denoising arguments: clean patch -> 0, corrupt -> 1.
        return (patched - clean_baseline) / denominator

    def fit(
        self,
        adapter: CausalLMAdapter,
        pairs: list[CounterfactualPair],
        *,
        layer: int,
        answers: AnswerSpec,
    ) -> FitResult:
        if not pairs:
            raise ValueError("DAS requires at least one counterfactual pair")
        self._layer = layer
        torch.manual_seed(self.config.seed)
        ids = _answer_ids(adapter, answers)
        prepared, clean_baseline, corrupted_baseline, activations, labels = self._prepare(
            adapter, pairs, layer, ids
        )
        rotation = _RotateLayer(adapter.hidden_size, adapter.device_spec.device)
        rotation = torch.nn.utils.parametrizations.orthogonal(
            rotation, "weight", use_trivialization=False
        )
        optimizer = torch.optim.Adam(
            rotation.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        history: list[dict[str, float | int]] = []
        best_loss = float("inf")
        best_basis: Tensor | None = None
        adapter.model.eval()
        for epoch in range(self.config.epochs):
            rotation.train()
            train_total = 0.0
            for batch in prepared:
                optimizer.zero_grad(set_to_none=True)
                basis = rotation.weight[:, : self.dimension]
                loss = self._batch_loss(
                    adapter, batch, basis, ids, clean_baseline, corrupted_baseline
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rotation.parameters(), self.config.max_grad_norm)
                optimizer.step()
                train_total += float(loss.detach())
            train_loss = train_total / len(prepared)

            rotation.eval()
            eval_total = 0.0
            with torch.no_grad():
                for batch in prepared:
                    eval_total += float(
                        self._batch_loss(
                            adapter,
                            batch,
                            rotation.weight[:, : self.dimension],
                            ids,
                            clean_baseline,
                            corrupted_baseline,
                        )
                    )
            eval_loss = eval_total / len(prepared)
            history.append({"epoch": epoch, "train_loss": train_loss, "evaluation_loss": eval_loss})
            if train_loss < best_loss:
                best_loss = train_loss
                best_basis = rotation.weight[:, : self.dimension].detach().clone()

        assert best_basis is not None
        direction = best_basis.cpu().numpy()
        mean_diff = activations[labels == 1].mean(0) - activations[labels == 0].mean(0)
        orientation_dot = float(direction[:, 0] @ mean_diff)
        if orientation_dot < 0:
            direction[:, 0] *= -1
        output_direction: np.ndarray = direction[:, 0] if self.dimension == 1 else direction
        return FitResult(
            self.name,
            output_direction,
            {
                "implementation": "tigges_rotation",
                "subspace_dimension": self.dimension,
                "epochs": self.config.epochs,
                "clean_baseline_margin": clean_baseline,
                "corrupted_baseline_margin": corrupted_baseline,
                "best_train_loss": best_loss,
                "loss_history": history,
                "orientation_convention": "negative_to_positive_first_basis_vector",
                "orientation_reference": "toy_train_class_mean_difference",
                "raw_orientation_dot": orientation_dot,
                "orientation_sign_flipped": bool(orientation_dot < 0),
            },
        )

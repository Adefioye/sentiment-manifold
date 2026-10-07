"""Modular binary mean-difference, logistic-regression, and MLP-1 probes."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _validate(activations: FloatArray, labels: IntArray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(activations, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not len(y):
        raise ValueError("Expected aligned activations [samples, features] and labels")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes must be present")
    if not np.isfinite(x).all():
        raise ValueError("Activations contain NaN or infinity")
    return x, y


class BinaryProbe(Protocol):
    method: str
    native_threshold: float
    diagnostics: dict[str, Any]

    @property
    def direction(self) -> np.ndarray | None: ...

    def native_scores(self, activations: FloatArray) -> np.ndarray: ...

    def midpoint_scores(self, activations: FloatArray) -> np.ndarray: ...

    def state(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _MeanDifferenceProbe:
    vector: np.ndarray
    training_threshold: float
    method: str = "mean_diff"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def native_threshold(self) -> float:
        return self.training_threshold

    @property
    def direction(self) -> np.ndarray:
        return self.vector

    def native_scores(self, activations: FloatArray) -> np.ndarray:
        return np.asarray(activations) @ self.vector

    def midpoint_scores(self, activations: FloatArray) -> np.ndarray:
        return self.native_scores(activations)

    def state(self) -> dict[str, Any]:
        return {"direction": self.vector, "native_threshold": self.native_threshold}


@dataclass(frozen=True)
class LogisticProbeConfig:
    c: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 5000
    tol: float = 1e-4
    class_weight: str | None = None


@dataclass(frozen=True)
class _LogisticProbe:
    coefficient: np.ndarray
    intercept: float
    config: LogisticProbeConfig
    method: str = "logistic_regression"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def native_threshold(self) -> float:
        return 0.0

    @property
    def direction(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.coefficient))
        return (self.coefficient / norm).astype(np.float32)

    def native_scores(self, activations: FloatArray) -> np.ndarray:
        return np.asarray(activations) @ self.coefficient + self.intercept

    def midpoint_scores(self, activations: FloatArray) -> np.ndarray:
        return np.asarray(activations) @ self.direction

    def state(self) -> dict[str, Any]:
        return {
            "coefficient": self.coefficient,
            "direction": self.direction,
            "intercept": self.intercept,
            "config": asdict(self.config),
        }


def fit_mean_difference_probe(activations: FloatArray, labels: IntArray) -> BinaryProbe:
    x, y = _validate(activations, labels)
    raw = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Mean difference produced a zero or non-finite direction")
    direction = (raw / norm).astype(np.float32)
    scores = x @ direction
    threshold = float(0.5 * (scores[y == 0].mean() + scores[y == 1].mean()))
    return _MeanDifferenceProbe(
        direction,
        threshold,
        diagnostics={"n_parameters": int(x.shape[1]), "direction_norm": 1.0},
    )


def fit_logistic_probe(
    activations: FloatArray,
    labels: IntArray,
    *,
    config: LogisticProbeConfig,
    seed: int,
) -> BinaryProbe:
    x, y = _validate(activations, labels)
    model = LogisticRegression(
        C=config.c,
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        class_weight=config.class_weight,
        random_state=seed,
    ).fit(x, y)
    coefficient = np.asarray(model.coef_[0], dtype=np.float32)
    intercept = float(model.intercept_[0])
    return _LogisticProbe(
        coefficient,
        intercept,
        config,
        diagnostics={
            "n_parameters": int(len(coefficient) + 1),
            "iterations": int(np.max(model.n_iter_)),
            "training_native_accuracy": float(model.score(x, y)),
        },
    )


@dataclass(frozen=True)
class MLP1ProbeConfig:
    hidden_size: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.0
    batch_size: int = 16
    max_epochs: int = 500
    patience: int = 20
    min_delta: float = 1e-4
    standardize_inputs: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("MLP sizes and epoch counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("MLP dropout must be in [0, 1)")


class _MLP1Network(nn.Module):
    def __init__(self, input_size: int, config: MLP1ProbeConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values).squeeze(-1)


@dataclass
class _MLP1Probe:
    network: _MLP1Network
    mean: np.ndarray
    scale: np.ndarray
    config: MLP1ProbeConfig
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    method: str = "mlp1"
    native_threshold: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> None:
        return None

    def _scaled(self, activations: FloatArray) -> Tensor:
        values = np.asarray(activations, dtype=np.float32)
        return torch.from_numpy((values - self.mean) / self.scale)

    def native_scores(self, activations: FloatArray) -> np.ndarray:
        self.network.eval()
        with torch.inference_mode():
            return self.network(self._scaled(activations)).cpu().numpy()

    def midpoint_scores(self, activations: FloatArray) -> np.ndarray:
        return self.native_scores(activations)

    def state(self) -> dict[str, Any]:
        return {
            "state_dict": self.network.state_dict(),
            "mean": self.mean,
            "scale": self.scale,
            "config": asdict(self.config),
            "best_epoch": self.best_epoch,
            "history": self.history,
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_mlp1_probe(
    training_activations: FloatArray,
    training_labels: IntArray,
    validation_activations: FloatArray,
    validation_labels: IntArray,
    *,
    config: MLP1ProbeConfig,
    seed: int,
    select_final_checkpoint: bool = False,
) -> BinaryProbe:
    train_x, train_y = _validate(training_activations, training_labels)
    validation_x, validation_y = _validate(validation_activations, validation_labels)
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("Training and validation activation widths differ")
    _seed_everything(seed)
    mean = train_x.mean(axis=0) if config.standardize_inputs else np.zeros(train_x.shape[1])
    scale = train_x.std(axis=0) if config.standardize_inputs else np.ones(train_x.shape[1])
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    train_tensor = torch.from_numpy((train_x - mean) / scale)
    train_targets = torch.from_numpy(train_y.astype(np.float32))
    validation_tensor = torch.from_numpy((validation_x - mean) / scale)
    validation_targets = torch.from_numpy(validation_y.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor, train_targets),
        batch_size=min(config.batch_size, len(train_tensor)),
        shuffle=True,
        generator=generator,
    )
    network = _MLP1Network(train_x.shape[1], config)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(config.max_epochs):
        network.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
        network.eval()
        with torch.inference_mode():
            validation_loss = float(loss_fn(network(validation_tensor), validation_targets))
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(train_tensor),
                "validation_loss": validation_loss,
            }
        )
        if select_final_checkpoint:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            continue
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("MLP-1 training did not produce a finite validation checkpoint")
    network.load_state_dict(best_state)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    return _MLP1Probe(
        network=network,
        mean=mean,
        scale=scale,
        config=config,
        history=tuple(history),
        best_epoch=best_epoch,
        diagnostics={
            "n_parameters": int(parameter_count),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
    )


__all__ = [
    "BinaryProbe",
    "LogisticProbeConfig",
    "MLP1ProbeConfig",
    "fit_logistic_probe",
    "fit_mean_difference_probe",
    "fit_mlp1_probe",
]

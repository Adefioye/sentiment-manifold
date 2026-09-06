"""Shared data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextExample:
    text: str
    label: int
    example_id: str
    focus_start: int | None = None
    focus_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("Binary sentiment labels must be 0 or 1")


@dataclass(frozen=True)
class CounterfactualPair:
    clean: TextExample
    corrupted: TextExample

    def __post_init__(self) -> None:
        if self.clean.label == self.corrupted.label:
            raise ValueError("A counterfactual pair must cross sentiment labels")

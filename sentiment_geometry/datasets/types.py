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
    named_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("Binary sentiment labels must be 0 or 1")
        for name, (start, end) in self.named_spans.items():
            if start < 0 or end <= start or end > len(self.text):
                raise ValueError(
                    f"Named span {name!r} must satisfy 0 <= start < end <= text length"
                )


@dataclass(frozen=True)
class CounterfactualPair:
    clean: TextExample
    corrupted: TextExample

    def __post_init__(self) -> None:
        if self.clean.label == self.corrupted.label:
            raise ValueError("A counterfactual pair must cross sentiment labels")

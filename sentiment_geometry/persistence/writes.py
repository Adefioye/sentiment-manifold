"""Atomic writes for artifacts stored on local or mounted filesystems."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def staged_path(destination: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace the destination on success."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(destination: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text without exposing a partially written destination."""

    destination = Path(destination)
    with staged_path(destination) as temporary:
        temporary.write_text(text, encoding=encoding)
    return destination


__all__ = ["staged_path", "write_text_atomic"]

"""Generic JSON and tabular artifact persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .writes import staged_path, write_text_atomic


class RunArtifactStore:
    """Write named artifacts beneath one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / filename
        return write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))

    def write_rows(self, filename: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.run_dir / filename
        with staged_path(path) as temporary:
            pd.DataFrame(rows).to_csv(temporary, index=False)
        return path


__all__ = ["RunArtifactStore"]

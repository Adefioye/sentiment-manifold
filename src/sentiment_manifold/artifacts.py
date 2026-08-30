"""Portable direction checkpoints and run metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DirectionArtifact:
    method: str
    model_name: str
    layer: int
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.ndim == 1:
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError("Direction must have finite non-zero norm")
            self.vector = vector / norm
            return
        if vector.ndim != 2 or vector.shape[1] < 1 or not np.isfinite(vector).all():
            raise ValueError("Subspace must have finite shape [d_model, d_subspace]")
        q, r = np.linalg.qr(vector)
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        self.vector = (q[:, : vector.shape[1]] * signs).astype(np.float32)
        self.metadata.setdefault("subspace_dimension", int(vector.shape[1]))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = asdict(self)
        header.pop("vector")
        np.savez_compressed(path, vector=self.vector, metadata=json.dumps(header, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DirectionArtifact":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(vector=data["vector"], **metadata)

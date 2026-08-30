"""Output locations for local, Colab, and Google Drive runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


def resolve_output_dir(
    output_dir: str | Path | None = None,
    *,
    use_google_drive: bool = False,
    drive_subdir: str = "sentiment-manifold/results",
) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    elif os.environ.get("SENTIMENT_MANIFOLD_OUTPUT_DIR"):
        path = Path(os.environ["SENTIMENT_MANIFOLD_OUTPUT_DIR"])
    elif use_google_drive:
        path = Path("/content/drive/MyDrive") / drive_subdir
    else:
        path = Path("outputs/results")
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_dir(
    checkpoint_dir: str | Path | None = None,
    *,
    use_google_drive: bool = False,
    drive_subdir: str = "sentiment-manifold/checkpoints",
) -> Path:
    """Resolve persistent model checkpoints independently from lightweight results."""

    if checkpoint_dir is not None:
        path = Path(checkpoint_dir)
    elif os.environ.get("SENTIMENT_MANIFOLD_CHECKPOINT_DIR"):
        path = Path(os.environ["SENTIMENT_MANIFOLD_CHECKPOINT_DIR"])
    elif use_google_drive:
        path = Path("/content/drive/MyDrive") / drive_subdir
    else:
        path = Path("checkpoints")
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_variant_dir(
    checkpoint_root: str | Path,
    *,
    model_name: str,
    phase: str,
    method: str,
    fingerprint_payload: dict[str, Any],
) -> Path:
    """Return a model/configuration-scoped directory that cannot overwrite other fits."""

    payload = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def slug(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")

    return Path(checkpoint_root) / slug(model_name) / slug(phase) / slug(method) / fingerprint


def maybe_mount_google_drive(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Google Drive mounting is only available inside Google Colab") from exc
    drive.mount("/content/drive")

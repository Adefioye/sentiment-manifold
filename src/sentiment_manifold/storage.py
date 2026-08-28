"""Output locations for local, Colab, and Google Drive runs."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_output_dir(
    output_dir: str | Path | None = None,
    *,
    use_google_drive: bool = False,
    drive_subdir: str = "sentiment-manifold",
) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    elif os.environ.get("SENTIMENT_MANIFOLD_OUTPUT_DIR"):
        path = Path(os.environ["SENTIMENT_MANIFOLD_OUTPUT_DIR"])
    elif use_google_drive:
        path = Path("/content/drive/MyDrive") / drive_subdir
    else:
        path = Path("outputs")
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_mount_google_drive(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Google Drive mounting is only available inside Google Colab") from exc
    drive.mount("/content/drive")

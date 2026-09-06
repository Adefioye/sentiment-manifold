"""Timestamped layouts for durable experiment runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .writes import write_text_atomic

RUN_MANIFEST_SCHEMA_VERSION = 1
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validated_segment(value: str, *, field_name: str) -> str:
    if value in {".", ".."} or not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(
            f"{field_name} must be one filesystem-safe path segment; got {value!r}"
        )
    return value


def _localized_time(now: datetime | None, timezone_name: str) -> datetime:
    try:
        location = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
    if now is None:
        return datetime.now(location)
    if now.tzinfo is None:
        return now.replace(tzinfo=location)
    return now.astimezone(location)


@dataclass(frozen=True)
class TimestampedRunLayout:
    """Durable directories and provenance for one experiment execution."""

    experiment_name: str
    run_id: str
    timezone_name: str
    started_at: str
    started_at_utc: str
    root: Path
    results_dir: Path
    directions_dir: Path
    figures_dir: Path
    resumed: bool = False

    @property
    def manifest_path(self) -> Path:
        return self.root / "run_manifest.json"

    def update_manifest(self, *, status: str, metadata: dict[str, Any] | None = None) -> Path:
        """Update mutable run status without discarding immutable provenance."""

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        if metadata:
            manifest.update(metadata)
        write_text_atomic(
            self.manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        return self.manifest_path


def prepare_timestamped_run(
    storage_root: str | Path,
    *,
    experiment_name: str,
    timezone_name: str = "UTC",
    resume_run_id: str | None = None,
    now: datetime | None = None,
) -> TimestampedRunLayout:
    """Create a new minute-stamped run or explicitly resume an existing one.

    New directories are never silently reused. To continue an interrupted run, callers must
    provide its exact ``resume_run_id``.
    """

    experiment_name = _validated_segment(experiment_name, field_name="experiment_name")
    local_time = _localized_time(now, timezone_name)
    runs_root = Path(storage_root).expanduser().resolve() / experiment_name / "runs"

    resumed = resume_run_id is not None
    if resumed:
        run_id = _validated_segment(resume_run_id, field_name="resume_run_id")
    else:
        zone_abbreviation = local_time.tzname() or "UTC"
        run_id = local_time.strftime(f"%Y-%m-%d_%H-%M_{zone_abbreviation}")
        _validated_segment(run_id, field_name="generated run_id")

    root = runs_root / run_id
    manifest_path = root / "run_manifest.json"
    if resumed:
        if not root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume run without its manifest: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_name") != experiment_name:
            raise ValueError(
                f"Run {run_id!r} belongs to {manifest.get('experiment_name')!r}, "
                f"not {experiment_name!r}"
            )
        timezone_name = str(manifest["timezone"])
        started_at = str(manifest["started_at"])
        started_at_utc = str(manifest["started_at_utc"])
    else:
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Run directory already exists for this minute: {root}. "
                "Set resume_run_id explicitly to continue it."
            ) from exc
        started_at = local_time.isoformat(timespec="minutes")
        started_at_utc = local_time.astimezone(timezone.utc).isoformat(timespec="minutes")

    results_dir = root / "results"
    directions_dir = root / "directions"
    figures_dir = root / "figures"
    for path in (results_dir, directions_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    layout = TimestampedRunLayout(
        experiment_name=experiment_name,
        run_id=run_id,
        timezone_name=timezone_name,
        started_at=started_at,
        started_at_utc=started_at_utc,
        root=root,
        results_dir=results_dir,
        directions_dir=directions_dir,
        figures_dir=figures_dir,
        resumed=resumed,
    )
    if not resumed:
        write_text_atomic(
            manifest_path,
            json.dumps(
                {
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                    "experiment_name": experiment_name,
                    "run_id": run_id,
                    "timezone": timezone_name,
                    "started_at": started_at,
                    "started_at_utc": started_at_utc,
                    "status": "initialized",
                    "paths": {
                        "results": "results",
                        "directions": "directions",
                        "figures": "figures",
                    },
                },
                indent=2,
                sort_keys=True,
            ),
        )
    else:
        layout.update_manifest(status="resumed")
    return layout


__all__ = ["TimestampedRunLayout", "prepare_timestamped_run"]

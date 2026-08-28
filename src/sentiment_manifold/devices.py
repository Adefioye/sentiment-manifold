"""Device and dtype selection shared by scripts and notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    dtype: torch.dtype

    @property
    def autocast_enabled(self) -> bool:
        return self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}


def resolve_device(preferred: str = "auto", dtype: str = "auto") -> DeviceSpec:
    """Resolve CUDA, Apple MPS, or CPU with a safe default dtype."""
    preferred = preferred.lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preferred in {"cuda", "mps", "cpu"}:
        device = torch.device(preferred)
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    else:
        raise ValueError(f"Unknown device {preferred!r}; choose auto, cuda, mps, or cpu")

    if dtype == "auto":
        if device.type == "cuda":
            resolved_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            resolved_dtype = torch.float16
        else:
            resolved_dtype = torch.float32
    else:
        aliases = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            resolved_dtype = aliases[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown dtype {dtype!r}") from exc
    if device.type == "cpu" and resolved_dtype == torch.float16:
        raise ValueError("float16 on CPU is unsupported for this pipeline; use float32")
    return DeviceSpec(device=device, dtype=resolved_dtype)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

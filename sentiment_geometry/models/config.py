"""Model loading configuration shared by experiment workflows."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ALIASES = {
    "gpt2-small": "gpt2",
    "qwen-0.6b": "Qwen/Qwen3-0.6B-Base",
    "gemma-2b": "google/gemma-2b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


@dataclass
class ModelConfig:
    name: str = "gpt2-small"
    revision: str | None = None
    prepend_bos: bool = True
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16

    @property
    def hub_name(self) -> str:
        return MODEL_ALIASES.get(self.name, self.name)


__all__ = ["MODEL_ALIASES", "ModelConfig"]

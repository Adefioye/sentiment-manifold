"""Hugging Face causal-LM adapter with uniform residual-boundary hooks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..devices import DeviceSpec
from ..types import TextExample


@dataclass
class TokenizedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    focus_positions: Tensor | None = None

    def to(self, device: torch.device) -> "TokenizedBatch":
        return TokenizedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            focus_positions=None
            if self.focus_positions is None
            else self.focus_positions.to(device),
        )


class CausalLMAdapter:
    """Expose layer boundaries 0..n_layers for GPT-2 and Qwen-style models."""

    def __init__(self, model, tokenizer, model_name: str, device_spec: DeviceSpec) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device_spec = device_spec
        self.blocks, self.final_norm = self._find_transformer_parts(model)
        self.n_layers = len(self.blocks)
        self.hidden_size = int(model.config.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device_spec: DeviceSpec,
        *,
        revision: str | None = None,
    ) -> "CausalLMAdapter":
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=device_spec.dtype,
        ).to(device_spec.device)
        model.eval()
        model.requires_grad_(False)
        return cls(model, tokenizer, model_name, device_spec)

    @staticmethod
    def _find_transformer_parts(model):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h, model.transformer.ln_f
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers, model.model.norm
        raise TypeError(
            f"Unsupported model architecture {type(model).__name__}; expected GPT-2 or Qwen/Llama layout"
        )

    def tokenize(self, examples: Sequence[TextExample] | Sequence[str]) -> TokenizedBatch:
        texts = [item.text if isinstance(item, TextExample) else item for item in examples]
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        focus_positions: Tensor | None = None
        if examples and isinstance(examples[0], TextExample):
            positions = []
            for example, row in zip(examples, offsets):
                assert isinstance(example, TextExample)
                if example.focus_start is None or example.focus_end is None:
                    positions.append(-1)
                    continue
                overlaps = [
                    index
                    for index, (start, end) in enumerate(row.tolist())
                    if end > example.focus_start and start < example.focus_end
                ]
                if not overlaps:
                    raise ValueError(f"Could not locate focus span in {example.example_id}")
                positions.append(overlaps[-1])
            focus_positions = torch.tensor(positions, dtype=torch.long)
        return TokenizedBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            focus_positions=focus_positions,
        )

    def single_token_id(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"Expected a single-token answer for {text!r}, got {ids}. "
                "Choose tokenizer-compatible labels before running causal metrics."
            )
        return int(ids[0])

    @staticmethod
    def last_positions(attention_mask: Tensor) -> Tensor:
        return attention_mask.sum(dim=1).long() - 1

    def boundary_module(self, layer: int):
        if layer < 0 or layer > self.n_layers:
            raise ValueError(f"Layer boundary must be in [0, {self.n_layers}]")
        return self.blocks[layer] if layer < self.n_layers else self.final_norm

    def boundary_activations(self, batch: TokenizedBatch, layer: int) -> Tensor:
        """Return the residual stream entering a layer boundary.

        Boundary ``0`` is the input to the first block and boundary ``n_layers``
        is the residual stream before the model's final normalization. This
        matches the original reproduction's resid-pre/resid-post convention.
        """
        captured: list[Tensor] = []

        def capture(_module, args):
            captured.append(args[0])

        handle = self.boundary_module(layer).register_forward_pre_hook(capture)
        try:
            self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one boundary activation, captured {len(captured)}")
        return captured[0]

    @contextmanager
    def edit_boundary(
        self,
        layer: int,
        editor: Callable[[Tensor], Tensor],
    ) -> Iterator[None]:
        module = self.boundary_module(layer)

        def pre_hook(_module, args):
            hidden = args[0]
            edited = editor(hidden)
            return (edited, *args[1:])

        handle = module.register_forward_pre_hook(pre_hook)
        try:
            yield
        finally:
            handle.remove()

    def extract_activations(
        self,
        examples: Sequence[TextExample],
        layer: int,
        *,
        position: str = "focus",
        batch_size: int = 16,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = self.tokenize(batch_examples).to(self.device_spec.device)
            with torch.inference_mode():
                hidden = self.boundary_activations(batch, layer)
            if position == "focus":
                if batch.focus_positions is None or (batch.focus_positions < 0).any():
                    raise ValueError("Focus positions are unavailable for one or more examples")
                positions = batch.focus_positions
            elif position == "final":
                positions = self.last_positions(batch.attention_mask)
            else:
                raise ValueError("Activation extraction position must be 'focus' or 'final'")
            rows = hidden[torch.arange(len(batch_examples), device=hidden.device), positions]
            chunks.append(rows.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def focus_is_single_token(self, example: TextExample) -> bool:
        if example.focus_start is None:
            return False
        batch = self.tokenize([example])
        encoded = self.tokenizer(
            example.text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        overlaps = [
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if end > example.focus_start and start < example.focus_end
        ]
        return len(overlaps) == 1 and batch.focus_positions is not None

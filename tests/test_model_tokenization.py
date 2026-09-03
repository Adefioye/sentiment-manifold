import torch
from types import SimpleNamespace

from sentiment_manifold.models.huggingface import CausalLMAdapter
from sentiment_manifold.types import TextExample


class _ToyTokenizer:
    bos_token_id = 99

    def __call__(
        self,
        texts,
        *,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        assert texts == ["ab"]
        assert padding and return_tensors == "pt"
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "offset_mapping": torch.tensor([[[0, 1], [1, 2]]]),
        }


def test_prepend_bos_matches_transformerlens_and_shifts_focus_position():
    adapter = CausalLMAdapter.__new__(CausalLMAdapter)
    adapter.tokenizer = _ToyTokenizer()
    adapter.prepend_bos = True
    example = TextExample(
        text="ab",
        label=1,
        example_id="example",
        focus_start=1,
        focus_end=2,
    )

    batch = adapter.tokenize([example])

    torch.testing.assert_close(batch.input_ids, torch.tensor([[99, 10, 11]]))
    torch.testing.assert_close(batch.attention_mask, torch.tensor([[1, 1, 1]]))
    torch.testing.assert_close(batch.focus_positions, torch.tensor([2]))


def test_gpt_neox_architecture_is_supported_for_pythia():
    layers = [object(), object()]
    norm = object()
    model = SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=layers, final_layer_norm=norm)
    )
    found_layers, found_norm = CausalLMAdapter._find_transformer_parts(model)
    assert found_layers is layers
    assert found_norm is norm

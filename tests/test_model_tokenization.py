import re
from types import SimpleNamespace

import torch

from sentiment_geometry.datasets import TextExample
from sentiment_geometry.models.huggingface import CausalLMAdapter


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


class _WhitespaceTokenizer:
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
        assert len(texts) == 1
        matches = list(re.finditer(r"\S+", texts[0]))
        assert padding and return_tensors == "pt" and return_offsets_mapping
        return {
            "input_ids": torch.tensor([[index + 1 for index in range(len(matches))]]),
            "attention_mask": torch.ones((1, len(matches)), dtype=torch.long),
            "offset_mapping": torch.tensor(
                [[[match.start(), match.end()] for match in matches]], dtype=torch.long
            ),
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
        named_spans={"adjective": (1, 2), "verb": (0, 1), "summary": (1, 2)},
    )

    batch = adapter.tokenize([example])

    torch.testing.assert_close(batch.input_ids, torch.tensor([[99, 10, 11]]))
    torch.testing.assert_close(batch.attention_mask, torch.tensor([[1, 1, 1]]))
    torch.testing.assert_close(batch.focus_positions, torch.tensor([2]))
    assert batch.named_positions is not None
    torch.testing.assert_close(batch.named_positions["adjective"], torch.tensor([2]))
    torch.testing.assert_close(batch.named_positions["verb"], torch.tensor([1]))
    torch.testing.assert_close(batch.named_positions["summary"], torch.tensor([2]))


def test_toy_named_positions_resolve_adj_vrb_sum_and_end_with_or_without_bos():
    text = "I thought this movie was amazing, I loved it. \nConclusion: This movie is"
    adjective_start = text.index("amazing")
    verb_start = text.index("loved")
    summary_start = text.rindex("movie")
    example = TextExample(
        text=text,
        label=1,
        example_id="toy-prompt",
        focus_start=adjective_start,
        focus_end=adjective_start + len("amazing"),
        named_spans={
            "adjective": (adjective_start, adjective_start + len("amazing")),
            "verb": (verb_start, verb_start + len("loved")),
            "summary": (summary_start, summary_start + len("movie")),
        },
    )

    expected = {
        True: {"adjective": 6, "verb": 8, "summary": 12, "final": 13},
        False: {"adjective": 5, "verb": 7, "summary": 11, "final": 12},
    }
    for prepend_bos, positions in expected.items():
        adapter = CausalLMAdapter.__new__(CausalLMAdapter)
        adapter.tokenizer = _WhitespaceTokenizer()
        adapter.prepend_bos = prepend_bos
        batch = adapter.tokenize([example])
        for name, expected_position in positions.items():
            assert int(adapter.activation_positions(batch, name)[0]) == expected_position


def test_gpt_neox_architecture_is_supported_for_pythia():
    layers = [object(), object()]
    norm = object()
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers, final_layer_norm=norm))
    found_layers, found_norm = CausalLMAdapter._find_transformer_parts(model)
    assert found_layers is layers
    assert found_norm is norm

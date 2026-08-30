"""Exact ToyMovieReview prompt family used to learn sentiment directions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import yaml

from ..types import CounterfactualPair, TextExample


@dataclass(frozen=True)
class ToyMovieReview:
    train: list[TextExample]
    test: list[TextExample]
    answers: dict[int, tuple[str, ...]]
    template: str
    adjectives: dict[str, dict[int, tuple[str, ...]]]
    verbs: dict[int, tuple[str, ...]]

    def paired(self, split: str) -> list[CounterfactualPair]:
        examples = self.train if split == "train" else self.test
        return pair_toy_examples(examples)

    def tokenizer_filtered(self, tokenizer) -> "ToyMovieReview":
        """Mirror upstream filtering of adjectives and verbs to one leading-space token."""

        def retained(words: tuple[str, ...]) -> list[str]:
            return [
                word
                for word in words
                if len(tokenizer(" " + word.strip(), add_special_tokens=False)["input_ids"]) == 1
            ]

        filtered_verbs = {label: retained(words) for label, words in self.verbs.items()}
        if not filtered_verbs[0] or not filtered_verbs[1]:
            raise RuntimeError("Tokenizer filtering removed every verb in one sentiment class")
        splits: dict[str, list[TextExample]] = {}
        filtered_adjectives: dict[str, dict[int, tuple[str, ...]]] = {}
        for split in ("train", "test"):
            filtered_adjectives[split] = {}
            split_examples: list[TextExample] = []
            for label in (1, 0):
                adjectives = retained(self.adjectives[split][label])
                filtered_adjectives[split][label] = tuple(adjectives)
                split_examples.extend(
                    _make_examples(
                        split,
                        label,
                        adjectives,
                        filtered_verbs[label],
                        self.template,
                    )
                )
            splits[split] = split_examples
        return ToyMovieReview(
            train=splits["train"],
            test=splits["test"],
            answers=self.answers,
            template=self.template,
            adjectives=filtered_adjectives,
            verbs={label: tuple(words) for label, words in filtered_verbs.items()},
        )


def pair_toy_examples(examples: list[TextExample]) -> list[CounterfactualPair]:
    """Construct Tigges's cyclic clean/corrupted pairing for an example subset."""

    positive = [example for example in examples if example.label == 1]
    negative = [example for example in examples if example.label == 0]
    n = min(len(positive), len(negative))
    # Tigges interleaves positive/negative clean prompts, then constructs the
    # corrupted batch with ``all_prompts[1:] + [all_prompts[0]]``.
    interleaved = [example for pair in zip(positive[:n], negative[:n]) for example in pair]
    corrupted = interleaved[1:] + interleaved[:1]
    return [
        CounterfactualPair(clean=clean, corrupted=corrupt)
        for clean, corrupt in zip(interleaved, corrupted)
    ]


def _make_examples(
    split: str,
    label: int,
    adjectives: list[str],
    verbs: list[str],
    template: str,
) -> list[TextExample]:
    examples: list[TextExample] = []
    for index, (adjective, verb) in enumerate(zip(adjectives, cycle(verbs))):
        text = template.format(adjective=adjective, verb=verb)
        focus_start = text.index(adjective)
        examples.append(
            TextExample(
                text=text,
                label=label,
                example_id=f"toy-{split}-{'pos' if label else 'neg'}-{index:03d}",
                focus_start=focus_start,
                focus_end=focus_start + len(adjective),
                metadata={"adjective": adjective, "verb": verb, "split": split},
            )
        )
    return examples


def load_toy_movie_review(path: str | Path) -> ToyMovieReview:
    raw = yaml.safe_load(Path(path).read_text())
    template = raw["template"]
    positive_verbs = raw["verbs"]["positive"]
    negative_verbs = raw["verbs"]["negative"]
    adjectives = {
        split: {
            1: tuple(raw[split]["positive_adjectives"]),
            0: tuple(raw[split]["negative_adjectives"]),
        }
        for split in ("train", "test")
    }
    verbs = {1: tuple(positive_verbs), 0: tuple(negative_verbs)}
    splits: dict[str, list[TextExample]] = {}
    for split in ("train", "test"):
        splits[split] = _make_examples(
            split,
            1,
            raw[split]["positive_adjectives"],
            positive_verbs,
            template,
        ) + _make_examples(
            split,
            0,
            raw[split]["negative_adjectives"],
            negative_verbs,
            template,
        )
    return ToyMovieReview(
        train=splits["train"],
        test=splits["test"],
        answers={
            1: tuple(raw["answers"]["positive"]),
            0: tuple(raw["answers"]["negative"]),
        },
        template=template,
        adjectives=adjectives,
        verbs=verbs,
    )

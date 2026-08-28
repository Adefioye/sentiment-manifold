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
    answers: dict[int, str]

    def paired(self, split: str) -> list[CounterfactualPair]:
        examples = self.train if split == "train" else self.test
        positive = [example for example in examples if example.label == 1]
        negative = [example for example in examples if example.label == 0]
        n = min(len(positive), len(negative))
        pairs: list[CounterfactualPair] = []
        for pos, neg in zip(positive[:n], negative[:n]):
            pairs.extend((CounterfactualPair(pos, neg), CounterfactualPair(neg, pos)))
        return pairs


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
        answers={1: raw["answers"]["positive"], 0: raw["answers"]["negative"]},
    )

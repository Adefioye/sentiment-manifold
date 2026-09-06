"""Exact ToyMovieReview prompt family used to learn sentiment directions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import yaml

from .types import CounterfactualPair, TextExample


@dataclass(frozen=True)
class ToyMovieReview:
    train: list[TextExample]
    test: list[TextExample]
    answers: dict[int, tuple[str, ...]]
    template: str
    adjectives: dict[str, dict[int, tuple[str, ...]]]
    verbs: dict[int, tuple[str, ...]]
    adverbs: dict[int, tuple[str, ...]]
    adverb_template: str
    adverb_answers: dict[int, tuple[str, ...]]

    def paired(self, split: str) -> list[CounterfactualPair]:
        examples = self.train if split == "train" else self.test
        return pair_toy_examples(examples)

    def tokenizer_filtered(self, tokenizer) -> ToyMovieReview:
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
            adverbs=self.adverbs,
            adverb_template=self.adverb_template,
            adverb_answers=self.adverb_answers,
        )


@dataclass(frozen=True)
class ToyEvaluationSet:
    """One mandatory ToyMovieReview evaluation family."""

    name: str
    examples: tuple[TextExample, ...]
    pairs: tuple[CounterfactualPair, ...]
    answers: dict[int, tuple[str, ...]]


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


def _deduplicate(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(words))


def _leading_space_token_length(tokenizer, word: str) -> int:
    return len(tokenizer(" " + word.strip(), add_special_tokens=False)["input_ids"])


def _pair_equal_length_examples(
    examples: list[TextExample], tokenizer, *, prepend_bos: bool
) -> list[CounterfactualPair]:
    """Make deterministic bidirectional pairs with equal full-prompt token length."""

    buckets: dict[tuple[int, int], list[TextExample]] = defaultdict(list)
    for example in examples:
        length = len(
            tokenizer(
                example.text,
                add_special_tokens=not prepend_bos,
            )["input_ids"]
        ) + int(prepend_bos)
        buckets[(length, example.label)].append(example)
    pairs: list[CounterfactualPair] = []
    for length in sorted({length for length, _ in buckets}):
        positives = buckets[(length, 1)]
        negatives = buckets[(length, 0)]
        for positive, negative in zip(positives, negatives):
            pairs.extend(
                (
                    CounterfactualPair(clean=positive, corrupted=negative),
                    CounterfactualPair(clean=negative, corrupted=positive),
                )
            )
    return pairs


def _make_word_examples(
    *,
    words: dict[int, tuple[str, ...]],
    template: str,
    placeholder: str,
    dataset_name: str,
    fixed_values: dict[str, str] | None = None,
) -> list[TextExample]:
    fixed_values = fixed_values or {}
    examples: list[TextExample] = []
    for label in (1, 0):
        for index, word in enumerate(words[label]):
            values = {**fixed_values, placeholder: word}
            text = template.format(**values)
            focus_start = text.index(word)
            examples.append(
                TextExample(
                    text=text,
                    label=label,
                    example_id=f"{dataset_name}-{'pos' if label else 'neg'}-{index:03d}",
                    focus_start=focus_start,
                    focus_end=focus_start + len(word),
                    metadata={
                        "evaluation_dataset": dataset_name,
                        "focus_word": word,
                        "focus_word_type": placeholder,
                    },
                )
            )
    return examples


def build_toy_evaluation_sets(
    dataset: ToyMovieReview,
    tokenizer,
    *,
    prepend_bos: bool,
) -> dict[str, ToyEvaluationSet]:
    """Build the required adjective, verb, and SimpleAdverb evaluations.

    The verb panel uses the eight paper-era ToyMovieReview verbs with a neutral
    adjective so the changing lexical item is the verb.  The adverb panel
    mirrors the upstream SimpleAdverb prompt and its exact two-token filter.
    All panels contain both intervention directions and equal-length prompts.
    """

    filtered = dataset.tokenizer_filtered(tokenizer)
    adjective_examples = tuple(filtered.test)
    adjective_pairs = tuple(filtered.paired("test"))

    verb_words = {
        label: tuple(
            word
            for word in _deduplicate(filtered.verbs[label])
            if _leading_space_token_length(tokenizer, word) == 1
        )
        for label in (1, 0)
    }
    verb_examples = _make_word_examples(
        words=verb_words,
        template=dataset.template,
        placeholder="verb",
        dataset_name="toy_verbs",
        fixed_values={"adjective": "average"},
    )
    verb_pairs = _pair_equal_length_examples(verb_examples, tokenizer, prepend_bos=prepend_bos)

    adverb_words = {
        label: tuple(
            word
            for word in _deduplicate(dataset.adverbs[label])
            if _leading_space_token_length(tokenizer, word) == 2
        )
        for label in (1, 0)
    }
    adverb_examples = _make_word_examples(
        words=adverb_words,
        template=dataset.adverb_template,
        placeholder="adverb",
        dataset_name="toy_adverbs",
    )
    adverb_pairs = _pair_equal_length_examples(adverb_examples, tokenizer, prepend_bos=prepend_bos)

    result = {
        "toy_adjectives": ToyEvaluationSet(
            "toy_adjectives", adjective_examples, adjective_pairs, filtered.answers
        ),
        "toy_verbs": ToyEvaluationSet(
            "toy_verbs", tuple(verb_examples), tuple(verb_pairs), filtered.answers
        ),
        "toy_adverbs": ToyEvaluationSet(
            "toy_adverbs",
            tuple(adverb_examples),
            tuple(adverb_pairs),
            dataset.adverb_answers,
        ),
    }
    empty = [name for name, evaluation in result.items() if not evaluation.pairs]
    if empty:
        raise RuntimeError(
            "Tokenizer filtering left no equal-length directional pairs for required "
            f"Toy evaluation datasets: {empty}"
        )
    return result


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
    adverbs = {
        1: tuple(raw["adverbs"]["positive"]),
        0: tuple(raw["adverbs"]["negative"]),
    }
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
        adverbs=adverbs,
        adverb_template=raw["adverb_template"],
        adverb_answers={
            1: tuple(raw["adverb_answers"]["positive"]),
            0: tuple(raw["adverb_answers"]["negative"]),
        },
    )

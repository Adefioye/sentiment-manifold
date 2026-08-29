"""Stanford Sentiment Treebank loader matching the paper's binary collapse."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import random

from ..types import CounterfactualPair, TextExample


SPLIT_NAMES = {1: "train", 2: "test", 3: "dev"}


def _read_phrase_scores(root: Path) -> dict[str, float]:
    scores_by_id: dict[int, float] = {}
    with (root / "sentiment_labels.txt").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            phrase_id, score = line.rstrip("\n").split("|")
            scores_by_id[int(phrase_id)] = float(score)
    scores: dict[str, float] = {}
    dictionary = root / "dictionary_fixed.txt"
    if not dictionary.exists():
        dictionary = root / "dictionary.txt"
    with dictionary.open(encoding="utf-8") as handle:
        for line in handle:
            phrase, phrase_id = line.rstrip("\n").rsplit("|", 1)
            if int(phrase_id) in scores_by_id and phrase not in scores:
                scores[phrase] = scores_by_id[int(phrase_id)]
    return scores


def load_sst(
    root: str | Path,
    split: str = "dev",
    *,
    scaffold: str = "continuation",
) -> list[TextExample]:
    """Load full sentences, discard neutral labels, and apply the paper scaffold."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"SST root does not exist: {root}. Set data.sst_root in the config."
        )
    scores = _read_phrase_scores(root)
    sentence_file = root / "datasetSentences_fixed.txt"
    if not sentence_file.exists():
        sentence_file = root / "datasetSentences.txt"
    sentences: dict[int, str] = {}
    with sentence_file.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sentences[int(row["sentence_index"])] = row["sentence"]
    split_ids: dict[int, str] = {}
    with (root / "datasetSplit.txt").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_ids[int(row["sentence_index"])] = SPLIT_NAMES[int(row["splitset_label"])]

    suffix = " Overall the movie was just very" if scaffold == "continuation" else ""
    examples: list[TextExample] = []
    for sentence_id, sentence in sentences.items():
        if split_ids.get(sentence_id) != split or sentence not in scores:
            continue
        score = scores[sentence]
        if score <= 0.4:
            label = 0
        elif score >= 0.6:
            label = 1
        else:
            continue
        text = sentence + suffix
        examples.append(
            TextExample(
                text=text,
                label=label,
                example_id=f"sst-{split}-{sentence_id}",
                metadata={"sentence": sentence, "score": score, "split": split},
            )
        )
    return examples


def pair_sst_by_token_length(
    examples: list[TextExample],
    tokenizer,
    *,
    max_pairs: int | None = None,
    seed: int = 0,
) -> list[CounterfactualPair]:
    """Pair positive/negative examples with equal token length in both directions."""
    buckets: dict[tuple[int, int], list[TextExample]] = defaultdict(list)
    for example in examples:
        length = len(tokenizer(example.text, add_special_tokens=True)["input_ids"])
        buckets[(length, example.label)].append(example)
    rng = random.Random(seed)
    pairs: list[CounterfactualPair] = []
    lengths = sorted({length for length, _ in buckets})
    for length in lengths:
        positive = buckets[(length, 1)]
        negative = buckets[(length, 0)]
        rng.shuffle(positive)
        rng.shuffle(negative)
        for pos, neg in zip(positive, negative):
            pairs.extend(
                (
                    CounterfactualPair(clean=pos, corrupted=neg),
                    CounterfactualPair(clean=neg, corrupted=pos),
                )
            )
            if max_pairs is not None and len(pairs) >= 2 * max_pairs:
                return pairs[: 2 * max_pairs]
    return pairs

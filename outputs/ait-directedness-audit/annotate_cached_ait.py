"""Model-assisted object-directedness audit for the cached AIT binary artifact.

This is a discovery artifact, not an original SemEval annotation. It scores five
exclusive labels using constrained next-token probabilities from a local
instruction-tuned model. Run the script with two model sizes and combine only
their agreements for conservative corpus summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = {
    "A": "target_directed",
    "B": "objectless_general",
    "C": "reported_quoted",
    "D": "mixed",
    "E": "ambiguous",
}

CODEBOOK = """Classify affect holder/directedness. Reply with one letter only.
A: author's affect/evaluation has an explicit external target or cause (person, object, event, situation, claim, topic, addressee). Example: "I hate this phone."
B: author's own mood/feeling/bodily affect has no explicit external target or cause. Time/location alone is not a target. Example: "Feeling low today."
C: affect/evaluation only belongs to another person or is merely reported/quoted; author stance is absent. Example: "She said she was devastated."
D: clear combination of B with a separate A or C frame. Example: "I feel miserable, and I hate this phone."
E: fragmentary, highly ironic, missing-context, or otherwise unclear affect holder/target.
Use A if author clearly reacts to an explicit cause; use C only when author stance is absent.
Tweet: """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260929)
    return parser.parse_args()


def load_rows(root: Path) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(root / f"{split}-00000-of-00001.parquet")
            for split in ("train", "validation", "test")
        ],
        ignore_index=True,
    ).sort_values("example_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_root)
    if args.sample_size is not None:
        rows = rows.sample(n=args.sample_size, random_state=args.sample_seed).sort_values(
            "example_id"
        ).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
    ).eval()

    letters = list(LABELS)
    token_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful linguistic annotator. Follow the exclusive "
                        "codebook exactly."
                    ),
                },
                {"role": "user", "content": CODEBOOK + str(text)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in rows["text"]
    ]

    records: list[dict[str, object]] = []
    for start in range(0, len(prompts), args.batch_size):
        stop = min(start + args.batch_size, len(prompts))
        batch = tokenizer(
            prompts[start:stop],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            logits = model(**batch).logits[:, -1, token_ids]
        probabilities = logits.softmax(dim=-1).cpu()
        for (_, source), probs in zip(rows.iloc[start:stop].iterrows(), probabilities):
            order = torch.argsort(probs, descending=True)
            best = int(order[0])
            second = int(order[1])
            record = source.to_dict()
            record.update(
                {
                    "annotation_letter": letters[best],
                    "annotation": LABELS[letters[best]],
                    "top_probability": float(probs[best]),
                    "second_letter": letters[second],
                    "second_probability": float(probs[second]),
                    "probability_margin": float(probs[best] - probs[second]),
                }
            )
            for index, letter in enumerate(letters):
                record[f"probability_{letter}"] = float(probs[index])
            records.append(record)
        print(f"annotated {stop}/{len(prompts)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

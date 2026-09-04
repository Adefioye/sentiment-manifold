# RQ2 — Valence/sentiment alignment and OOD transfer

## Research question

How well do valence directions learned from AIT V-oc align with sentiment directions, and how
causally useful are those directions on SST, IMDb, and DynaSent across model families?

## Data contract

- AIT V-oc is the supervised valence-direction training dataset. Its gold ordinal labels are
  collapsed to binary polarity for pairing, but examples are not correctness-filtered by a model.
- SST, IMDb, and DynaSent are zero-shot evaluation datasets. Pythia-2.8B must agree with each
  dataset's gold binary label before an example can enter an equal-length patching pair.
- Gold labels are never replaced by Pythia predictions.
- Model-specific pairs have equal full-prompt length under the selected tokenizer. Common pairs
  have equal length under all four configured tokenizers.

## Supported pairing tokenizers

- GPT-2 Small
- Qwen3-0.6B Base
- Gemma 2B
- Pythia 1.4B

Pythia-2.8B is the default **selection model**, not an additional pairing target.

## Guides

- [Preprocessing guide](PREPROCESSING.md): dataset commands, correctness filtering, output
  configurations, provenance, publishing, and cross-model pairing policy.
- [Colab notebook](../../notebooks/02_colab_preprocess_publish_explore_rq2.ipynb): end-to-end
  preprocessing, private Hub publication, reload checks, and descriptive data exploration.
- [RQ1 reproduction](../rq-1/README.md): the separate Tigges Table 1 protocol, which retains its
  original Pythia-1.4B SST correctness filter.

## Status

Preprocessing and tokenizer-aware pair construction are implemented. Direction fitting and causal
evaluation on these RQ2 artifacts should preserve train/validation/test separation and freeze the
chosen pair policy before confirmation runs.

# Experiment Logs

This file records sentiment-manifold experiments in reverse chronological order. Each entry states
the date, purpose, protocol, status, and primary outputs so that later experiments can follow the
same structure.

## 2026-09-22 — Frozen END-direction OOD evaluation

**Status:** GPT-2 Small evaluation complete; Qwen3-0.6B Base resume configured at batch size 8
after an out-of-memory interruption.

**Description:** Evaluate whether the previously selected END-position sentiment directions transfer
to SST, IMDb, DynaSent R1, and DynaSent R2 for GPT-2 Small and Qwen3-0.6B Base.

**Protocol:** Reuse the mean-difference, logistic-regression, and one-dimensional DAS directions and
layers selected on ToyMovieReview ADVERB by `logit_flip_percent` in parent run
`2026-09-18_20-22_CDT`; do not refit directions or reselect layers on the OOD datasets. The resumed
run evaluates only Qwen and preserves the completed GPT-2 files. Report separate numeric-only tables
for `logit_flip_percent` and `sign_flip_percent` for both models.

**Primary entry point:** `notebooks/05_colab_evaluate_frozen_end_directions.ipynb`

**Outputs:** Model-level and combined metrics, layer-selection records, dataset summaries, per-case
patching records, resolved configurations, and manifests under Google Drive run
`end-position-ood-evaluation/runs/2026-09-22_00-18_CDT`.

## 2026-09-19 — Sentiment position comparison

**Status:** Implementation and automated validation complete; full GPU sweep pending.

**Description:** Compare whether causal sentiment directions depend on the activation position used
for fitting. Mean difference, logistic regression, and one-dimensional DAS directions are fitted on
ToyMovieReview training prompts at ADJ, VRB, SUM (the second `movie`), and END (the final `is`) across
all non-embedding residual boundaries for GPT-2 Small and Qwen3-0.6B Base.

**Protocol:** The complete ToyMovieReview ADVERB panel selects the DAS checkpoint and then the layer
with the highest `logit_flip_percent` for each model × method × fitting-position direction. That
layer is frozen before post-selection evaluation on ADVERB, held-out ADJ, and SST. The reported
metrics are `logit_difference_percent`, `logit_flip_percent`, and `sign_flip_percent`.

**Configuration:** `configs/sentiment_position_comparison.yaml`

**Outputs:** Direction artifacts, per-pair patching records, aggregate metric CSVs, layer-selection
records, DAS epoch metrics, direction similarities, manifests, and metric plots are saved in the
timestamped Google Drive run directory.

## Entry template

## YYYY-MM-DD — Experiment name

**Status:** Planned, running, completed, failed, or superseded.

**Description:** Briefly state the question and what is being compared.

**Protocol:** Record the training data, selection data and metric, frozen evaluation data, models,
methods, interventions, and controls.

**Primary entry point:** Path to the executable script or notebook if it exists.

**Configuration:** Path to the resolved or source configuration.

**Outputs:** Location and names of the principal artifacts and result tables.

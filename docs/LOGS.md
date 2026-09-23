# Experiment Logs

This file records sentiment-manifold experiments in reverse chronological order. Each entry states
the date, purpose, protocol, status, and primary outputs so that later experiments can follow the
same structure.

## 2026-09-23 — Full AIT last-token direction training

**Status:** Completed for GPT-2 Small and Qwen3-0.6B Base in one resumed Drive run.

**Description:** Train last-token mean-difference, logistic-regression, and one-dimensional DAS
directions across all non-embedding residual boundaries using every available example in each
model's tokenizer-matched AIT train, validation, and test splits.

**Protocol:** Fit on the original train split, select DAS epochs and the best layer for every method
on validation `logit_flip_percent`, and reserve the original test split for locked final evaluation.
GPT-2 completed in the initial phase; Qwen was restarted and completed with model/evaluation and DAS
batch sizes of 16. The run contains 321/99/273 matched train/validation/test pairs for GPT-2 and
323/118/274 for Qwen.

**Primary entry point:** `notebooks/07_colab_train_full_ait_last_token_directions.ipynb`

**Configuration:** `configs/full_ait_valence_directions.yaml`

**Outputs:** Per-model direction checkpoints and CSVs, essential two-model validation/test tables,
per-case patching records, first/middle/last-boundary cosine summaries and plots, manifests, and
logs under Google Drive run `full-ait-last-token-directions/runs/2026-09-23_09-10_CDT`.

## 2026-09-23 — Sampled AIT last-token direction training

**Status:** Completed.

**Description:** Train last-token mean-difference, logistic-regression, and one-dimensional DAS
valence directions for GPT-2 Small and Qwen3-0.6B Base across all non-embedding residual
boundaries using each model's tokenizer-matched AIT data.

**Protocol:** Use 55 training examples, 30 validation directed cases, and 30 locked-test directed
cases per model. Validation selects the DAS epoch and the best layer for every method by
`logit_flip_percent`; test data is opened only for final logit-flip and sign-flip evaluation.

**Primary entry point:** `notebooks/06_colab_train_ait_last_token_directions.ipynb`

**Configuration:** `configs/ait_valence_directions.yaml`

**Outputs:** Direction checkpoints, per-model and combined CSVs, per-case patching records,
selection tables, direction-similarity summaries, manifests, and plots under Google Drive run
`ait-last-token-directions/runs/2026-09-23_02-47_CDT`.

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

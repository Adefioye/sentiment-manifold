---
name: sentiment-research
description: Plan, implement, run, diagnose, and interpret reproducible sentiment-direction and sentiment-manifold experiments in this repository. Use for ToyMovieReview, SST, OpenWebText, GPT-2, Qwen, direction fitting, DAS, activation patching, steering, layer sweeps, representation geometry, circuit follow-up, notebooks, or paper replication involving sentiment in language models.
---

# Sentiment Research

Use the executable package as the source of truth. Keep scientific decisions explicit in configuration and preserve benchmark provenance.

## Start with the research phase

Classify the request before editing code:

1. **Replication:** preserve Tigges prompts, splits, token filtering, intervention semantics, and metrics. Read `references/methodology.md` and `../../../docs/REPRODUCTION.md`.
2. **Discovery:** explore layers, directions, dimensions, geometry, or failure modes using train/validation data only. Read `../../../docs/rq-sentiment-manifolds/03-methodology.md`.
3. **Confirmation:** freeze the selected method, layer, strength, and controls before opening the held-out/OOD result.
4. **Mechanism:** begin component/path analysis only after a representation survives causal baselines.

Never describe probe accuracy, PCA layout, UMAP, reconstruction, or feature correlations as evidence of causal use by themselves.

## Implement through stable boundaries

- Put dataset parsing and pairing in `src/sentiment_manifold/data/`.
- Put model architecture and hook differences in `src/sentiment_manifold/models/`.
- Expose every fitting method through a normalized, positive-oriented direction artifact.
- Put causal metrics in `src/sentiment_manifold/evaluation/`; do not bury them in notebooks.
- Keep notebooks thin: call package APIs, visualize saved tables, and avoid unique experiment logic.
- Route all paths and sweeps through `configs/reproduction.yaml`.
- Preserve `cuda`/`mps`/`cpu` behavior and avoid unconditional CUDA calls.

Read `references/library-routing.md` before introducing a new mech-interp dependency.

## Use the baseline ladder

For a new representation claim, compare in this order:

1. mean difference, K-means centroid difference, logistic-regression weight, PCA-1;
2. 1D DAS;
3. dimension-matched PCA/DAS or fixed linear subspace;
4. only then splines, local atlases, SAEs/feature blocks, flows, or pullback paths.

For every nonlinear candidate include the same-subspace straight chord, matched dimension, matched intervention norm/effect, random direction, and full-activation upper bound where feasible.

## Validate before reporting

Run focused unit tests, then a tiny CPU smoke configuration. Check:

- exact dataset counts and immutable IDs;
- tokenizer-specific single-token labels and focus spans;
- layer numbering `0..n_layers` and saved model/tokenizer revisions;
- positive direction orientation and unit norm;
- no train/test leakage in layer or hyperparameter selection;
- per-pair recovery/flip records, not aggregates alone;
- OpenWebText as an unlabeled loss/support diagnostic, not a sentiment-selection label.

Report discrepancies from the paper rather than adjusting the benchmark silently. Preserve negative results and resolved configs.


# Implementation summary

## What is implemented

The package supplies a shared Hugging Face backend for GPT-2 Small (`gpt2`) and the 28-layer Qwen3-0.6B base model (`Qwen/Qwen3-0.6B-Base`). The base checkpoint is used for a fairer continuation-model comparison with GPT-2; the post-trained chat checkpoint would be a separate experiment. The adapter exposes residual boundaries `0..n_layers`, activation extraction, and scoped activation editing without model-specific logic leaking into fitters or evaluations.

Five positive-oriented unit-direction APIs are implemented:

- mean difference;
- K-means centroid difference;
- L2 logistic-regression weight;
- first principal component, sign-oriented by class means;
- one-dimensional DAS optimized by directional counterfactual replacement and next-token cross-entropy.

The experiment runner fits every configured method at every configured boundary and writes portable compressed direction checkpoints. It reports:

- ToyMovieReview held-out projection accuracy, patching recovery, and flip rate;
- SST final-position projection accuracy and all-token paired patching recovery/flip rate;
- within-layer absolute cosine similarity among all five directions;
- optional exploratory OpenWebText baseline loss, resample-ablated loss, loss increase, and four
  matched random-direction controls, enabled only by
  `experiment.openwebtext_resample_ablation` or `--with-openwebtext-resample-ablation`;
- one best SST-validation layer per method.

## Package map

```text
src/sentiment_manifold/
├── data/          ToyMovieReview, SST, OpenWebText
├── models/        Hugging Face residual-boundary adapter
├── directions/    common fitter API, four linear fits, DAS
├── evaluation/    projections, causal patching, OWT ablation
├── experiment.py  configured all-layer sweep
├── plotting.py    layer curves and similarity heatmaps
└── storage.py     local/Colab/Drive output routing
```

Notebooks call these APIs rather than defining separate experiment implementations. `CLAUDE.md` and `.claude/skills/sentiment-research/` provide project-aware agent guidance.

## Deliberate differences from the reference code

The reference repository uses a historical TransformerLens fork and notebook-like scripts. This package uses native Hugging Face models to give GPT-2 and Qwen the same interface, explicit configuration, portable artifacts, unit tests, and safe CUDA/MPS/CPU selection. Because hook implementations can differ, modern-backend GPT-2 results are a faithful methodological reproduction, not automatically an exact numerical reproduction. Use the legacy parity procedure in `REPRODUCTION.md` before claiming number-level agreement.

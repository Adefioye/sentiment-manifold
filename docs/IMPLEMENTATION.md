# Implementation summary

## What is implemented

The package supplies a shared Hugging Face backend for GPT-2 Small (`gpt2`) and the 28-layer Qwen3-0.6B base model (`Qwen/Qwen3-0.6B-Base`). The base checkpoint is used for a fairer continuation-model comparison with GPT-2; the post-trained chat checkpoint would be a separate experiment. The adapter exposes residual boundaries `0..n_layers`, activation extraction, and scoped activation editing without model-specific logic leaking into fitters or evaluations.

The configured Tigges direction sweep contains:

- mean difference;
- K-means centroid difference;
- L2 logistic-regression weight;
- first principal component, with its raw arbitrary sign recorded and its exported 1D artifact
  consistently oriented positive for interpretability;
- one-, two-, and three-dimensional DAS learned through an orthogonally parameterized rotation and
  Tigges's normalized logit-difference recovery objective for 64 epochs;
- Tigges's layer-indexed Gaussian random control.

The sign of a one-dimensional axis is arbitrary: positive orientation does not make it a different
line. It only fixes the reporting convention. For 2D/3D DAS the intervention depends on the subspace,
not on the signs of individual basis columns. Every artifact records the convention and reference in
`direction_metadata.csv`.

The experiment runner fits every configured method at every configured boundary and writes portable compressed direction checkpoints. It reports:

- ToyMovieReview held-out projection accuracy, patching recovery, and target-directed logit-flip
  percentage;
- SST final-position projection accuracy and all-token paired patching recovery/logit-flip
  percentage;
- within-layer absolute cosine similarity among the one-dimensional directions;
- optional exploratory OpenWebText baseline loss, resample-ablated loss, loss increase, and four
  matched random-direction controls, enabled only by
  `experiment.openwebtext_resample_ablation` or `--with-openwebtext-resample-ablation`;
- one best paper-style SST-test layer per method.

Logit-difference recovery uses the clean target's answer ordering, matching Tigges et al.'s
``[correct, incorrect]`` answer-token convention. The model runs on the corrupted prompt; only the
selected activation projection is copied from the clean run at every prompt position, producing the
patched run. The primary `*_logit_flip_percent` exactly follows Tigges's `logit_flip_denoising`: it
centers per-prompt correct-minus-incorrect differences, converts their signs to accuracy, and
calibrates patched accuracy between corrupted and clean accuracy. `*_sign_flip_percent` is retained
as a separate audit statistic for literal corrupted-to-patched sign changes toward the clean target.
It must not be substituted for the paper metric. `metrics.csv` preserves normalized and paper-scale
columns for both.

## Package map

```text
src/sentiment_manifold/
├── data/          ToyMovieReview, SST, OpenWebText
├── models/        Hugging Face residual-boundary adapter
├── directions/    common fitter API, four linear fits, DAS
├── evaluation/    projections, causal patching, OWT ablation
├── experiment.py  configured all-layer sweep
├── plotting.py    causal curves, DAS loss curves, and similarity heatmaps
└── storage.py     local/Colab/Drive output routing
```

Notebooks call these APIs rather than defining separate experiment implementations. `CLAUDE.md` and `.claude/skills/sentiment-research/` provide project-aware agent guidance.

## Deliberate differences from the reference code

The reference repository uses TransformerLens and notebook-like scripts. This package uses native
Hugging Face models to give GPT-2 and Qwen the same interface, explicit configuration, portable
artifacts, and tests. The GPT-2 revision and BOS convention are pinned, but hook implementations,
the deliberately rejected upstream `SIMPLE_TRAIN` prompt-count anomaly, reconstructed SST pair
identities, and runtime libraries can still produce numerical differences. Treat GPT-2 as
implementation-parity work until
the saved prompt/token/pair manifests and causal curves agree with the reference run; Qwen is an
extension rather than a paper replication. `REPRODUCTION.md` records these boundaries explicitly.

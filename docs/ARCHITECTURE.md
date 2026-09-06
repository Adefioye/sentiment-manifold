# Code architecture

The executable Python package lives directly at `sentiment_geometry/`; this repository does not use
a `src/` indirection. All imports begin with `sentiment_geometry` so local modules cannot shadow the
third-party Hugging Face `datasets` package.

## Domain boundaries

- `datasets` owns text examples, counterfactual pairs, loading, pairing, and preprocessing.
- `models` owns architecture adapters, tokenization, residual-boundary hooks, and device behavior.
- `activations` owns reusable batching and activation extraction from model adapters.
- `fitting_methods` owns direction/subspace fitters, their configuration, and fitted artifacts.
- `interventions` owns pure tensor transformations that change model activations.
- `evaluation` applies interventions and calculates causal or language-model outcomes.
- `analysis` calculates non-causal projection and geometric diagnostics.
- `selection` owns validation-only tuning and application of frozen selections.
- `experiments` composes domain APIs into configuration-driven scientific workflows.
- `reporting` converts saved results into selected tables and figures.
- `persistence` owns output/checkpoint locations and serialization conventions.
- `cli` parses command-line arguments and delegates to public APIs.

Dependencies should point toward domain APIs rather than experiment internals. Datasets never run
models; fitting methods never write reports; interventions never aggregate evaluation metrics; and
notebooks never define unique experiment logic. Domain-specific types live with the domain that
owns their invariants instead of a shared `types.py` or `utils.py` catch-all.

## Repository-level assets

`configs/`, `data/`, `docs/`, `notebooks/`, `scripts/`, `tests/`, and generated `outputs/` remain
outside the Python package. The root `data/` directory contains static or generated resources;
`sentiment_geometry/datasets/` contains the Python code that interprets them.

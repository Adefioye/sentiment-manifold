# Sentiment Manifold

Research code for reproducible causal studies of sentiment representations in language models.
The current implemented research question reproduces the essential sentiment-direction experiments
from Tigges et al., *Language Models Linearly Represent Sentiment*, and RQ2 preprocessing for
GPT-2 Small, Qwen3-0.6B Base, Gemma 2B, and Pythia 1.4B.

## Quick start

```bash
cd sentiment-manifold
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebooks]'

sentiment-manifold inspect-data --model gpt2-small
sentiment-manifold preprocess-sst --binarization both \
  --filter-model pythia-1.4b --output-dir data/processed/sst-pythia-1.4b
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small
sentiment-manifold plot --run-dir outputs/results/gpt2-small
```

Model aliases are `gpt2-small`, `qwen-0.6b`, `gemma-2b`, and `pythia-1.4b`. Commands support CUDA,
MPS, and CPU through `--device auto`, `cuda`, `mps`, or `cpu`.

RQ2 preprocessing uses Pythia-2.8B to correctness-filter SST, IMDb, and DynaSent; AIT is not
correctness-filtered. The explicit Pythia-1.4B option above preserves the separate RQ1 benchmark.

For Colab, use
[`notebooks/01_colab_sst_to_gpt2_results.ipynb`](notebooks/01_colab_sst_to_gpt2_results.ipynb).

## Documentation

- **RQ1 — Tigges reproduction:** [overview and command index](docs/rq-1/README.md),
  [full reproduction protocol](docs/rq-1/REPRODUCTION.md),
  [per-method commands and optional tuning](docs/rq-1/METHOD_COMMANDS.md), and
  [implementation details](docs/rq-1/IMPLEMENTATION.md).
- **RQ2 — valence/sentiment alignment data:**
  [overview](docs/rq-2/README.md) and
  [AIT, SST, IMDb, and DynaSent preprocessing guide](docs/rq-2/PREPROCESSING.md).
- **Sentiment-manifold research program:**
  [research questions, methodology, diagnostics, and roadmap](docs/rq-sentiment-manifolds/README.md).

Planned work will keep separate research-question directories for geometric alignment between
valence and sentiment directions and for nonlinear/manifold representations of sentiment.

## Outputs

Lightweight results are written to `outputs/results/<model>/`. Reusable fitted directions are
written separately to `checkpoints/<model>/<phase>/<method>/<configuration>/`. Both locations can
be changed through configuration or command-line options.

Before reporting an exact reproduction, follow the parity checklist and known-boundary discussion
in the [RQ1 reproduction guide](docs/rq-1/REPRODUCTION.md).

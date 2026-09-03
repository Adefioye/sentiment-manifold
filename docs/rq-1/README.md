# RQ1 — Reproduce Tigges et al.

## Research question

Can the essential experiments in Tigges et al., *Language Models Linearly Represent Sentiment*, be
reproduced with an auditable Hugging Face implementation, and how do the same methods behave on a
small modern base model?

This directory is the complete entry point for the implemented reproduction. GPT-2 Small is the
paper-reproduction target. Qwen3-0.6B Base is an extension and must not be described as part of the
original paper replication.

## Guides

- [Reproduction protocol](REPRODUCTION.md): stages, Colab storage, parity checks, metrics, and saved
  artifacts.
- [Per-method commands](METHOD_COMMANDS.md): copy-ready runs for every direction method and the
  leakage-safe optional tuning/confirmation workflow.
- [Implementation summary](IMPLEMENTATION.md): supported models, direction methods, causal metrics,
  package layout, and deliberate differences from the reference code.

## Implemented scope

The experiment fits the following methods at every configured residual-stream boundary:

- positive-minus-negative mean difference;
- K-means centroid difference;
- L2 logistic-regression weight;
- first principal component with a recorded orientation convention;
- 1D, 2D, and 3D Distributed Alignment Search (DAS);
- layer-indexed random-direction controls.

It evaluates held-out ToyMovieReview projection accuracy, ToyMovieReview and SST directional
patching, logit-difference recovery, logit-flip metrics, and within-layer direction similarity.
OpenWebText language-model-loss resample ablation is available only as an optional exploratory
diagnostic; it is not the paper's GPT-4-labelled OpenWebText projection experiment.

## Installation and first run

Run commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebooks]'

sentiment-manifold inspect-data --config configs/reproduction.yaml --model gpt2-small
sentiment-manifold preprocess-sst --binarization both
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --device auto
sentiment-manifold plot --run-dir outputs/results/gpt2-small
```

For the 28-layer Qwen3-0.6B base checkpoint:

```bash
sentiment-manifold inspect-data --config configs/qwen_reproduction.yaml --model qwen-0.6b
sentiment-manifold reproduce --config configs/qwen_reproduction.yaml --model qwen-0.6b --device auto
sentiment-manifold plot --run-dir outputs/results/qwen-0.6b
```

Use `--device auto`, `cuda`, `mps`, or `cpu`. Automatic selection prefers CUDA, then MPS, then CPU.
For a quick systems check rather than a reportable run, reduce the layer list and DAS epochs in a
copied configuration and limit the SST pair count.

## Secure SST preprocessing and Hugging Face upload

The SST preprocessing command uses `EleutherAI/pythia-1.4b`. It saves both supported binary-label
families and can upload the processed dataset to a private Hugging Face repository.

Supply a write-capable token through a secret environment variable; do not paste a literal `hf_…`
token into a notebook cell or shell command:

```bash
HF_TOKEN="$MY_HF_WRITE_TOKEN" sentiment-manifold preprocess-sst --push-to-hub --private
```

Here `MY_HF_WRITE_TOKEN` should already have been populated by a password manager, CI secret, or
Colab secret. If `HF_TOKEN` is already exported, run:

```bash
sentiment-manifold preprocess-sst --push-to-hub --private
```

A differently named secret and Hugging Face's token-file convention are also supported:

```bash
sentiment-manifold preprocess-sst --push-to-hub --private --hf-token-env MY_PRIVATE_HF_TOKEN
HF_TOKEN_PATH=/run/secrets/huggingface-token sentiment-manifold preprocess-sst --push-to-hub --private
```

Credentials are checked before Pythia is downloaded or evaluated. The token is passed to the Hub
APIs and is not stored in run metadata or uploaded artifacts. Choose another destination with
`--hub-repo-id namespace/name`; use `--public` only when a public dataset repository is intended.

The default `--binarization both` produces these local Hugging Face configurations under
`data/processed/sst-pythia-1.4b`:

- `{tigges,neutral_removed}_binarized`;
- `{tigges,neutral_removed}_pythia_scored`;
- `{tigges,neutral_removed}_pythia_correct`;
- `{tigges,neutral_removed}_matched_pairs`;
- `{tigges,neutral_removed}_directed_pairs`.

The `tigges` rule maps scores `<= 0.5` to negative and scores `> 0.5` to positive, matching the
paper code's `int(round(score))`. The `neutral_removed` alternative maps scores `<= 0.4` to
negative, removes `(0.4, 0.6]`, and maps scores `> 0.6` to positive. Both use SST test prompts of
the form `Review Text: … Review Sentiment:` with `Positive` and `Negative` answers and retain only
examples classified correctly by Pythia-1.4B.

The reproduction runner consumes `tigges_pythia_correct` and reconstructs equal-token-length pairs
with the target model's tokenizer. Pythia remains the correctness filter.

To read a private uploaded configuration:

```python
import os
from datasets import load_dataset

candidate_pool = load_dataset(
    "your-namespace/sentiment-manifold-sst-pythia-1.4b",
    "tigges_pythia_correct",
    token=os.environ["HF_TOKEN"],
)
```

## Principal reproduction commands

```bash
sentiment-manifold inspect-data --config configs/reproduction.yaml
sentiment-manifold preprocess-sst --binarization both
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --device auto
sentiment-manifold plot --run-dir outputs/results/gpt2-small
```

For individual mean-difference, K-means, logistic-regression, PCA, DAS, and random-control runs, use
the [per-method command guide](METHOD_COMMANDS.md).

## Optional OpenWebText resample ablation

The optional diagnostic shuffles each example's scalar projection onto a fitted direction between
unrelated OpenWebText examples while preserving the orthogonal activation component. It reports the
resulting language-model-loss increase alongside matched random-direction controls.

```bash
sentiment-manifold reproduce \
  --config configs/reproduction.yaml \
  --model gpt2-small \
  --with-openwebtext-resample-ablation
```

The equivalent configuration is:

```yaml
experiment:
  openwebtext_resample_ablation: true
```

This sweep is expensive and should remain disabled for smoke runs. The paper's principal
OpenWebText analysis instead examines first-layer sentiment projections of GPT-4-labelled extreme
tokens and must be reproduced separately.

## Result and checkpoint locations

```text
outputs/results/<model>/
checkpoints/<model>/<phase>/<method>/<configuration-fingerprint>/
```

Completed runs keep all per-layer measurements in `metrics.csv`; `best_layers.csv` independently
records the maximizing layer for each ToyMovieReview/SST × logit-difference/logit-flip cell. The
plotting command renders this summary and the saved causal and similarity diagnostics under the
run's `figures/` directory.

In Colab, lightweight result files can remain local while fitted directions persist on Drive:

```bash
sentiment-manifold reproduce \
  --config configs/reproduction.yaml \
  --model gpt2-small \
  --checkpoint-dir /content/drive/MyDrive/sentiment-manifold/checkpoints
```

## Reproducibility boundary

The project preserves the paper-intended ToyMovieReview 55/30 adjective split, eight verbs,
reference prompt bytes, aligned answer pairs, BOS convention, cyclic clean/corrupted pairing,
all-position patching, and causal metrics. SST comes from the adjacent
`eliciting-latent-sentiment` checkout rather than being duplicated.

The package uses native Hugging Face hooks instead of the historical TransformerLens backend, and
it deterministically reconstructs SST pairs from the saved Pythia-correct candidate pool. Exact
paper-number claims therefore require the manifest and metric checks in the
[full reproduction protocol](REPRODUCTION.md).

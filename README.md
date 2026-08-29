# Sentiment Manifold

A standalone, reusable reproduction of the essential sentiment-direction experiments from Tigges et al., *Language Models Linearly Represent Sentiment*.

The implemented baseline fits Tigges's mean-difference, K-means, logistic-regression, PCA, 1D/2D/3D DAS, and random controls at every residual-stream layer of GPT-2 Small and Qwen3-0.6B. It evaluates ToyMovieReview generalization, paired causal recovery on SST, and direction similarity. OpenWebText language-model loss under directional resample ablation is available as an optional exploratory diagnostic.

## Quick start

```bash
cd sentiment-manifold
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebooks]'

sentiment-manifold inspect-data --model gpt2-small
sentiment-manifold preprocess-sst --binarization both
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small
sentiment-manifold plot --run-dir outputs/gpt2-small
```

For the 28-layer Qwen3-0.6B base checkpoint, use `--model qwen-0.6b`. Use `--device auto`, `cuda`, `mps`, or `cpu`; automatic selection prefers CUDA, then MPS, then CPU.

## Preprocess SST with Pythia-1.4B

Provide a Hugging Face user access token with write permission through `HF_TOKEN`, then run this single command from the project directory:

```bash
HF_TOKEN="$MY_HF_WRITE_TOKEN" sentiment-manifold preprocess-sst --push-to-hub --private
```

`MY_HF_WRITE_TOKEN` should already be populated by your shell, password manager, CI secret, or Colab secret. Avoid placing the literal `hf_...` value directly in shell history. For an exported secret, simply run `sentiment-manifold preprocess-sst --push-to-hub --private`; Hugging Face gives `HF_TOKEN` priority over any stored login. A differently named secret is supported without copying it:

```bash
MY_PRIVATE_HF_TOKEN="$MY_PRIVATE_HF_TOKEN" sentiment-manifold preprocess-sst --push-to-hub --private --hf-token-env MY_PRIVATE_HF_TOKEN
```

The command also supports Hugging Face's standard secure token-file setting:

```bash
HF_TOKEN_PATH=/run/secrets/huggingface-token sentiment-manifold preprocess-sst --push-to-hub --private
```

Credentials are checked before Pythia is downloaded or evaluated. The token is passed directly to the Hub APIs and is never written to local metadata or uploaded files.

The command uses `EleutherAI/pythia-1.4b` and saves two binary-label families. `tigges` assigns
negative to scores `<= 0.5` and positive to scores `> 0.5`, matching the paper code's
`int(round(score))`. `neutral_removed` preserves the alternative rule: negative `<= 0.4`, remove
`(0.4, 0.6]`, and positive `> 0.6`. Both use the SST test split, the
`Review Text: … Review Sentiment:` classifier with `Positive`/`Negative` answers, and retain only
examples Pythia classifies correctly. The default `--binarization both` saves ten local Hugging Face
configurations under `data/processed/sst-pythia-1.4b` and can publish them to
`<authenticated-user>/sentiment-manifold-sst-pythia-1.4b`:

- `{tigges,neutral_removed}_binarized`;
- `{tigges,neutral_removed}_pythia_scored`;
- `{tigges,neutral_removed}_pythia_correct`;
- `{tigges,neutral_removed}_matched_pairs`;
- `{tigges,neutral_removed}_directed_pairs`.

The reproduction runner loads `tigges_pythia_correct` and re-pairs it with the GPT-2 tokenizer;
Pythia remains the only correctness filter. The candidate pool is uncapped. Pass
`--hub-repo-id namespace/name` to choose a different repository or `--public` to publish publicly.

Read a private configuration using the same environment token:

```python
import os
from datasets import load_dataset

candidate_pool = load_dataset(
    "your-namespace/sentiment-manifold-sst-pythia-1.4b",
    "tigges_pythia_correct",
    token=os.environ["HF_TOKEN"],
)
```

## Optional OpenWebText resample ablation

Resample ablation is disabled by default and is not required for reproducing the paper's principal OpenWebText result. It shuffles the scalar projection onto a fitted sentiment direction between unrelated OpenWebText examples, preserves each activation's orthogonal component, and reports the resulting language-model loss increase. Random directions provide matched controls.

Enable this exploratory diagnostic explicitly:

```bash
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --with-openwebtext-resample-ablation
```

The equivalent configuration is:

```yaml
experiment:
  openwebtext_resample_ablation: true
```

It is expensive because it evaluates every selected layer and fitting method, followed by the configured random-direction controls. Keep it disabled for smoke runs. The paper's OpenWebText evaluation instead studies first-layer sentiment projections using GPT-4-labelled tokens; that correlational evaluation should be reported separately from this optional loss diagnostic.

The default configuration fits every layer with resample ablation disabled. For a smoke run, reduce `experiment.layers`, `das.epochs`, and the SST pair limit. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) and [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Reproducibility boundary

The ToyMovieReview 55/30 adjective split and eight verbs follow the paper-era inventory recoverable
from the MIT-licensed reference repository's history. Prompt bytes, answer pairs, BOS convention,
cyclic clean/corrupted pairing, all-position patching, and causal metrics otherwise follow the
accompanying code in `../eliciting-latent-sentiment`. SST is read from that checkout without
duplicating the dataset. The package uses Hugging Face hooks rather than the historical TransformerLens
backend, so exact paper-number claims still require the parity checks in the reproduction guide.

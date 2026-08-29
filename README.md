# Sentiment Manifold

A standalone, reusable reproduction of the essential sentiment-direction experiments from Tigges et al., *Language Models Linearly Represent Sentiment*.

The implemented baseline fits five directions—mean difference, K-means, logistic regression, PCA, and one-dimensional DAS—at every residual-stream layer of GPT-2 Small and Qwen3-0.6B. It evaluates ToyMovieReview generalization, paired causal recovery on SST, and direction similarity. OpenWebText language-model loss under directional resample ablation is available as an optional exploratory diagnostic.

## Quick start

```bash
cd sentiment-manifold
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebooks]'

sentiment-manifold inspect-data --model gpt2-small
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

The command uses `EleutherAI/pythia-1.4b`, removes SST's official neutral interval `(0.4, 0.6]`, filters the test candidates to examples Pythia classifies correctly, constructs every possible non-reused opposite-label match with equal Pythia token lengths, and represents each match in both patching directions. It saves five local Hugging Face configurations under `data/processed/sst-pythia-1.4b` and publishes them to `<authenticated-user>/sentiment-manifold-sst-pythia-1.4b` as a private dataset repository:

- `neutral_removed`: all retained source sentences across train, validation, and test splits;
- `pythia_scored`: retained test sentences with Positive/Negative logits and correctness;
- `pythia_correct`: the complete unpaired candidate pool for target-tokenizer re-pairing;
- `matched_pairs`: maximal deterministic positive/negative matches without example reuse;
- `directed_pairs`: both clean/corrupted intervention directions for every match.

The candidate pool is intentionally uncapped. Downstream GPT-2 and Qwen experiments should re-pair `pythia_correct` with their own tokenizer because equal Pythia lengths do not imply equal lengths for another tokenizer. Pass `--hub-repo-id namespace/name` to choose a different repository or `--public` to publish publicly.

Read a private configuration using the same environment token:

```python
import os
from datasets import load_dataset

candidate_pool = load_dataset(
    "your-namespace/sentiment-manifold-sst-pythia-1.4b",
    "pythia_correct",
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

The ToyMovieReview vocabulary and prompt match the MIT-licensed reference repository in `../eliciting-latent-sentiment`. SST is read from that checkout by default without duplicating the dataset. The new package uses Hugging Face model hooks rather than the bundled historical TransformerLens fork so GPT-2 and Qwen share the same API. Exact paper-number claims require the pinned legacy GPT-2 run described in the reproduction guide.

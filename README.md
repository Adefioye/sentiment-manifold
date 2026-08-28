# Sentiment Manifold

A standalone, reusable reproduction of the essential sentiment-direction experiments from Tigges et al., *Language Models Linearly Represent Sentiment*.

The implemented baseline fits five directions—mean difference, K-means, logistic regression, PCA, and one-dimensional DAS—at every residual-stream layer of GPT-2 Small and Qwen3-0.6B. It evaluates ToyMovieReview generalization, paired causal recovery on SST, direction similarity, and OpenWebText language-model loss under directional resample ablation.

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

The default configuration fits every layer but leaves the expensive OpenWebText sweep disabled. For a smoke run, reduce `experiment.layers`, `das.epochs`, and the SST pair limit; enable OpenWebText only for the frozen confirmation run. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) and [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Reproducibility boundary

The ToyMovieReview vocabulary and prompt match the MIT-licensed reference repository in `../eliciting-latent-sentiment`. SST is read from that checkout by default without duplicating the dataset. The new package uses Hugging Face model hooks rather than the bundled historical TransformerLens fork so GPT-2 and Qwen share the same API. Exact paper-number claims require the pinned legacy GPT-2 run described in the reproduction guide.

# Sentiment-manifold working conventions

Use the project skill at [`.claude/skills/sentiment-research/SKILL.md`](.claude/skills/sentiment-research/SKILL.md) for sentiment-direction, mechanistic-interpretability, dataset, intervention, evaluation, and reproduction work.

## Tigges et al. source-of-truth rule

When the user asks a question about Tigges et al., answer it based on the paper *Language Models Linearly Represent Sentiment* and the authors' local reference implementation in [`../eliciting-latent-sentiment`](../eliciting-latent-sentiment/). Consult the relevant paper material and code before answering. Clearly distinguish claims stated in the paper from behavior found only in the code, and explicitly report any discrepancy between them rather than silently reconciling it or answering from memory.

Preserve the separation among replication, discovery, and confirmation. Do not change the ToyMovieReview or SST benchmark while claiming an exact reproduction. Fit on train data, select layers and hyperparameters on validation data, and report locked test/OOD results without reselection.

Treat probes and geometry as hypotheses, not mechanisms. Require causal intervention and appropriate controls before making representation claims. Compare nonlinear methods to dimension-matched linear baselines and preserve failed results.

Keep model-specific behavior inside `src/sentiment_manifold/models/`, fitting logic behind the common direction API, datasets free of model execution, and experiments configuration-driven. Support CUDA, MPS, and CPU; never assume a particular accelerator or Colab filesystem.

# Sentiment-direction methodology

## Tigges reproduction contract

- Fit mean difference, K-means, logistic regression, PCA, and 1D DAS on tokenizer-filtered ToyMovieReview training adjectives.
- Treat residual boundaries as `0..n_layers`: input to block 0 through residual output before final normalization.
- Orient every line from negative toward positive and save a unit vector.
- Evaluate held-out ToyMovieReview and equal-token-length binary SST counterfactuals.
- Use directional replacement `h' = h + d d^T(h_source - h)`.
- Report unclipped logit-difference recovery and source-label flip rate.
- Select each method's layer on SST validation only. Keep SST test and other OOD data locked for confirmation.
- Reproduce the paper's OpenWebText projection/GPT-4-label analysis separately. Treat LM-loss
  resample ablation as an optional exploratory diagnostic; it has no sentiment ground truth and
  must not be presented as the paper's core OpenWebText evaluation.

## Claim ladder

1. **Decodable:** a projection predicts sentiment.
2. **Localized:** a clean/corrupt intervention identifies where sentiment matters.
3. **Causally sufficient:** direction replacement restores the intended output.
4. **Necessary:** ablation damages the behavior selectively.
5. **Mechanistic:** a component/path account explains how the variable is written and read.

Do not skip levels in prose. A direction can be decodable without being used.

## Manifold extension gate

Require graded, neutral/mixed, or compositional states before fitting curvature. Compare every manifold against a dimension-matched linear subspace and a straight chord in the same subspace. Select using held-out reconstruction/support, causal recovery, specificity, seed stability, and complexity—not visual shape.

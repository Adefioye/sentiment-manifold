# Fixed-layer random-label selectivity

This experiment freezes GPT-2 Small and Qwen3-0.6B Base, extracts the final
non-padding residual-stream activation at one explicitly configured boundary,
and compares mean difference, logistic regression, one-dimensional DAS, and a
one-hidden-layer ReLU probe.

The control is an example-level, class-count-preserving label permutation. It
is not a word-type control task. Five paired seeds apply the same optimization
seed to the real and randomized tasks; the randomized side additionally gets a
new label permutation.

For mean difference, logistic regression, DAS, and the MLP output logit, the
midpoint threshold is fitted using training data only. Logistic regression and
MLP-1 also retain their native learned decision rule. DAS uses held-out raw IIA
as its native discrete accuracy and reports Tigges recovery, calibrated
logit-flip, and literal sign-flip metrics separately.

Set one residual boundary per model in
`configs/selectivity/fixed_layer.yaml`, or provide the boundaries explicitly:

```bash
sentiment-geometry fixed-layer-selectivity \
  --layer gpt2-small=YOUR_GPT2_BOUNDARY \
  --layer qwen-0.6b=YOUR_QWEN_BOUNDARY
```

The run writes immutable manifests, random-label assignments, cached
activations, probe checkpoints, per-example predictions, causal patching
records, metrics, and paired selectivity summaries. Training selectivity is the
random-label memorization diagnostic; validation and test selectivity measure
held-out behavior and should be interpreted alongside the two underlying
accuracies.

Plots are reconstructed only from the saved tables:

```bash
sentiment-geometry plot-fixed-layer-selectivity --run-dir PATH_TO_RUN
```

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

The frozen dataset-specific residual boundaries are configured in
`configs/selectivity/fixed_layer.yaml`:

| Dataset | GPT-2 Small | Qwen3-0.6B Base |
|---|---:|---:|
| ToyMovieReview | 10 | 26 |
| Full AIT | 11 | 26 |

Run the complete configuration directly:

```bash
sentiment-geometry fixed-layer-selectivity
```

Dataset-specific command-line overrides use
`--layer MODEL:DATASET=BOUNDARY` when a deliberate sensitivity run is needed.

The three tunable methods use explicit manual trial lists rather than a Cartesian
grid. Logistic regression exposes C, penalty, and class weight; DAS exposes
learning rate, weight decay, and epoch budget; MLP-1 exposes hidden width,
learning rate, and weight decay. One real-validation tuning pass per
model/dataset selects the winner, then all paired real/random seeds reuse that
configuration. Within each pair, both fits use the same optimization seed. DAS
selects primarily by validation IIA. MLP-1 fixes the random-label training
duration to the paired real-task duration.

The four Colab entry points are:

- [`12_colab_fixed_layer_mean_difference_selectivity.ipynb`](../notebooks/12_colab_fixed_layer_mean_difference_selectivity.ipynb)
- [`13_colab_fixed_layer_logistic_regression_selectivity.ipynb`](../notebooks/13_colab_fixed_layer_logistic_regression_selectivity.ipynb)
- [`14_colab_fixed_layer_das_selectivity.ipynb`](../notebooks/14_colab_fixed_layer_das_selectivity.ipynb)
- [`15_colab_fixed_layer_mlp1_selectivity.ipynb`](../notebooks/15_colab_fixed_layer_mlp1_selectivity.ipynb)

Use the same `RUN_ID` in all four. Each writes to
`RUN_ROOT/methods/<method>/<model>/`; the final cell creates
`RUN_ROOT/combined/` only after all eight method/model directories are complete.

Each run writes immutable manifests, random-label assignments, cached
activations, probe checkpoints, per-example predictions, causal patching
records, explicit tuning trials, metrics, and paired selectivity summaries. Training selectivity is the
random-label memorization diagnostic; validation and test selectivity measure
held-out behavior and should be interpreted alongside the two underlying
accuracies.

Colab progress is enabled by default. The notebooks show nested progress for
activation batches, dataset/method/seed stages, MLP-1 epochs, DAS preparation
and epochs, and final directional-patching batches. MLP-1 bars show current
training/validation loss and the best epoch; DAS bars show post-epoch training
loss, checkpoint metric, and best epoch. Set `SHOW_PROGRESS = False` in a
notebook, or `progress.enabled: false` in the YAML, for quiet execution.

After a model finishes both the real and random-label tasks, its notebook
displays real/random training memorization, all-split native and midpoint
performance, midpoint-minus-native gaps and agreement, paired selectivity with
confidence intervals, selected hyperparameters, and fit diagnostics. DAS also
shows IIA, patched/clean/corrupted accuracy, recovery, calibrated logit flip,
and literal sign flip. MLP-1 and DAS save epoch-level data to
`training_history.csv` and display training curves. Per-model diagnostic plots
are created only after that model run completes; cross-method plots still wait
for all eight method/model runs.

The layer is the boundary at which last-token activations are extracted. Mean
difference, logistic regression, and one-dimensional DAS still learn a
direction at that boundary; MLP-1 learns a nonlinear decision function and has
no single layer direction.

Plots are reconstructed only from the saved tables:

```bash
sentiment-geometry plot-fixed-layer-selectivity --run-dir PATH_TO_RUN
```

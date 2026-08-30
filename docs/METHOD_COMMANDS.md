# Per-method reproduction and optional tuning

There are two deliberately separate workflows:

1. **Paper-parity reproduction** fits a method on all ToyMovieReview training examples at every
   layer, evaluates ToyMovieReview test and SST test, and independently reports each metric's best
   layer in `best_layers.csv`. It uses the fixed reference settings in
   `configs/reproduction.yaml`; it does not tune hyperparameters.
2. **Optional validation tuning** splits only ToyMovieReview training examples into fit and
   validation subsets. It selects one layer and hyperparameter configuration without loading
   ToyMovieReview test, SST, or OpenWebText. A separate confirmation command then refits that frozen
   configuration on all ToyMovieReview training examples and evaluates the locked datasets once.

Run commands from the `sentiment-manifold` directory after installing the package with
`pip install -e '.[dev,notebooks]'`.

## Run one method with the paper-parity settings

Use a distinct output directory for each method:

```bash
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method mean_diff --output-dir outputs/by-method/mean_diff
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method kmeans --output-dir outputs/by-method/kmeans
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method logistic_regression --output-dir outputs/by-method/logistic_regression
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method pca --output-dir outputs/by-method/pca
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method das --output-dir outputs/by-method/das
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method das2d --output-dir outputs/by-method/das2d
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method das3d --output-dir outputs/by-method/das3d
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --method random --output-dir outputs/by-method/random
```

Each directory contains a `gpt2-small/best_layers.csv`. These commands still scan every layer. The
`--method` option limits fitting and evaluation to that method; it does not lock or preselect a
layer. Repeat `--method` in one command if several methods should share a combined output table.

For a fixed, manually chosen configuration, command-line overrides are available. These are useful
for diagnostics, but a value chosen after looking at test results is not a clean tuned result:

```bash
sentiment-manifold reproduce --config configs/reproduction.yaml --method kmeans --kmeans-n-init 50 --output-dir outputs/manual/kmeans-n50
sentiment-manifold reproduce --config configs/reproduction.yaml --method logistic_regression --logistic-c 0.1 --output-dir outputs/manual/logistic-c0.1
sentiment-manifold reproduce --config configs/reproduction.yaml --method das --das-learning-rate 0.0003 --das-epochs 32 --output-dir outputs/manual/das-lr3e-4-e32
```

Changing one of these options invalidates an incompatible saved direction, so `resume: true` cannot
silently reuse an artifact fitted with older hyperparameters.

## Optional validation-only tuning

The default grid in `configs/tuning.yaml` checks all GPT-2 Small layer boundaries. Mean difference
and PCA have no fitted hyperparameters, so their tuning commands select only a layer. K-means checks
`n_init`; logistic regression checks `C`; each DAS dimension checks learning rate and epoch count.
Stochastic methods are compared by the mean validation score across three seeds.

```bash
sentiment-manifold tune --config configs/tuning.yaml --method mean_diff
sentiment-manifold tune --config configs/tuning.yaml --method kmeans
sentiment-manifold tune --config configs/tuning.yaml --method logistic_regression
sentiment-manifold tune --config configs/tuning.yaml --method pca
sentiment-manifold tune --config configs/tuning.yaml --method das
sentiment-manifold tune --config configs/tuning.yaml --method das2d
sentiment-manifold tune --config configs/tuning.yaml --method das3d
```

DAS tuning is expensive: with the supplied grid, one DAS dimension requires 13 layers × 4
configurations × 3 seeds = 156 fits. Edit only the lists under `tuning:` to define a justified
budget before running. A temporary `--layer 0 --layer 6 --layer 12` scout is supported, but its
selection is only over those three boundaries and should be described that way.

Each tuning run writes to `outputs/tuning/gpt2-small/<method>/`:

- `tuning_split.csv` records the fixed fit/validation membership;
- `tuning_trials.csv` contains every layer, configuration, seed, and validation metric;
- `selected_configs.csv` contains the configuration selected by mean validation performance;
- `trial_directions/`, per-pair patching records, and DAS loss histories make selection auditable.

The default selection outcome is Toy validation logit-difference recovery. To select by the
paper-style logit-flip score instead, set:

```yaml
tuning:
  selection_metric: toy_validation_logit_flip_percent
```

Do not run a second selection with a different metric after inspecting final test results; choose
the validation criterion before confirmation.

## Refit and evaluate a frozen selection

Confirmation reads exactly one row, refits that method at the selected layer on all ToyMovieReview
training examples, then evaluates ToyMovieReview test and SST test. Give it a new output directory:

```bash
sentiment-manifold confirm --config configs/reproduction.yaml --selection outputs/tuning/gpt2-small/kmeans/selected_configs.csv --output-dir outputs/confirmed/kmeans
sentiment-manifold confirm --config configs/reproduction.yaml --selection outputs/tuning/gpt2-small/logistic_regression/selected_configs.csv --output-dir outputs/confirmed/logistic_regression
sentiment-manifold confirm --config configs/reproduction.yaml --selection outputs/tuning/gpt2-small/das/selected_configs.csv --output-dir outputs/confirmed/das
sentiment-manifold confirm --config configs/reproduction.yaml --selection outputs/tuning/gpt2-small/das2d/selected_configs.csv --output-dir outputs/confirmed/das2d
sentiment-manifold confirm --config configs/reproduction.yaml --selection outputs/tuning/gpt2-small/das3d/selected_configs.csv --output-dir outputs/confirmed/das3d
```

The confirmation `best_layers.csv` has four paper metrics but only the frozen layer, so it reports
four evaluations of one validation-selected direction. This differs intentionally from the strict
paper-parity table, which independently takes the best test-set layer for every dataset/metric cell.
`confirmation_selection.csv` and `confirmation_provenance.json` preserve the exact selected row and
its source path. A selection made for one model cannot be confirmed on another model. OpenWebText
resample ablation remains opt-in and should not be enabled during model selection.

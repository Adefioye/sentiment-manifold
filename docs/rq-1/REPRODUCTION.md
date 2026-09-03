# RQ1 reproduction guide

## Stages

1. Run `inspect-data` for each tokenizer and record how many of the paper's adjectives remain one token.
2. Preprocess both SST binary-label variants with Pythia-1.4B. The reproduction runner consumes the
   `tigges_pythia_correct` test configuration.
3. Run a small layer subset on CPU/MPS/CUDA and confirm artifacts and metrics are finite.
4. Run all GPT-2 Small boundaries with OpenWebText disabled. For the paper-style Table 1 report,
   independently take the maximum across layers for each method in all four dataset/metric columns.
5. Compare modern GPT-2 results with paper plots. Investigate hook, normalization, prompt, token,
   or pairing discrepancies before changing methodology.
6. Reproduce the paper's first-layer OpenWebText projection analysis separately. Run resample
   ablation only as an explicitly labelled exploratory diagnostic.
7. Repeat the same protocol for Qwen3-0.6B; describe it as an extension, not part of the original paper replication.

The paper specifies 55 training adjectives, 30 test adjectives, and eight verbs. The current upstream
`prompts.yaml` has lost one historical training adjective and later expanded the verb lists. Repository
history identifies the paper-era inventory: `extraordinary` is the 55th training adjective, and the
eight verbs are `enjoyed`, `loved`, `liked`, `appreciated`, `admired`, `hated`, `disliked`, and
`despised`. This project uses that paper-intended inventory rather than the later expanded lists.

## Commands

```bash
sentiment-manifold inspect-data --config configs/reproduction.yaml
sentiment-manifold preprocess-sst --binarization both
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --device auto
sentiment-manifold reproduce --config configs/reproduction.yaml --model qwen-0.6b --device auto
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --with-openwebtext-resample-ablation
sentiment-manifold plot --run-dir outputs/results/gpt2-small
```

For separate commands for mean difference, K-means, logistic regression, PCA, DAS 1D/2D/3D, and
the random control—and for the optional Toy-training-only tuning workflow—see
[Per-method reproduction and optional tuning](METHOD_COMMANDS.md).

The last command is optional and exploratory; it is not the paper's core OpenWebText evaluation. For a smoke run, replace `experiment.layers: all` with a short list such as `[0, 6, 12]`, set `das.epochs: 1`, and limit SST pairs/OpenWebText samples. These changes are debugging settings and must not be reported as the final reproduction.

## Colab and Google Drive

Install the project editable in Colab and call `maybe_mount_google_drive(True)` only when persistent
checkpoints are wanted. Keep `experiment.output_dir: outputs/results` so CSVs, JSON metadata, and
plots remain on the Colab instance. Set `experiment.checkpoint_dir`, pass `--checkpoint-dir`, or set
`SENTIMENT_MANIFOLD_CHECKPOINT_DIR` to
`/content/drive/MyDrive/sentiment-manifold/checkpoints`. The run notebook performs this split when
`USE_GOOGLE_DRIVE = True`.

`SENTIMENT_MANIFOLD_OUTPUT_DIR` controls the lightweight result root only. Pointing that variable at
Drive still moves all results to Drive, so it should normally remain unset in Colab. The automatic
layout is model-scoped:

```text
outputs/results/<model>/
checkpoints/<model>/<phase>/<method>/<configuration-fingerprint>/
```

The configuration fingerprint includes the resolved model/tokenizer revisions, exact Toy training
prompts, and fitting settings. Consequently, another model or another tuned configuration cannot
overwrite an existing checkpoint.

## Numerical comparison checklist

Record the exact model and tokenizer revision, dependency lock, dtype, accelerator, seed, prompt bytes, retained vocabulary, answer token IDs, paired SST IDs, layer-boundary definition, direction orientation, DAS losses, and per-pair metrics. Recovery remains unclipped; values above 100% are possible and informative.

For Table 1-style reporting, use the `*_logit_diff_percent` and `*_logit_flip_percent` columns in
`metrics.csv`; the latter matches the reference code's centered, baseline-calibrated accuracy score.
The corresponding `*_sign_flip_percent` column is the literal pre/post sign-change rate and should be
identical on a paper-parity dataset. Normalized columns remain available as diagnostics. Per-pair
raw clean/corrupted/patched logit differences and sign-flip indicators are recorded in
`patching_records.csv` for auditing.

`best_layers.csv` applies that reporting rule independently. It contains four rows per method—one
for each ToyMovieReview/SST × logit-difference/logit-flip cell—with `layer` and `value_percent`.
Consequently, its rows are paper-table maxima, not a claim that one locked layer transfers between
datasets and metrics. Tied maxima use the lower layer boundary deterministically.

## Saved run artifacts

Every completed experiment writes analysis-ready artifacts under the configured result directory.
Files are updated incrementally while a run proceeds, so an interrupted run still retains completed
cells:

- `resolved_config.json`: requested configuration plus resolved model/tokenizer revisions, BOS and
  padding settings, device, dtype, and library versions;
- `prompt_manifest.csv`: exact ToyMovieReview text, Python representation, UTF-8 bytes, SHA-256,
  token IDs, token count, and adjective focus position;
- `toy_vocabulary.csv`: every source adjective/verb, tokenizer IDs, and one-token retention decision;
- `answer_tokens.csv`: Tigges's five aligned ToyMovieReview answer pairs and SST classification labels;
- `sst_candidate_manifest.csv`: every Pythia-correct SST candidate with source text, exact
  classification prompt bytes, labels, scores, and Pythia/GPT-2 token lengths;
- `pair_manifest.csv`: exact Toy train/test and SST clean/corrupted IDs, prompt hashes, token IDs,
  labels, and equal-length checks;
- `direction_metadata.csv`: artifact path, dimensionality, and orientation convention;
- `das_losses.csv`: one friendly row per method/layer/epoch with train and evaluation loss (present
  when a DAS method is run);
- `patching_records.csv`: per-pair clean, corrupted, and patched logit differences, margins,
  recovery, and literal flip indicator;
- `metrics.csv`: per-method/layer aggregate projection and causal metrics;
- `direction_similarities.csv`: one-dimensional absolute cosine similarities;
- `best_layers.csv`: independently maximized layer and value for every method and each of the four
  Table 1 dataset/metric columns;
- `openwebtext_controls.csv`: optional random controls when the exploratory resample ablation runs.

Normalized 1D vectors and orthonormal DAS subspace bases are saved separately under the configured
checkpoint root. `direction_metadata.csv` records their exact paths. Validation tuning similarly
stores trial direction checkpoints outside its result directory.

`sentiment-manifold plot` consumes these saved CSVs and writes the causal layer curves, a rendered
Table 1-style best-result table, DAS loss curves, and direction-similarity heatmaps under `figures/`;
it does not rerun an experiment.

## Reference-code parity now enforced

- the GPT-2 model/tokenizer commit is pinned and its resolved revisions are saved;
- tokenization explicitly prepends BOS, matching TransformerLens `prepend_bos=True`;
- ToyMovieReview uses the paper's 55/30 adjective split, eight paper-era verbs, reference prompt
  bytes, and five positive/negative answers;
- tokenizer filtering is recorded rather than silently changing the prompt vocabulary;
- clean/corrupted terminology and the reference cyclic prompt shift are used;
- directional patching edits every token position (`placeholders = ['ALL']` upstream);
- K-means uses `n_init=10`, logistic regression uses `max_iter=1000`, and PCA's arbitrary raw sign is
  recorded before the exported axis is oriented for stable reporting;
- DAS uses an orthogonally parameterized rotation, normalized logit-difference objective, 64 epochs,
  GPT-2 Small batch size 128, and configured 1D/2D/3D variants;
- the random direction control follows the upstream seed-42, layer-indexed sequence.
- SST uses the test split, `Review Text: … Review Sentiment:`, `Positive`/`Negative` answers,
  Tigges's binary collapse, and only Pythia-1.4B-correct candidates; the runner re-pairs that saved
  pool at equal GPT-2 token length before all-position patching.

Known parity boundaries remain and must not be hidden in reporting:

- the runner uses Hugging Face hooks rather than TransformerLens;
- the current upstream `SIMPLE_TRAIN` implementation computes its prompt count from the smaller core
  lists before loading the paper-era training lists; this code-path anomaly is not followed because
  it conflicts with the paper's stated dataset;
- SST equal-length pairs are reconstructed deterministically from the saved Pythia-correct pool
  using the target tokenizer instead of loading an upstream shuffled pickle, so pair identities can
  differ even when the task definition and aggregate metric agree.

These are why hook boundary, prompt/token manifest, paired IDs, and metric comparisons remain
required before claiming exact numerical reproduction. The output artifacts make each difference
auditable rather than silently changing the paper methodology.

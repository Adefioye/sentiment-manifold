# RQ2 — Valence/sentiment alignment and OOD transfer

## Research question

How well do valence directions learned from AIT V-oc align with sentiment directions, and how
causally useful are those directions on SST, IMDb, DynaSent, and CEBaB across model families?

Use Difference-in-means, logistic regression and DAS fitting methods for learning `sentiment direction` and `valence direction`.

- [x] **1.** For a start, fit the three methods on ToyMovieReview to learn sentiment directions using `gpt2-small` and `qwen-0.6b`; compare direction similarity at the first, middle, and last layers; and measure logit-difference recovery and logit-flip percent by patching all token positions. Fit separate directions from activations at the adjective (ADJ), verb (VRB), second `movie` (SUM), and final `is` (END) positions, then compare their in-distribution ToyMovieReview and out-of-distribution SST performance.

- [ ] **2.** Secondly, we would then train the 3 methods to learn sentiment and valence direction using ToyMovieReview and AIT respectively for all 4 models. Here, we use about same amount of datasets for training both directions. What this means is, roughly about 55 data samples for both ToyMovieReview and AIT. We observe the effect on logit difference and logit flip percent. The method of extracting activations for learning sentiment direction depends on which performs best from 1 above. For valence direction, activation extraction should involve all activations of all tokens per data sample and then averaging them to train the fitting methods.

- [ ] **3.** Thirdly, we can then train on large amount of the data samples on AIT and observe if there is discernible performance on eval datasets like ToyMovieReview and SST using metrics we have been using above.

- [ ] **4.** Run first, middle, and last layer within-model geometric alignment for sentiment directions and secondly the same layer within-model geometric alignments for sentiment and valence directions.

## Data contract

- AIT V-oc is the supervised valence-direction training dataset. Its gold ordinal labels are
  collapsed to binary polarity for pairing, but examples are not correctness-filtered by a model.
- SST, IMDb, and DynaSent are zero-shot evaluation datasets. Pythia-2.8B must agree with each
  dataset's gold binary label before an example can enter an equal-length patching pair.
- CEBaB is a human-counterfactual OOD evaluation dataset. It uses the authors' majority rating,
  keeps original/edit families grouped, and defers correctness selection to each evaluated model.
- Gold labels are never replaced by Pythia predictions.
- Model-specific pairs have equal full-prompt length under the selected tokenizer. Common pairs
  have equal length under all four configured tokenizers.

## Supported pairing tokenizers

- GPT-2 Small
- Qwen3-0.6B Base
- Gemma 2B
- Pythia 1.4B

Pythia-2.8B is the default **selection model**, not an additional pairing target.

## Pairing prompt-length policy

GPT-2 Small has the shortest configured context window: 1,024 tokens, including the explicitly
prepended BOS token. Context limits apply to each complete
`Review Text: {text} Review Sentiment:` prompt, not to the dataset as a whole.

RQ2 now enforces `max_pairing_prompt_tokens = 1000`: every model-specific matched or directed
prompt must contain at most 1,000 tokens under its pairing tokenizer, including BOS when that
tokenizer prepends one. Common pairs must satisfy the same limit under all selected tokenizers.
The binary, scored, correct, and pairing-candidate configurations retain excluded rows for
provenance; only examples used in matched/directed intervention pairs are constrained.

An audit of the published IMDb artifact found:

- 945 of the 50,000 binary prompts exceed 1,024 GPT-2 tokens: 493 train and 452 test.
- Of the 20,533 Pythia-correct test pairing candidates, 408 have more than 1,000 GPT-2 prompt
  tokens and are ineligible for GPT-2 pairing under the active limit.
- Before this policy, `gpt2_small_matched_pairs` contained 7,645 matches and 15,290 directed rows.
  The 1,000-token cap removes eight matches and 16 directed rows, leaving 7,637 and 15,274.

The private IMDb Hub artifact was updated at revision
`06586d20342aa46fa61525b1f7609ab065983e3e` with these final pair populations:

| Pairing tokenizer | Matched before | Matched after | Directed before | Directed after | Maximum after |
|---|---:|---:|---:|---:|---:|
| GPT-2 Small | 7,645 | 7,637 | 15,290 | 15,274 | 990 |
| Qwen 0.6B | 7,676 | 7,609 | 15,352 | 15,218 | 992 |
| Gemma 2B | 7,671 | 7,608 | 15,342 | 15,216 | 996 |
| Pythia 1.4B | 7,630 | 7,558 | 15,260 | 15,116 | 994 |
| Common | 243 | 243 | 486 | 486 | 520 across its four tokenizers |

The compressed Hub repository decreased from 443,453,737 bytes (422.910 MiB) to 435,374,512
bytes (415.205 MiB), a reduction of 8,079,225 bytes (7.705 MiB, 1.82%). Binary, scored,
correct, and pairing-candidate rows did not change.

The threshold uses full-prompt token counts, not raw-review length. It is applied before pair
construction and recorded in preprocessing metadata. Freeze the resulting population before
confirmation runs rather than changing the limit after inspecting evaluation results.

## Guides

- [Preprocessing guide](PREPROCESSING.md): dataset commands, correctness filtering, output
  configurations, provenance, publishing, and cross-model pairing policy.
- [Colab notebook](../../notebooks/02_colab_preprocess_publish_explore_rq2.ipynb): end-to-end
  preprocessing, private Hub publication, reload checks, and descriptive data exploration.
- [RQ1 reproduction](../rq-1/README.md): the separate Tigges Table 1 protocol, which retains its
  original Pythia-1.4B SST correctness filter.

## Status

Preprocessing and tokenizer-aware pair construction are implemented. Direction fitting and causal
evaluation on these RQ2 artifacts should preserve train/validation/test separation and freeze the
chosen pair policy before confirmation runs.

## Question 1 experiment

Question 1 uses the domain-named sentiment-position comparison API so it does not alter the RQ1
reproduction protocol. It fits mean difference, logistic regression, and one-dimensional DAS
independently at four prompt-token positions: the adjective (ADJ), verb (VRB), second `movie`
(SUM), and final `is` token (END). ADJ, VRB, and SUM are resolved from character spans through
each model's tokenizer rather than assumed token indices; END is the last non-padding token. It
sweeps residual boundaries
`1..n_layers`; boundary `0` is the embedding residual and is excluded. The full ADVERB panel is
used as validation data for both DAS epoch selection and residual-boundary selection. The selected
boundary maximizes ADVERB `logit_flip_percent` independently for each model × method ×
fitting-position direction. That boundary is frozen before evaluation on held-out ToyMovieReview
adjectives and the model-specific SST `directed_pairs` configuration.

ADVERB validation uses the upstream SimpleAdverb vocabulary, deduplicated and subject to its exact
two-token filter. Every retained ADVERB case is reused for both checkpoint and layer selection; it
is not presented as an unbiased final result. ADJ and SST never select their own best layers.

The active Toy panels are selected by `data.toy_evaluations`. The current configuration selects
`toy_adjectives` and `toy_adverbs`; `toy_verbs` remains supported and can be restored through the
configuration. A run fails if a selected panel cannot produce tokenizer-compatible, equal-length
directional cases.

The complete configuration is in
[`configs/sentiment_position_comparison.yaml`](../../configs/sentiment_position_comparison.yaml).
The step-by-step
[`Colab sentiment-position notebook`](../../notebooks/03_colab_sentiment_position_comparison.ipynb)
runs both pinned models and writes every direction, result table, manifest, and figure directly to
a collision-safe, minute-stamped Google Drive directory. Its artifact audit expects 144 GPT-2 Small
directions and 336 Qwen3-0.6B directions.
The read-only
[`sentiment-position exploration notebook`](../../notebooks/04_colab_explore_sentiment_position_results.ipynb)
displays frozen-layer metric tables, cosine similarities, and model-by-position layer curves
from a completed Drive run without saving additional analysis artifacts.
The same workflow is available as a regular Python API:

```python
from sentiment_geometry.experiments import (
    SentimentPositionExperiment,
    SentimentPositionExperimentConfig,
)

config = SentimentPositionExperimentConfig.load(
    "configs/sentiment_position_comparison.yaml"
)
SentimentPositionExperiment(config).run()
```

Run the explicit, full experiment command with:

```bash
./scripts/run_sentiment_position_comparison.sh
```

The script passes both models, all three methods, all four fitting positions, the complete non-embedding
layer sweep, fitting hyperparameters, SST repository, authentication environment variable, output
directory, and checkpoint directory explicitly. `HF_TOKEN` or `HF_TOKEN_PATH` must provide access
to the private SST repository. Checkpoints are resumable.

Each model directory saves `resolved_config.json`, `dataset_summary.csv`, `prompt_manifest.csv`,
`pair_manifest.csv`, `toy_vocabulary.csv`, `answer_tokens.csv`, `direction_metadata.csv`,
`das_epoch_metrics.csv`, `patching_records.csv`, `metrics.csv`, `selected_metrics.csv`,
`direction_similarities.csv`, and `layer_selection.csv`. `das_epoch_metrics.csv` contains the
post-epoch training objective and genuine ADVERB validation metrics; `layer_selection.csv` records
the single ADVERB-logit-flip-selected boundary for every model × method × fitting-position cell.
`selected_metrics.csv` reports all three causal metrics for ADVERB, ADJ, and SST after rerunning each
dataset at that frozen boundary. Its ADVERB rows are post-selection evaluations, not reused
layer-sweep rows. Absolute cosine is the
primary similarity at boundaries `[1, floor(n_layers/2), n_layers]`; signed cosine is retained for
audit. For Toy prompts, `prompt_manifest.csv` records `adjective_position`, `verb_position`,
`summary_position`, and `final_position`, making the tokenizer-resolved ADJ/VRB/SUM/END indices
directly inspectable.

Plots are recreated only from saved tables:

```bash
sentiment-geometry plot --run-dir outputs/sentiment-position-comparison/gpt2-small
sentiment-geometry plot --run-dir outputs/sentiment-position-comparison/qwen-0.6b
```

The implementation is complete and smoke-tested. The checkmark records implementation completion;
the full GPU experiment and its scientific interpretation are still pending.

## AIT valence-direction experiment

The AIT workflow has its own configuration at
[`configs/ait_valence_directions.yaml`](../../configs/ait_valence_directions.yaml) and a public
`AITValenceDirectionExperiment` API. It loads the pinned private AIT Hub artifact's
`common_matched_pairs` configuration so all configured models use identical, equal-length prompt
pairs. The deterministic sample contract is 55 train examples, 30 eval directed cases, and 30 test
directed cases.

Mean difference and logistic regression fit masked mean-pooled residual activations over all
non-padding, non-special prompt tokens. One-dimensional DAS retains causal training semantics: it
uses the directed train pairs and patches all non-padding token positions. At every layer, the DAS
epoch is selected by eval-set validation loss. The disjoint AIT test role then selects one layer per
method using `logit_flip_percent`; because it performs selection, it is not treated as an unbiased
final evaluation set.

```python
from sentiment_geometry.experiments import (
    AITValenceDirectionExperiment,
    AITValenceExperimentConfig,
)

config = AITValenceExperimentConfig.load("configs/ait_valence_directions.yaml")
AITValenceDirectionExperiment(config).run()
```

The equivalent CLI is:

```bash
sentiment-geometry train-ait-valence --config configs/ait_valence_directions.yaml
```

Private Hub authentication is read from `HF_TOKEN` by default, or from the file named by
`HF_TOKEN_PATH`. A different environment-variable name can be declared in the YAML or passed with
`--hf-token-env`; the credential itself is never written to a config, manifest, CSV, or checkpoint.

The output root contains immutable sample/pair manifests, a dataset summary, requested and resolved
configuration, combined CSVs, and an experiment manifest. Each model directory contains resumable
direction checkpoints plus `metrics.csv`, `patching_records.csv`, `direction_metadata.csv`,
`das_epoch_metrics.csv`, `layer_selection.csv`, and `selected_metrics.csv`. A future Colab notebook
should only set runtime paths and credentials, invoke this API, and display the saved tables.

# RQ2 preprocessing guide

This workflow prepares AIT V-oc, SST, IMDb, and DynaSent R1/R2 for valence-versus-sentiment
geometry and causal-transfer experiments.

## Correctness-filter policy

SST, IMDb, and DynaSent use `EleutherAI/pythia-2.8b` for zero-shot correctness selection by
default. Every selected record is prompted as:

```text
Review Text: {text} Review Sentiment:
```

The next-token logits for ` Positive` and ` Negative` determine the prediction. An example enters
the pairing pool only when that prediction agrees with its gold binary label. The scored and
correct subsets are saved separately for auditing; Pythia predictions never replace gold labels.

AIT is deliberately different: it supplies the gold-labelled training data used to learn a
valence direction, so it is not filtered according to a model's existing ability.

The shared filtering options for SST, IMDb, and DynaSent are:

```bash
--filter-model pythia-2.8b \
--filter-revision REVISION \
--device auto \
--dtype auto \
--batch-size 16
```

Pin `--filter-revision` for reportable runs. Pythia-1.4B remains available only to preserve the
separate Tigges RQ1 SST reproduction:

```bash
sentiment-manifold preprocess-sst \
  --filter-model pythia-1.4b \
  --output-dir data/processed/sst-pythia-1.4b
```

## Installation and authentication

```bash
cd sentiment-manifold
pip install -e '.[dev,notebooks]'
```

The commands save locally unless `--push-to-hub` is supplied. Uploads are private by default;
`--public` is the only public opt-in. Supply credentials through `HF_TOKEN` or `HF_TOKEN_PATH` and
never place a token in a notebook, tracked file, command argument, or output log.

Private Hub access does not grant redistribution permission. Check each source's license and terms,
especially for tweets, IMDb reviews, and DynaSent Round 1 Yelp-derived text.

The [Colab preprocessing and exploration notebook](../../notebooks/02_colab_preprocess_publish_explore_rq2.ipynb)
runs all four pipelines, verifies that requested Hub repositories are private, and produces
text-free CSV summaries. Because AIT V-oc restricts redistribution, its Hub upload is guarded by
an explicit permission acknowledgement and remains local by default.

The notebook is divided into two independent parts. Part I performs Colab preprocessing and
publishing. Part II can be run by itself on a local computer: it imports its own dependencies,
authenticates to the private Hub repositories, pins their current commits, and loads Parquet files
from one configuration directory at a time. It does not require `/content`, Google Drive, a project
checkout, or any source-data repository. This directory-isolated loader also remains compatible
with repositories published before configuration directories were declared in dataset-card YAML.

Newly published repositories map every configuration to its own `data_dir` and include a root
`metadata.json` containing preprocessing provenance. This prevents schemas such as `binary` and
`common_directed_pairs` from being combined by the generic Parquet loader.

## Pairing tokenizers

The default is all four:

- `gpt2-small` → `gpt2`
- `qwen-0.6b` → `Qwen/Qwen3-0.6B-Base`
- `gemma-2b` → `google/gemma-2b`
- `pythia-1.4b` → `EleutherAI/pythia-1.4b`

Select a subset by repeating `--pairing-model`:

```bash
--pairing-model gpt2-small --pairing-model gemma-2b
```

Pin tokenizer versions separately from the correctness-filter model:

```bash
--pairing-revision gpt2-small=REVISION \
--pairing-revision gemma-2b=REVISION
```

`common_matched_pairs` and `common_directed_pairs` require equal full-prompt length under every
selected tokenizer. Source text is never truncated. Over-context rows are marked and excluded from
the relevant model's pairs.

Matched and directed pair construction also enforces a full-prompt cap of 1,000 tokens by default,
including an explicitly prepended BOS token. Override it only as an explicit preprocessing choice:

```bash
--max-pairing-prompt-tokens 1000
```

The limit applies under the selected tokenizer for model-specific pairs and under every selected
tokenizer for common pairs. Base, scored, correct, and `pairing_candidates` configurations retain
longer rows for provenance; those rows cannot enter matched or directed pairs.

## AIT V-oc

Use the English train/dev/test gold files from SemEval-2018 Task 1 Affect in Tweets. The official
files have a `.txt` extension but contain four **tab-delimited** columns: `ID`, `Tweet`,
`Affect Dimension`, and `Intensity Class`. Keep the tabs intact; a space-aligned copy from a
document preview is not equivalent to the source file.

```bash
sentiment-manifold preprocess-ait \
  --ait-root data/raw/ait \
  --output-dir data/processed/ait-valence-binary
```

When discovery is ambiguous, provide all three paths explicitly:

```bash
sentiment-manifold preprocess-ait \
  --train-file data/raw/ait/2018-Valence-oc-En-train.txt \
  --validation-file data/raw/ait/2018-Valence-oc-En-dev.txt \
  --test-file data/raw/ait/2018-Valence-oc-En-test-gold.txt
```

The binary policy is `-3,-2,-1 → negative`, `0 → excluded`, and `1,2,3 → positive`. The retained
`original_valence_class` is ordinal, not an interval-scale continuous score. All three splits are
paired by default, but training, selection, and final testing must remain separate. Tweet text is
retained verbatim—including emojis, mentions, hashtags, punctuation, and strings such as
`&amp;`—so source provenance and tokenizer-length calculations are not silently changed.

## SST

```bash
sentiment-manifold preprocess-sst \
  --sst-root ../eliciting-latent-sentiment/stanfordSentimentTreebank \
  --binarization both \
  --output-dir data/processed/sst-pythia-2.8b
```

The default RQ2 artifact is filtered with Pythia-2.8B. The `tigges` label policy maps scores
`<= 0.5` to negative and scores `> 0.5` to positive. The `neutral_removed` alternative maps scores
`<= 0.4` to negative, removes `(0.4, 0.6]`, and maps scores `> 0.6` to positive. Only test is scored
and paired.

## IMDb

```bash
sentiment-manifold preprocess-imdb \
  --dataset-name stanfordnlp/imdb \
  --output-dir data/processed/imdb-pythia-2.8b
```

The source text and gold labels are retained. Only test is Pythia-filtered and paired by default.
Add `--pairing-split train` only when a training-pair artifact is required. Pin the source snapshot
with `--dataset-revision REVISION`.

## DynaSent R1/R2

Download and extract `dynasent-v1.1.zip` from the official DynaSent repository, then run:

```bash
sentiment-manifold preprocess-dynasent \
  --dynasent-root data/raw/dynasent \
  --output-dir data/processed/dynasent-r1-r2-pythia-2.8b
```

Both rounds are processed but remain separate. Gold positive/negative examples are retained;
neutral, mixed, and no-majority records are excluded before correctness scoring. Only test is
Pythia-filtered and paired by default.

## Output configurations

SST writes each configuration below with a `tigges_` or `neutral_removed_` prefix:

- `binarized`
- `pythia_scored`
- `pythia_correct`
- `matched_pairs` and `directed_pairs` for the filter tokenizer
- `pairing_candidates`
- `<model>_matched_pairs` and `<model>_directed_pairs`
- `common_matched_pairs` and `common_directed_pairs`

IMDb writes `binary`, `pythia_scored`, `pythia_correct`, and the generic pairing configurations.
DynaSent writes the same family with `r1_` or `r2_` prefixes. AIT writes `binary` and generic
pairing configurations only; it has no `pythia_*` configurations.

Directional rows use `source_*` for the activation donor carrying the desired label and `target_*`
for the receiver before patching. Thus `negative_to_positive` has a positive source and negative
target.

## Provenance and comparison policy

Each output records source checksums, gold-label policy, prompt template, filter-model revision,
filter counts, tokenizer revisions, BOS policy, context lengths, and pair counts in `metadata.json`.

Before causal evaluation, freeze whether comparisons use each model's maximal pair set or the
smaller common set. Use the common set when flip rates must be directly comparable across models.
Always report the number scored, retained, and paired; correctness selection changes the evaluated
population.

To publish privately:

```bash
sentiment-manifold preprocess-imdb --push-to-hub --private
```

Set `--hub-repo-id ACCOUNT/REPOSITORY` to override the generated repository name.

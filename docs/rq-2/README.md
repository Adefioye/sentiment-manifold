# RQ2 data preprocessing: valence/sentiment alignment and OOD transfer

This workflow prepares binary AIT, SST, IMDb, and DynaSent R1/R2 records for the planned
valence-versus-sentiment geometry and causal-transfer experiments. It creates model-specific
equal-length prompt pairs for:

- `gpt2-small` → `gpt2`
- `qwen-0.6b` → `Qwen/Qwen3-0.6B-Base`
- `gemma-2b` → `google/gemma-2b`
- `pythia-1.4b` → `EleutherAI/pythia-1.4b`

The default is all four tokenizers. `common_matched_pairs` and `common_directed_pairs` require the
full positive and negative prompts to have equal length under all selected tokenizers. Source text
is never truncated: a row that exceeds a model's context window is marked `*_fits_context=false`
and is excluded from that model's pairs.

## Install and authenticate securely

```bash
cd sentiment-manifold
pip install -e '.[dev,notebooks]'
```

Gemma requires accepting its Hugging Face access terms. In Colab, save a secret named `HF_TOKEN`
with notebook access enabled, then load it without printing it:

```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

The preprocessing commands publish privately by default. `--public` is the only public opt-in.
You can also set `HF_TOKEN_PATH` to a mounted secret file. Never put a token in a notebook cell,
shell history, Git-tracked file, command argument, or output log.

Private Hub access does **not** grant redistribution permission. Check each source's license and
terms before uploading derived text. This is especially important for tweets, IMDb reviews, and
DynaSent Round 1 Yelp-derived text.

## Select tokenizers

Omit `--pairing-model` to process all four. To process only a subset, repeat the option:

```bash
--pairing-model gpt2-small --pairing-model gemma-2b
```

For immutable provenance, pin any tokenizer revision with repeatable values:

```bash
--pairing-revision gpt2-small=REVISION \
--pairing-revision gemma-2b=REVISION
```

The resolved tokenizer commit, BOS policy, vocabulary size, context length, source checksums, label
policy, prompt template, and pair counts are written to `metadata.json`.

## AIT Valence-oc (binary training data)

Use the English V-oc train/dev/test gold files from
[SemEval-2018 Task 1: Affect in Tweets](https://www.saifmohammad.com/WebPages/affectintweets.htm).
Place them under one directory and run:

```bash
sentiment-manifold preprocess-ait \
  --ait-root data/raw/ait \
  --output-dir data/processed/ait-valence-binary \
  --push-to-hub --private
```

If discovery is ambiguous, supply all three explicitly:

```bash
sentiment-manifold preprocess-ait \
  --train-file data/raw/ait/2018-Valence-oc-En-train.txt \
  --validation-file data/raw/ait/2018-Valence-oc-En-dev.txt \
  --test-file data/raw/ait/2018-Valence-oc-En-test-gold.txt
```

Binary policy: `-3,-2,-1 → negative`, `0 → excluded`, and `1,2,3 → positive`. The original V-oc
class is retained as `original_valence_class`; it is ordinal and must not be described as a
continuous interval-scale score. AIT is intended to fit the valence direction. By default, all
three splits receive pair configurations, but training, model selection, and final testing must
still remain separate.

## SST

This command retains the RQ1 Pythia-1.4B correctness filter and legacy pair artifacts, then adds
RQ2 tokenizer-specific and common pairs from the correctly classified candidates:

```bash
sentiment-manifold preprocess-sst \
  --sst-root ../eliciting-latent-sentiment/stanfordSentimentTreebank \
  --binarization both \
  --output-dir data/processed/sst-pythia-1.4b \
  --push-to-hub --private
```

`--revision` pins the Pythia model used for the RQ1 correctness filter. Use
`--pairing-revision MODEL=REVISION` separately for tokenizer provenance. The generic RQ2 pairs are
not a replacement for the paper-oriented legacy `*_matched_pairs` configs.

## IMDb

IMDb downloads through the Hugging Face `stanfordnlp/imdb` dataset. Text is retained verbatim and
only its labeled train/test splits are kept:

```bash
sentiment-manifold preprocess-imdb \
  --dataset-name stanfordnlp/imdb \
  --output-dir data/processed/imdb-binary \
  --push-to-hub --private
```

Only the test split is paired by default. Add `--pairing-split train` if a training-pair artifact is
needed. Pin the dataset snapshot with `--dataset-revision REVISION` for a frozen run.

## DynaSent R1/R2

Download `dynasent-v1.1.zip` from the
[official DynaSent repository](https://github.com/cgpotts/dynasent), then extract it:

```bash
mkdir -p data/raw/dynasent
curl -L https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip \
  -o data/raw/dynasent/dynasent-v1.1.zip
unzip data/raw/dynasent/dynasent-v1.1.zip -d data/raw/dynasent

sentiment-manifold preprocess-dynasent \
  --dynasent-root data/raw/dynasent \
  --output-dir data/processed/dynasent-r1-r2-binary \
  --push-to-hub --private
```

Both rounds are processed by default; repeat `--round` to choose. Gold positive/negative examples
are retained, while neutral, mixed, and no-majority records are excluded. Only test is paired by
default. R1 and R2 retain separate configs so their different collection processes are not pooled.

## Output configurations

Each non-SST dataset writes:

- `binary`: normalized binary rows without prompts;
- `pairing_candidates`: prompts plus every selected tokenizer's raw/prompt lengths and context flag;
- `<model>_matched_pairs`: deterministic, non-reused, opposite-label matches;
- `<model>_directed_pairs`: both negative→positive and positive→negative cases;
- `common_matched_pairs` and `common_directed_pairs`: equal length for all selected tokenizers.

SST prefixes these names with `tigges_` or `neutral_removed_` and also preserves its five legacy
RQ1 configurations. DynaSent prefixes them with `r1_` or `r2_`.

Directional rows use `source_*` for the activation donor with the desired label and `target_*` for
the receiver prompt before patching. Therefore, `negative_to_positive` has a positive source and a
negative target. Pair generation uses only dataset labels and tokenizer lengths; it does not use
model-predicted labels, logits, GPT labels, or sentiment-direction projections.

## Local-only runs and visibility

Omit `--push-to-hub` to save locally only. When publishing, a custom private repository can be set:

```bash
sentiment-manifold preprocess-imdb \
  --hub-repo-id YOUR_ACCOUNT/YOUR_PRIVATE_REPO \
  --push-to-hub --private
```

Before using the data in causal experiments, freeze source and tokenizer revisions, inspect the
pair counts per model, and decide in advance whether comparisons use each model's maximal pair set
or the smaller common set. Use the common set for directly comparable cross-model flip rates.

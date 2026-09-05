# RQ2 — Valence/sentiment alignment and OOD transfer

## Research question

How well do valence directions learned from AIT V-oc align with sentiment directions, and how
causally useful are those directions on SST, IMDb, and DynaSent across model families?

Use Difference-in-means, logistic regression and DAS fitting methods for learning `sentiment direction` and `valence direction`.

- [ ] **1.** For a start, I want to fit the three methods on ToyMovieReview datasets to learn sentiment sentiment direction using `gpt2-small and qwen-0.6b`, check first, middle and last layer similarity of sentiment directions and then test out logit difference and logit flip percent by patching all token positions. However, the activations used for the fitting methods would be extracted from [last token position] and [adjective] position in the prompt. Ideally, I want to know which of the activations provide better on in-distribution and out-distribution ToyMovieReview and SST datasets respectively.

- [ ] **2.** Secondly, we would then train the 3 methods to learn sentiment and valence direction using ToyMovieReview and AIT respectively for all 4 models. Here, we use about same amount of datasets for training both directions. What this means is, roughly about 55 data samples for both ToyMovieReview and AIT. We observe the effect on logit difference and logit flip percent. The method of extracting activations for learning sentiment direction depends on which performs best from 1 above. For valence direction, activation extraction should involve all activations of all tokens per data sample and then averaging them to train the fitting methods.

- [ ] **3.** Thirdly, we can then train on large amount of the data samples on AIT and observe if there is discernible performance on eval datasets like ToyMovieReview and SST using metrics we have been using above.

- [ ] **4.** Run first, middle, and last layer within-model geometric alignment for sentiment directions and secondly the same layer within-model geometric alignments for sentiment and valence directions.

## Data contract

- AIT V-oc is the supervised valence-direction training dataset. Its gold ordinal labels are
  collapsed to binary polarity for pairing, but examples are not correctness-filtered by a model.
- SST, IMDb, and DynaSent are zero-shot evaluation datasets. Pythia-2.8B must agree with each
  dataset's gold binary label before an example can enter an equal-length patching pair.
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

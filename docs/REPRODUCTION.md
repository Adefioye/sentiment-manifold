# Reproduction guide

## Stages

1. Run `inspect-data` for each tokenizer and record how many of the paper's adjectives remain one token.
2. Run a small layer subset on CPU/MPS/CUDA and confirm artifacts and metrics are finite.
3. Run all GPT-2 Small boundaries with OpenWebText disabled. Select each method's layer by SST **dev** causal recovery.
4. Compare modern GPT-2 results with the cached/reference TransformerLens directions and paper plots. Investigate hook, normalization, prompt, token, or pairing discrepancies before changing methodology.
5. Freeze the pipeline and switch SST to `test`. Do not reselect the layer.
6. Reproduce the paper's first-layer OpenWebText projection analysis separately. Run resample
   ablation only as an explicitly labelled exploratory diagnostic.
7. Repeat the same protocol for Qwen3-0.6B; describe it as an extension, not part of the original paper replication.

The checked-in upstream `prompts.yaml` currently contains 54 training adjectives (30 positive and 24 negative) and 30 test adjectives, while the paper describes a 55/30 split. This project preserves the verifiable source list and records the discrepancy instead of inventing a 55th word. If an archived paper artifact supplies the missing item, add it with provenance and treat the resulting run as a separately versioned dataset.

## Commands

```bash
sentiment-manifold inspect-data --config configs/reproduction.yaml
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --device auto
sentiment-manifold reproduce --config configs/reproduction.yaml --model qwen-0.6b --device auto
sentiment-manifold reproduce --config configs/reproduction.yaml --model gpt2-small --with-openwebtext-resample-ablation
sentiment-manifold plot --run-dir outputs/gpt2-small
```

The last command is optional and exploratory; it is not the paper's core OpenWebText evaluation. For a smoke run, replace `experiment.layers: all` with a short list such as `[0, 6, 12]`, set `das.epochs: 1`, and limit SST pairs/OpenWebText samples. These changes are debugging settings and must not be reported as the final reproduction.

## Colab and Google Drive

Install the project editable in Colab and call `maybe_mount_google_drive(True)` only when persistent checkpoints are wanted. Set `experiment.output_dir` or `SENTIMENT_MANIFOLD_OUTPUT_DIR` to `/content/drive/MyDrive/sentiment-manifold`; otherwise outputs stay in the Colab runtime. The notebooks expose this choice near the top.

## Numerical comparison checklist

Record the exact model and tokenizer revision, dependency lock, dtype, accelerator, seed, prompt bytes, retained vocabulary, answer token IDs, paired SST IDs, layer-boundary definition, direction orientation, DAS losses, and per-pair metrics. Recovery remains unclipped; values above 100% are possible and informative.

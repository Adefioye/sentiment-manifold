#!/usr/bin/env bash
set -euo pipefail

sentiment-geometry compare-sentiment-positions \
  --config configs/sentiment_position_comparison.yaml \
  --model gpt2-small \
  --model qwen-0.6b \
  --method mean_diff \
  --method logistic_regression \
  --method das \
  --fit-position adjective \
  --fit-position final \
  --all-non-embedding-layers \
  --device auto \
  --dtype auto \
  --batch-size 16 \
  --seed 0 \
  --logistic-c 1.0 \
  --logistic-max-iter 1000 \
  --logistic-tol 0.0001 \
  --das-learning-rate 0.001 \
  --das-weight-decay 0.0 \
  --das-epochs 64 \
  --das-batch-size 128 \
  --das-max-grad-norm 1.0 \
  --sst-repo-id kokolamba/sentiment-manifold-sst-pythia-2.8b \
  --sst-revision bcffb933a34a48b409a7caf3a53f0fe7bb8152cc \
  --hf-token-env HF_TOKEN \
  --output-dir outputs/sentiment-position-comparison \
  --checkpoint-dir checkpoints/sentiment-position-comparison

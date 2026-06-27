# FNO-based Disk Evolution Predictor

This repository contains the code needed to train and evaluate neural operators
for transient FARGO disk-evolution prediction.

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
```

For GPU/server runs, install the server requirements instead:

```bash
.venv/bin/pip install -r requirements-server.txt
```

## Required Files

Place the full FARGO memmap dataset here:

```text
data/fargo_transient_10orbits_128f/
```

Place downloaded checkpoints here:

```text
artifacts/main_rollout/
  plain_fno_checkpoint.pkl
  fno_reflect2d_checkpoint.pkl

artifacts/baselines/
  plain_fno/plain_fno_checkpoint.pkl
  fno_reflect2d/fno_reflect2d_checkpoint.pkl
  fno_radial/fno_radial_checkpoint.pkl
  fno/fno_checkpoint.pkl
  convlstm/convlstm_checkpoint.pkl
  periodic_unet/periodic_unet_checkpoint.pkl
  unet/unet_checkpoint.pkl
```

`data/`, `results/`, `outputs/`, and `artifacts/` are ignored by git.

## Check Checkpoints

```bash
.venv/bin/python scripts/inspect_fargo_checkpoints.py \
  artifacts/main_rollout artifacts/baselines
```

## Notebook Demo

```bash
jupyter lab notebooks/fargo_checkpoint_reproduction_demo.ipynb
```

The notebook can inspect checkpoints without the full dataset. Rollout cells
require `data/fargo_transient_10orbits_128f/`.

## Evaluate A Saved Checkpoint

```bash
.venv/bin/python scripts/evaluate_fargo_checkpoint_rollouts.py \
  --checkpoint artifacts/main_rollout/plain_fno_checkpoint.pkl \
  --dataset data/fargo_transient_10orbits_128f \
  --output-dir results/demo_plain_fno_rollouts \
  --horizons 8 16 32 64 \
  --batch-size 1 \
  --eval-batches 12 \
  --no-physics-metrics \
  --jax-platform cpu
```

Use `--jax-platform cuda` on the server.

## Train Models

```bash
.venv/bin/python scripts/train_fargo_operator.py \
  --dataset data/fargo_transient_10orbits_128f \
  --output-dir results/main_rollout \
  --models plain_fno fno_reflect2d \
  --steps 6000 \
  --batch-size 8 \
  --width 64 \
  --depth 4 \
  --modes-r 24 \
  --modes-theta 32 \
  --lr 8e-4 \
  --eval-every 250 \
  --eval-batches 24 \
  --normalization-samples 2048 \
  --time-input-units normalized \
  --dt-units orbits \
  --rollout-train-weight 0.05 \
  --rollout-train-horizon 4 \
  --rollout-horizon 16 \
  --jax-platform cuda
```

## High-Resolution Validation Command

```bash
JAX_PLATFORMS=cuda python scripts/evaluate_fargo_checkpoint_rollouts.py \
  --checkpoint artifacts/main_rollout/plain_fno_checkpoint.pkl \
  --dataset data/hires6_eval_view \
  --output-dir results/hires6_eval_plain_fno_rollouts \
  --horizons 8 16 32 64 \
  --batch-size 1 \
  --eval-batches 12 \
  --no-physics-metrics \
  --jax-platform cuda
```

Reported high-resolution relL2 results:

```text
rollout@8   4.0845%
rollout@16  5.3909%
rollout@32  8.0821%
rollout@64 11.8092%
```

# FNO-based Disk Evolution Predictor

This repository contains the original steady PPDONet code, a vendored FARGO3D
solver, and the final v3.1 transient FARGO operator experiments.  The release
branch keeps the old exploratory scripts for traceability, while the final v3.1
code is organized as a normal machine-learning project under `src/`.

## Project Layout

```text
src/fargo_operator/                 modular v3.1 implementation
  config.py                         CLI arguments and validation
  constants.py                      shared constants and result dataclasses
  data.py                           memmap dataset loading, splits, batches
  models.py                         FNO/U-Net/ConvLSTM architectures
  training.py                       JAX/Haiku training loop
  evaluation.py                     RMSE, rollout, physics diagnostics
  outputs.py                        checkpoint, JSON, and loss plot output
  runner.py                         end-to-end benchmark orchestration
  checkpoints.py                    NumPy 1.x/2.x-compatible checkpoint loader

scripts/                            runnable entry points and legacy experiments
  benchmark_fargo_operators_v3.1.py thin v3.1 CLI compatibility entry
  evaluate_fargo_checkpoint_rollouts.py
  diagnose_rollout_spectrum.py
  inspect_fargo_checkpoints.py
  generate_fargo_data_v2.py

notebooks/                          reproduction demo notebook
artifacts/main_rollout/             local path for main long-rollout checkpoints
v3_1_repro_ckpts/                   local path for teammate baseline checkpoints
fargo3d/                            vendored FARGO3D source and setups
ppdonet/                            original steady PPDONet project/code/weights
fargo_*.py                          earlier modular v3 utilities kept intact
scripts/benchmark_fargo_operators*.py earlier experiment scripts kept intact
```

`data/`, `results/`, `outputs/`, `artifacts/`, and `v3_1_repro_ckpts/` are local
or downloaded artifacts and are ignored by default.  The code branch is pushed
without large checkpoint blobs; download or unpack the checkpoint bundle into
the paths shown below before reproducing the full report numbers.

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
```

For GPU/server runs, use `requirements-server.txt` and the cluster CUDA/JAX
setup.  Build FARGO3D only when generating new simulations:

```bash
cd fargo3d
make SETUP=fargo_nu PARALLEL=0 GPU=0
cd ..
```

## Checkpoints

Baseline v3.1 checkpoints from the teammate should be placed under
`v3_1_repro_ckpts/`:

| Model | Checkpoint | HPT RMSE | Rollout RMSE |
|---|---|---:|---:|
| Plain 2D FNO | `v3_1_repro_ckpts/plain_fno/plain_fno_checkpoint.pkl` | 0.019770 | 0.067651 |
| Reflect-2D FNO | `v3_1_repro_ckpts/fno_reflect2d/fno_reflect2d_checkpoint.pkl` | 0.022246 | 0.076654 |
| Geometry-radial FNO | `v3_1_repro_ckpts/fno_radial/fno_radial_checkpoint.pkl` | 0.024564 | 0.088451 |
| Geometry-aware FNO | `v3_1_repro_ckpts/fno/fno_checkpoint.pkl` | 0.027527 | 0.105782 |
| ConvLSTM | `v3_1_repro_ckpts/convlstm/convlstm_checkpoint.pkl` | 0.022543 | 0.134574 |
| Periodic U-Net | `v3_1_repro_ckpts/periodic_unet/periodic_unet_checkpoint.pkl` | 0.032950 | 0.142702 |
| U-Net | `v3_1_repro_ckpts/unet/unet_checkpoint.pkl` | 0.034199 | 0.141151 |

The main long-rollout checkpoints should be placed under
`artifacts/main_rollout/`:

```text
artifacts/main_rollout/plain_fno_checkpoint.pkl
artifacts/main_rollout/fno_reflect2d_checkpoint.pkl
```

These are the width-64/depth-4 rollout-training checkpoints from the server run:

```text
results/server_v31_main_reflect2d_rolltrain_w64d4_6000/
```

Their internal dataset path is the server dataset:

```text
fargo_data_v2/data/fargo_transient_10orbits_128f
```

When evaluating locally, pass `--dataset` to override that path.

The checkpoints were written on the server with NumPy 2.x.  Local NumPy 1.x
environments should load them through `src.fargo_operator.checkpoints` or the
compatibility module `fargo_checkpoint.py`, not raw `pickle.load`.

Inspect checkpoints:

```bash
.venv/bin/python scripts/inspect_fargo_checkpoints.py \
  v3_1_repro_ckpts artifacts/main_rollout
```

## Reproduction Demo

Open:

```text
notebooks/fargo_checkpoint_reproduction_demo.ipynb
```

The notebook can summarize checkpoints without the full dataset.  Rollout cells
require a memmap dataset such as:

```text
data/fargo_transient_10orbits_128f/
```

Evaluate saved checkpoints:

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

On the server, use `--jax-platform cuda`.

High-resolution external validation command used for the report/response:

```bash
JAX_PLATFORMS=cuda python scripts/evaluate_fargo_checkpoint_rollouts.py \
  --checkpoint results/server_v31_main_reflect2d_rolltrain_w64d4_6000/plain_fno_checkpoint.pkl \
  --dataset data/hires6_eval_view \
  --output-dir results/hires6_eval_plain_fno_rollouts \
  --horizons 8 16 32 64 \
  --batch-size 1 \
  --eval-batches 12 \
  --no-physics-metrics \
  --jax-platform cuda
```

Reported high-resolution relL2 results: rollout@8 = 4.0845%, rollout@16 =
5.3909%, rollout@32 = 8.0821%, rollout@64 = 11.8092%.

## Train v3.1 Models

The final v3.1 CLI is now modular but keeps the old command name:

```bash
.venv/bin/python scripts/benchmark_fargo_operators_v3.1.py \
  --dataset data/fargo_transient_10orbits_128f \
  --output-dir results/v31_repro \
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

## Download Checklist For Server Artifacts

Required for the long-rollout release/demo:

```text
results/server_v31_main_reflect2d_rolltrain_w64d4_6000/
  plain_fno_checkpoint.pkl
  fno_reflect2d_checkpoint.pkl
  metrics.json
  loss_history.json
  loss_plain_fno*.png
  loss_fno_reflect2d*.png

results/diagnostics_main_plain_fno_rollout64/
  rollout_spectral_diagnostics.json
  *.png        # if generated

results/hires6_eval_plain_fno_rollouts/
  rollout_metrics.json
```

Optional ConvLSTM long-rollout artifacts:

```text
results/server_long_convlstm_w48d3_b4_6000/
results/server_long_convlstm_w48d3_b4_trunc_remat_6000/
results/server_long_convlstm_w64d4_b16_6000/
```

For each optional ConvLSTM directory, keep `*_checkpoint.pkl`, `metrics.json`,
`loss_history.json`, and `loss_*.png`.

## Legacy Code

The repository intentionally keeps old experiment code, the original steady
PPDONet implementation, and FARGO3D source files.  They are not removed because
they document the project history and are still useful for comparison.  The
release-facing transient v3.1 implementation is the `src/fargo_operator/` package
plus the scripts listed above.

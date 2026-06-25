# Time-Dependent PPDONet

This repository keeps the original steady PPDONet code and the FARGO3D solver
alongside a small set of time-dependent operator experiments.

## Layout

```text
fargo3d/           vendored FARGO3D source and setup files
ppdonet/           original steady PPDONet code and pretrained weights
fno.py             reusable FNO building blocks for this project
ppdonet_steady.py  helpers for querying the steady PPDONet baseline
training.py        shared lightweight JAX/Haiku training utilities
scripts/           runnable data-generation and benchmark entry points
data/              local generated datasets, ignored by git
results/           local experiment outputs, ignored by git
```

`data/` and `results/` are intentionally local-only. Regenerate datasets and
metrics from the scripts instead of committing them.

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
```

Build FARGO3D when needed:

```bash
cd fargo3d
make SETUP=fargo_nu PARALLEL=0 GPU=0
cd ..
```

## Generate A Small FARGO Dataset

Use a short dataset name because FARGO3D has fixed-size internal path buffers on
some systems.

```bash
.venv/bin/python scripts/generate_fargo_dataset_v1.py \
  --dataset-name poc12 \
  --skip-build \
  --num-random-cases 8 \
  --seed 42 \
  --nx 64 --ny 32 \
  --ninterm 4 --ntot 24
```

This writes `data/poc12/dataset.npz` with 12 cases and 7 frames.

## Run The Proof-Of-Concept Benchmark

For a quick local check, reduce `--steps`. For the real run, use the server:

```bash
.venv/bin/python scripts/benchmark_fargo_operators.py \
  --dataset data/poc12/dataset.npz \
  --heldout-cases 1 5 9 \
  --steps 1500 \
  --batch-size 16 \
  --width 32 \
  --latent 48 \
  --depth 3 \
  --modes-r 8 \
  --modes-theta 12 \
  --channels log_sigma v_r v_theta \
  --models state_deeponet pointwise fno fno_flow \
  --train-spans 1 2 3 \
  --consistency-weight 0.1 \
  --consistency-spans 1 1 \
  --rollout-weight 0.25 \
  --speed-repeats 5 \
  --output-dir results/poc12_operator_benchmark
```

The benchmark always reports persistence and, by default, calls the bundled
pretrained PPDONet checkpoints as a steady time-independent baseline. It then
trains the selected one-step models on the requested channels. The FNO variant
is the main proof-of-concept model; the state-conditioned DeepONet and pointwise
residual model are sanity baselines.

## Run The Larger FARGO Operator Benchmark

The main transient implementation is split into root-level modules:

```text
fargo_data.py       memory-mapped dataset loading, temporal splits, batches
fargo_experiment.py method definitions and shared evaluation suites
fargo_model.py      core time-conditioned disk FNO architecture
fargo_training.py   training config/result objects and JAX/Haiku training loop
fargo_metrics.py    RMSE, relative L2, rollout, and semigroup evaluation
fargo_outputs.py    JSON, checkpoint, and loss-plot helpers
fargo_benchmark.py  orchestration used by the CLI script
fno.py              small shared FNO utilities for earlier steady demos
```

The v3 implementation centers on `fno`: a time-conditioned disk FNO that learns
the transient field-to-field update. `fno_flow` is an extension of the same
model family that adds multi-span training and optional semigroup consistency.
Use `fno` as the main baseline, then report whether `fno_flow` improves held-out
time/parameter and rollout metrics against the same persistence baselines.

Generate a larger memmap dataset with `scripts/generate_fargo_data_v2.py`, then
run:

```bash
.venv/bin/python scripts/benchmark_fargo_operators_v3.py \
  --dataset data/fargo_transient_10orbits_128f \
  --models fno \
  --steps 5000 \
  --batch-size 8 \
  --output-dir results/fargo_operator_benchmark_v3
```

Summarize a finished run:

```bash
python scripts/summarize_fargo_metrics.py results/fargo_operator_benchmark_v3/metrics.json
```

For local tuning, compare small FNO variants with the same dataset/capacity:

```bash
python scripts/summarize_fargo_metrics.py \
  results/v3_local_ablation_fno_span1_200/metrics.json \
  results/v3_local_ablation_fno_span12_200/metrics.json \
  results/v3_local_ablation_fno_span124_200/metrics.json \
  results/v3_local_tune_fno_w32d3m16_b2_800_lr8e4/metrics.json \
  results/v3_local_tune_fno_rollh4_w005_800/metrics.json
```

On the current 16-case local v3 dataset, span-1 FNO is the strongest default;
multi-span training is treated as an extension to validate, not the default. The
best local tuning run so far uses `--width 32 --depth 3 --modes-theta 16
--batch-size 2 --steps 800 --lr 8e-4 --rollout-train-weight 0.05
--rollout-train-horizon 4`, reducing held-out parameter/time error from the
persistence baseline's 5.57% to 2.85%, and rollout@8 from 22.74% to 11.01%.

On the NCSA Jupyter server, after generating the full v2 dataset under
`fargo_data_v2/data/fargo_transient_10orbits_128f`, start with a short training
smoke test:

```bash
python scripts/benchmark_fargo_operators_v3.py \
  --dataset fargo_data_v2/data/fargo_transient_10orbits_128f \
  --output-dir results/server_v3_smoke \
  --models fno \
  --steps 200 \
  --batch-size 2 \
  --width 32 \
  --depth 3 \
  --modes-theta 16 \
  --lr 8e-4 \
  --eval-every 50 \
  --eval-batches 4 \
  --normalization-samples 256 \
  --rollout-train-weight 0.05 \
  --rollout-train-horizon 4 \
  --rollout-horizons 1 2 4 8 \
  --jax-platform gpu
```

For a longer main FNO run, increase capacity and steps:

```bash
python scripts/benchmark_fargo_operators_v3.py \
  --dataset fargo_data_v2/data/fargo_transient_10orbits_128f \
  --output-dir results/server_v3_fno_w64_d4 \
  --models fno \
  --steps 12000 \
  --batch-size 8 \
  --width 64 \
  --depth 4 \
  --modes-theta 32 \
  --lr 8e-4 \
  --warmup-steps 500 \
  --min-lr-ratio 0.05 \
  --eval-every 250 \
  --eval-batches 24 \
  --normalization-samples 2048 \
  --fno-spans 1 \
  --rollout-train-weight 0.05 \
  --rollout-train-horizon 4 \
  --rollout-horizon 16 \
  --rollout-horizons 1 2 4 8 16 \
  --jax-platform gpu
```

Then run the FNO-flow extension as a controlled comparison:

```bash
python scripts/benchmark_fargo_operators_v3.py \
  --dataset fargo_data_v2/data/fargo_transient_10orbits_128f \
  --output-dir results/server_v3_compare_w64_d4 \
  --models fno fno_flow \
  --steps 12000 \
  --batch-size 8 \
  --width 64 \
  --depth 4 \
  --modes-theta 32 \
  --lr 8e-4 \
  --warmup-steps 500 \
  --min-lr-ratio 0.05 \
  --eval-every 250 \
  --eval-batches 24 \
  --normalization-samples 2048 \
  --fno-spans 1 \
  --fno-flow-spans 1 2 4 8 \
  --consistency-weight 0.02 \
  --consistency-spans 2 2 \
  --rollout-horizon 16 \
  --rollout-horizons 1 2 4 8 16 \
  --jax-platform gpu
```

The v3 metrics include persistence baselines, held-out parameter/time errors,
per-channel RMSE/relative L2, per-span errors, rollout errors at multiple
horizons, and the semigroup consistency error for `fno_flow`.

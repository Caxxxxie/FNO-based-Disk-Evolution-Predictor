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
.venv/bin/python scripts/generate_fargo_dataset.py \
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
  --models state_deeponet pointwise fno \
  --rollout-weight 0.25 \
  --speed-repeats 5 \
  --no-pretrained-ppdonet \
  --output-dir results/poc12_operator_benchmark
```

The benchmark always reports persistence, then trains the selected simple
one-step models. The FNO variant is the main proof-of-concept model; the
state-conditioned DeepONet and pointwise residual model are sanity baselines.

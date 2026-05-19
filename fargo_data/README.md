# Local FARGO3D Sanity Data

This folder stores small FARGO3D datasets used for local operator sanity checks.
The current dataset is intentionally tiny:

```text
smoke_fargo_nu:
  setup: fargo_nu
  cases: 4 parameter points
  frames: 7 time snapshots
  grid: 32 x 64
  field: log_sigma, plus sigma/v_r/v_theta in the NPZ
```

It is not a production-quality disk dataset. It is just enough to check whether
our time-dependent operator implementations can learn from real solver output
instead of a hand-made synthetic transient.

Regenerate it with:

```bash
.venv/bin/python scripts/generate_fargo_dataset.py \
  --dataset-name smoke_fargo_nu \
  --nx 64 --ny 32 --ninterm 4 --ntot 24 \
  --skip-build --keep-raw
```

The raw FARGO output is ignored by git. The packed `dataset.npz`, generated
parameter files, planet configs, and metadata are small enough to keep.

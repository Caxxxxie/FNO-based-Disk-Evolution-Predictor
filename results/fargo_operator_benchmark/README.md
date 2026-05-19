# FARGO Operator Benchmark

This benchmark compares a few tiny operator variants on
`fargo_data/smoke_fargo_nu/dataset.npz`.

Run:

```bash
.venv/bin/python scripts/benchmark_fargo_operators.py \
  --steps 300 --batch-size 8 --width 24 --latent 32 --depth 2 \
  --modes-r 6 --modes-theta 8 --speed-repeats 5 \
  --models ppdonet_style state_deeponet fno
```

The metric is RMSE in normalized `log_sigma` units. This is not a final scientific
benchmark; it is a local sanity check on real FARGO3D output.

Current short-run takeaway from the checked-in `metrics.json`:

```text
normalized log_sigma RMSE:
  pretrained PPDONet as steady predictor:
    heldout_param 7.5081, heldout_time 11.1081
  PPDONet-style final-frame DeepONet:
    heldout_param 0.7863, heldout_time 0.7863
  persistence x_n -> x_{n+1}:
    heldout_param 0.3630, heldout_time 0.1787, two-step 0.2067
  state-conditioned DeepONet step:
    heldout_param 0.3575, heldout_time 0.1772, two-step 0.2259
  FNO residual step:
    heldout_param 0.2765, heldout_time 0.1972, two-step 0.2629
```

So FNO is the best held-out-parameter learner in this tiny run, but it is not
yet the most stable rollout model. The useful next idea is not only "swap the
backbone"; it is to add rollout loss, physics/PINO residuals, or a correction
head so the time-step map behaves well after repeated application.

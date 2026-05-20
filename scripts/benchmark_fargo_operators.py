#!/usr/bin/env python3
"""Compare small operator variants on a real tiny FARGO3D time series.

This is intentionally a sanity benchmark. The bundled dataset is tiny, coarse,
and short; it is useful for checking whether an implementation can learn a
real solver-produced transient signal before we spend time on larger sweeps.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[1]
PPDONET_ROOT = ROOT / "ppdonet"
jax.config.update("jax_platforms", "cpu")


@dataclass
class MetricRow:
    train_rmse: float
    heldout_param_rmse: float
    heldout_time_rmse: float
    two_step_rollout_rmse: float | None
    speed_ms_per_batch: float
    train_rmse_by_channel: dict[str, float] | None = None
    heldout_param_rmse_by_channel: dict[str, float] | None = None
    heldout_time_rmse_by_channel: dict[str, float] | None = None
    two_step_rollout_rmse_by_channel: dict[str, float] | None = None
    semigroup_rmse: float | None = None
    semigroup_rmse_by_channel: dict[str, float] | None = None
    trained_steps: int | None = None
    best_step: int | None = None
    best_monitor_rmse: float | None = None
    heldout_param_rel_l2_pct: float | None = None
    heldout_param_rel_l2_pct_by_channel: dict[str, float] | None = None
    two_step_rollout_rel_l2_pct: float | None = None
    two_step_rollout_rel_l2_pct_by_channel: dict[str, float] | None = None
    train_rel_l2_pct: float | None = None
    train_rel_l2_pct_by_channel: dict[str, float] | None = None
    heldout_time_rel_l2_pct: float | None = None
    heldout_time_rel_l2_pct_by_channel: dict[str, float] | None = None
    direct_two_step_rmse: float | None = None
    direct_two_step_rmse_by_channel: dict[str, float] | None = None
    direct_two_step_rel_l2_pct: float | None = None
    direct_two_step_rel_l2_pct_by_channel: dict[str, float] | None = None


def relative_l2_pct(pred, target, eps: float = 1.0e-12):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    if pred_arr.ndim > 1:
        pred_flat = pred_arr.reshape((pred_arr.shape[0], -1))
        target_flat = target_arr.reshape((target_arr.shape[0], -1))
        num = np.linalg.norm(pred_flat - target_flat, axis=1)
        den = np.linalg.norm(target_flat, axis=1) + eps
        return float(100.0 * np.mean(num / den))
    num = np.linalg.norm(pred_arr - target_arr)
    den = np.linalg.norm(target_arr) + eps
    return float(100.0 * num / den)


def relative_l2_pct_by_channel(pred, target, channel_names, eps: float = 1.0e-12):
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    values = {}
    for idx, name in enumerate(channel_names):
        values[name] = relative_l2_pct(pred_arr[..., idx], target_arr[..., idx], eps)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "smoke_fargo_nu" / "dataset.npz")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--modes-r", type=int, default=8)
    parser.add_argument("--modes-theta", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-spans",
        type=int,
        nargs="+",
        default=[1],
        help="Supervised frame spans for learning Phi_dt. Use e.g. 1 2 3 for variable-time maps.",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Weight for semigroup consistency Phi_{a+b}(x) ~= Phi_b(Phi_a(x)).",
    )
    parser.add_argument(
        "--consistency-spans",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("SPAN_A", "SPAN_B"),
        help="Frame spans a,b used in semigroup consistency.",
    )
    parser.add_argument(
        "--identity-weight",
        type=float,
        default=0.0,
        help="Weight for zero-time identity consistency Phi_0(x) ~= x.",
    )
    parser.add_argument(
        "--flow-residual-output",
        action="store_true",
        help="Use a residual-form output for fno_flow while still training variable time spans.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate full training RMSE every N steps for early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without training RMSE improvement. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1.0e-4,
        help="Minimum training RMSE improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--heldout-case",
        type=int,
        default=1,
        help="Parameter case to hold out. Negative values count from the end.",
    )
    parser.add_argument(
        "--heldout-cases",
        type=int,
        nargs="+",
        default=None,
        help="Parameter cases to hold out. Overrides --heldout-case.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed-repeats", type=int, default=10)
    parser.add_argument("--no-pretrained-ppdonet", action="store_true")
    parser.add_argument("--save-plots", action="store_true", help="Save disk snapshot comparison figures.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["log_sigma", "v_r", "v_theta"],
        choices=["log_sigma", "v_r", "v_theta"],
        help="FARGO fields to train/evaluate jointly. Defaults to all bundled PPDONet channels.",
    )
    parser.add_argument(
        "--loss-channel-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-channel training loss weights, in the same order as --channels. Values are normalized to mean one.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["time_deeponet", "state_deeponet", "fno"],
        choices=["time_deeponet", "state_deeponet", "pointwise", "fno", "fno_flow"],
        help="Model subset to train. Persistence is always reported.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "fargo_operator_benchmark")
    args = parser.parse_args()
    if any(span <= 0 for span in args.train_spans):
        raise ValueError("--train-spans must contain positive integers")
    if any(span <= 0 for span in args.consistency_spans):
        raise ValueError("--consistency-spans must contain positive integers")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.loss_channel_weights is not None and len(args.loss_channel_weights) != len(args.channels):
        raise ValueError("--loss-channel-weights must have the same length as --channels")
    return args


def normalize_params(params: np.ndarray) -> np.ndarray:
    values = params.astype(np.float32).copy()
    values[:, 0] = np.log10(values[:, 0])
    values[:, 2] = np.log10(values[:, 2])
    # Match the bundled PPDONet training domain:
    # log10(alpha) in [-3.52, -1], ASPECTRATIO in [0.05, 0.1],
    # log10(planetmass) in [-4.3, -2.7].
    lower = np.asarray([-3.52, 0.05, -4.3], dtype=np.float32)
    upper = np.asarray([-1.0, 0.10, -2.7], dtype=np.float32)
    return (2.0 * (values - 0.5 * (lower + upper)) / (upper - lower)).astype(np.float32)


def coordinate_grid(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r_norm = 2.0 * (r - r.min()) / (r.max() - r.min()) - 1.0
    rr = np.broadcast_to(r_norm[:, None], (len(r), len(theta)))
    ss = np.broadcast_to(np.sin(theta)[None, :], (len(r), len(theta)))
    cc = np.broadcast_to(np.cos(theta)[None, :], (len(r), len(theta)))
    return np.stack([rr, ss, cc], axis=-1).astype(np.float32)


def state_features(x: np.ndarray) -> np.ndarray:
    # Mean/std plus a coarse 4 x 8 sensor grid. This is deliberately small so it
    # behaves like a simple state-conditioned DeepONet branch input.
    pooled = x.reshape((x.shape[0], 4, x.shape[1] // 4, 8, x.shape[2] // 8, x.shape[3])).mean(axis=(2, 4))
    stats = np.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=-1)
    return np.concatenate([stats, pooled.reshape((x.shape[0], -1))], axis=-1).astype(np.float32)


def state_features_jax(x: jnp.ndarray) -> jnp.ndarray:
    pooled = x.reshape((x.shape[0], 4, x.shape[1] // 4, 8, x.shape[2] // 8, x.shape[3])).mean(axis=(2, 4))
    stats = jnp.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=-1)
    return jnp.concatenate([stats, pooled.reshape((x.shape[0], -1))], axis=-1)


def load_dataset(path: Path, channels: list[str]):
    data = np.load(path)
    missing = [channel for channel in channels if channel not in data]
    if missing:
        raise KeyError(f"Dataset {path} does not contain requested channels: {missing}")
    fields = np.stack([data[channel].astype(np.float32) for channel in channels], axis=-1)
    raw_params = data["params"].astype(np.float32)
    params = normalize_params(raw_params)
    r = data["r"].astype(np.float32)
    theta = data["theta"].astype(np.float32)
    times = data["times"].astype(np.float32)
    meta = json.loads(data["meta"].item())
    return fields, params, raw_params, r, theta, times, meta


def normalize_fields(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean.reshape((1,) * (values.ndim - 1) + (-1,))) / std.reshape((1,) * (values.ndim - 1) + (-1,))).astype(np.float32)


def denormalize_fields(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std.reshape((1,) * (values.ndim - 1) + (-1,)) + mean.reshape((1,) * (values.ndim - 1) + (-1,))).astype(np.float32)


def make_step_data(x, params, times, param_ids, start_ids, mean, std, spans=(1,), max_target_index=None):
    xs, ys, mus, ts, dts = [], [], [], [], []
    t_scale = float(times[-1] - times[0])
    for p in param_ids:
        for sid in start_ids:
            for span in spans:
                end = sid + span
                if end >= x.shape[1]:
                    continue
                if max_target_index is not None and end > max_target_index:
                    continue
                xs.append(normalize_fields(x[p, sid], mean, std))
                ys.append(normalize_fields(x[p, end], mean, std))
                mus.append(params[p])
                ts.append([(times[sid] - times[0]) / t_scale])
                dts.append([(times[end] - times[sid]) / t_scale])
    xs = np.stack(xs, axis=0).astype(np.float32)
    return {
        "x": jnp.asarray(xs),
        "y": jnp.asarray(np.stack(ys, axis=0).astype(np.float32)),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "t": jnp.asarray(np.asarray(ts, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(dts, dtype=np.float32)),
        "state": jnp.asarray(state_features(xs)),
    }


def add_two_step_target(data, fields, param_ids, start_ids, mean, std):
    targets = []
    for p in param_ids:
        for sid in start_ids:
            targets.append(normalize_fields(fields[p, sid + 2], mean, std))
    data = dict(data)
    data["two_step_target"] = jnp.asarray(np.stack(targets, axis=0).astype(np.float32))
    return data


def make_semigroup_data(fields, params, times, param_ids, start_ids, span_a, span_b, mean, std):
    xs, ys, mus, ts_a, dts_a, ts_b, dts_b, ts_ab, dts_ab = [], [], [], [], [], [], [], [], []
    t_scale = float(times[-1] - times[0])
    total = span_a + span_b
    for p in param_ids:
        for sid in start_ids:
            mid = sid + span_a
            end = sid + total
            if end >= fields.shape[1]:
                continue
            xs.append(normalize_fields(fields[p, sid], mean, std))
            ys.append(normalize_fields(fields[p, end], mean, std))
            mus.append(params[p])
            ts_a.append([(times[sid] - times[0]) / t_scale])
            dts_a.append([(times[mid] - times[sid]) / t_scale])
            ts_b.append([(times[mid] - times[0]) / t_scale])
            dts_b.append([(times[end] - times[mid]) / t_scale])
            ts_ab.append([(times[sid] - times[0]) / t_scale])
            dts_ab.append([(times[end] - times[sid]) / t_scale])
    if not xs:
        raise ValueError(f"No semigroup samples available for spans {span_a}, {span_b}")
    xs = np.stack(xs, axis=0).astype(np.float32)
    return {
        "x": jnp.asarray(xs),
        "y": jnp.asarray(np.stack(ys, axis=0).astype(np.float32)),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "t": jnp.asarray(np.asarray(ts_a, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(dts_a, dtype=np.float32)),
        "state": jnp.asarray(state_features(xs)),
        "t_b": jnp.asarray(np.asarray(ts_b, dtype=np.float32)),
        "dt_b": jnp.asarray(np.asarray(dts_b, dtype=np.float32)),
        "t_ab": jnp.asarray(np.asarray(ts_ab, dtype=np.float32)),
        "dt_ab": jnp.asarray(np.asarray(dts_ab, dtype=np.float32)),
    }


def set_time_span(batch, t, dt):
    batch2 = dict(batch)
    batch2["t"] = t
    batch2["dt"] = dt
    return batch2


def set_zero_time(batch):
    return set_time_span(batch, batch["t"], jnp.zeros_like(batch["dt"]))


def advance_rollout_time(batch):
    batch2 = dict(batch)
    batch2["t"] = batch["t"] + batch["dt"]
    return batch2


class SpectralConv2D(hk.Module):
    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        mr = min(self.modes_r, ny // 2)
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real_pos = hk.get_parameter("weight_pos_real", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        imag_pos = hk.get_parameter("weight_pos_imag", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        real_neg = hk.get_parameter("weight_neg_real", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        imag_neg = hk.get_parameter("weight_neg_imag", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        weight_pos = real_pos + 1j * imag_pos
        weight_neg = real_neg + 1j * imag_neg
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low_pos = jnp.einsum("bhwi,hwio->bhwo", x_ft[:, :mr, :mt, :], weight_pos)
        low_neg = jnp.einsum("bhwi,hwio->bhwo", x_ft[:, -mr:, :mt, :], weight_neg)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low_pos)
        out_ft = out_ft.at[:, -mr:, :mt, :].set(low_neg)
        return jnp.fft.irfft2(out_ft, s=(ny, nx), axes=(1, 2))


def grid_inputs(batch, coords):
    x = batch["x"]
    b, ny, nx, _ = x.shape
    coord = jnp.broadcast_to(jnp.asarray(coords)[None, :, :, :], (b, ny, nx, 3))
    cond = jnp.concatenate([batch["mu"], batch["t"], batch["dt"]], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (b, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coord, cond], axis=-1)


def make_param_deeponet(coords, latent, width, output_channels: int, use_time: bool, use_state: bool, residual: bool):
    coords_flat = jnp.asarray(coords.reshape((-1, 3)))

    def forward(batch):
        x = batch["x"]
        b, ny, nx, _ = x.shape
        branch_parts = [batch["mu"]]
        if use_time:
            branch_parts.extend([batch["t"], batch["dt"]])
        if use_state:
            branch_parts.append(batch["state"])
        branch_input = jnp.concatenate(branch_parts, axis=-1)
        trunk = hk.nets.MLP([width, width, latent * output_channels], activation=jax.nn.tanh)(coords_flat)
        branch = hk.nets.MLP([width, width, latent * output_channels], activation=jax.nn.tanh)(branch_input)
        trunk = trunk.reshape((coords_flat.shape[0], output_channels, latent))
        branch = branch.reshape((b, output_channels, latent))
        y = jnp.einsum("bcl,ncl->bnc", branch, trunk) / math.sqrt(latent)
        y = y.reshape((b, ny, nx, output_channels))
        return x + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_fno(coords, width, modes_r, modes_theta, depth, output_channels: int, residual: bool = True):
    def forward(batch):
        h = hk.Linear(width)(grid_inputs(batch, coords))
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([width, output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_pointwise(coords, width, depth, output_channels: int):
    def forward(batch):
        residual = hk.nets.MLP([width] * depth + [output_channels], activation=jax.nn.gelu)(grid_inputs(batch, coords))
        return batch["x"] + batch["dt"][:, None, None, :] * residual

    return hk.without_apply_rng(hk.transform(forward))


def train_and_eval(
    model,
    train,
    evals,
    args,
    channel_names,
    seed_offset=0,
    rollout_train=None,
    consistency_train=None,
    use_identity_loss=False,
    mean=None,
    std=None,
):
    key = jax.random.PRNGKey(args.seed + seed_offset)
    params = model.init(key, {k: v[:1] for k, v in train.items()})
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.lr))
    opt_state = opt.init(params)
    rollout_weight = args.rollout_weight if rollout_train is not None else 0.0
    consistency_weight = args.consistency_weight if consistency_train is not None else 0.0
    identity_weight = args.identity_weight if use_identity_loss else 0.0
    channel_weights = None
    if args.loss_channel_weights is not None:
        raw_weights = jnp.asarray(args.loss_channel_weights, dtype=jnp.float32)
        channel_weights = raw_weights / jnp.mean(raw_weights)

    def mse_loss(pred, target):
        err = (pred - target) ** 2
        if channel_weights is not None:
            err = err * channel_weights.reshape((1, 1, 1, -1))
        return jnp.mean(err)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch)
        return mse_loss(pred, batch["y"])

    @jax.jit
    def rollout_loss_fn(params, batch, rollout_batch, consistency_batch):
        pred = model.apply(params, batch)
        loss = mse_loss(pred, batch["y"])
        if rollout_batch["x"].shape[0] > 0:
            x1 = model.apply(params, rollout_batch)
            batch2 = advance_rollout_time(rollout_batch)
            batch2["x"] = x1
            batch2["state"] = state_features_jax(x1)
            x2 = model.apply(params, batch2)
            rollout = mse_loss(x2, rollout_batch["two_step_target"])
            loss = loss + rollout_weight * rollout
        if consistency_batch["x"].shape[0] > 0:
            direct = model.apply(params, set_time_span(consistency_batch, consistency_batch["t_ab"], consistency_batch["dt_ab"]))
            first = model.apply(params, consistency_batch)
            second_batch = dict(consistency_batch)
            second_batch["x"] = first
            second_batch["state"] = state_features_jax(first)
            second_batch["t"] = consistency_batch["t_b"]
            second_batch["dt"] = consistency_batch["dt_b"]
            second = model.apply(params, second_batch)
            semigroup = mse_loss(direct, second)
            anchor = mse_loss(direct, consistency_batch["y"])
            loss = loss + consistency_weight * (semigroup + anchor)
        if identity_weight > 0.0:
            identity = model.apply(params, set_zero_time(batch))
            loss = loss + identity_weight * mse_loss(identity, batch["x"])
        return loss

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def rollout_train_step(params, opt_state, batch, rollout_batch, consistency_batch):
        loss, grads = jax.value_and_grad(rollout_loss_fn)(params, batch, rollout_batch, consistency_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def monitor_rmse(params, batch):
        return jnp.sqrt(loss_fn(params, batch))

    n = train["x"].shape[0]
    n_rollout = 0 if rollout_train is None else rollout_train["x"].shape[0]
    n_consistency = 0 if consistency_train is None else consistency_train["x"].shape[0]
    best_params = params
    best_rmse = math.inf
    best_step = 0
    stale_evals = 0
    trained_steps = 0
    for step in range(1, args.steps + 1):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        if n_rollout == 0 and n_consistency == 0 and identity_weight == 0.0:
            params, opt_state, _ = train_step(params, opt_state, batch)
        else:
            if n_rollout:
                key, rollout_key = jax.random.split(key)
                ridx = jax.random.randint(rollout_key, (args.batch_size,), 0, n_rollout)
                rollout_batch = {k: v[ridx] for k, v in rollout_train.items()}
            else:
                rollout_batch = {k: v[:0] for k, v in train.items()}
                rollout_batch["two_step_target"] = train["y"][:0]
            if n_consistency:
                key, consistency_key = jax.random.split(key)
                cidx = jax.random.randint(consistency_key, (args.batch_size,), 0, n_consistency)
                consistency_batch = {k: v[cidx] for k, v in consistency_train.items()}
            else:
                consistency_batch = {
                    "x": train["x"][:0],
                    "y": train["y"][:0],
                    "mu": train["mu"][:0],
                    "t": train["t"][:0],
                    "dt": train["dt"][:0],
                    "state": train["state"][:0],
                    "t_ab": train["t"][:0],
                    "dt_ab": train["dt"][:0],
                }
            params, opt_state, _ = rollout_train_step(params, opt_state, batch, rollout_batch, consistency_batch)
        trained_steps = step
        if step % args.eval_every == 0 or step == args.steps:
            current_rmse = float(monitor_rmse(params, train))
            if current_rmse < best_rmse - args.early_stop_min_delta:
                best_rmse = current_rmse
                best_step = step
                best_params = params
                stale_evals = 0
            else:
                stale_evals += 1
            if args.early_stop_patience and stale_evals >= args.early_stop_patience:
                break
    params = best_params

    @jax.jit
    def channel_mse(params, batch):
        pred = model.apply(params, batch)
        return jnp.mean((pred - batch["y"]) ** 2, axis=(0, 1, 2))

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

    def rmse_by_channel(data):
        values = np.sqrt(np.asarray(channel_mse(params, data)))
        return {name: float(value) for name, value in zip(channel_names, values)}

    def rel_l2_pct(data):
        pred = np.asarray(model.apply(params, data))
        target = np.asarray(data["y"])
        if mean is not None and std is not None:
            pred = denormalize_fields(pred, mean, std)
            target = denormalize_fields(target, mean, std)
        return (
            relative_l2_pct(pred, target),
            relative_l2_pct_by_channel(pred, target, channel_names),
        )

    @jax.jit
    def apply_fn(params, batch):
        return model.apply(params, batch)

    def speed(data):
        apply_fn(params, data).block_until_ready()
        start = time.perf_counter()
        for _ in range(args.speed_repeats):
            y = apply_fn(params, data)
        y.block_until_ready()
        return (time.perf_counter() - start) * 1000.0 / args.speed_repeats

    train_info = {
        "trained_steps": trained_steps,
        "best_step": best_step,
        "best_monitor_rmse": best_rmse,
    }
    return params, rmse, rmse_by_channel, rel_l2_pct, speed, train_info


def rollout_two_steps(model, params, data, channel_names, mean=None, std=None):
    y2 = data["two_step_target"]
    batch = {k: v for k, v in data.items() if k != "two_step_target"}
    x1 = model.apply(params, batch)
    batch2 = advance_rollout_time(batch)
    batch2["x"] = x1
    batch2["state"] = state_features_jax(x1)
    x2 = model.apply(params, batch2)
    err = (x2 - y2) ** 2
    by_channel = np.sqrt(np.asarray(jnp.mean(err, axis=(0, 1, 2))))
    pred_np = np.asarray(x2)
    target_np = np.asarray(y2)
    if mean is not None and std is not None:
        pred_np = denormalize_fields(pred_np, mean, std)
        target_np = denormalize_fields(target_np, mean, std)
    return (
        float(jnp.sqrt(jnp.mean(err))),
        {name: float(value) for name, value in zip(channel_names, by_channel)},
        relative_l2_pct(pred_np, target_np),
        relative_l2_pct_by_channel(pred_np, target_np, channel_names),
    )


def direct_two_step(model, params, data, channel_names, mean=None, std=None):
    y2 = data["two_step_target"]
    batch = {k: v for k, v in data.items() if k != "two_step_target"}
    batch2 = set_time_span(batch, batch["t"], batch["dt"] * 2.0)
    pred = model.apply(params, batch2)
    err = (pred - y2) ** 2
    by_channel = np.sqrt(np.asarray(jnp.mean(err, axis=(0, 1, 2))))
    pred_np = np.asarray(pred)
    target_np = np.asarray(y2)
    if mean is not None and std is not None:
        pred_np = denormalize_fields(pred_np, mean, std)
        target_np = denormalize_fields(target_np, mean, std)
    return (
        float(jnp.sqrt(jnp.mean(err))),
        {name: float(value) for name, value in zip(channel_names, by_channel)},
        relative_l2_pct(pred_np, target_np),
        relative_l2_pct_by_channel(pred_np, target_np, channel_names),
    )


def semigroup_error(model, params, data, channel_names):
    direct = model.apply(params, set_time_span(data, data["t_ab"], data["dt_ab"]))
    first = model.apply(params, data)
    second_batch = dict(data)
    second_batch["x"] = first
    second_batch["state"] = state_features_jax(first)
    second_batch["t"] = data["t_b"]
    second_batch["dt"] = data["dt_b"]
    second = model.apply(params, second_batch)
    err = (direct - second) ** 2
    by_channel = np.sqrt(np.asarray(jnp.mean(err, axis=(0, 1, 2))))
    return (
        float(jnp.sqrt(jnp.mean(err))),
        {name: float(value) for name, value in zip(channel_names, by_channel)},
    )


def persistence_metrics(train, param_eval, time_eval, rollout_data, args, channel_names, mean, std):
    def rmse(data):
        return float(jnp.sqrt(jnp.mean((data["x"] - data["y"]) ** 2)))

    def rmse_by_channel(data):
        values = np.sqrt(np.asarray(jnp.mean((data["x"] - data["y"]) ** 2, axis=(0, 1, 2))))
        return {name: float(value) for name, value in zip(channel_names, values)}

    def rel_l2(data):
        pred = denormalize_fields(np.asarray(data["x"]), mean, std)
        target = denormalize_fields(np.asarray(data["y"]), mean, std)
        return relative_l2_pct(pred, target), relative_l2_pct_by_channel(pred, target, channel_names)

    start = time.perf_counter()
    for _ in range(args.speed_repeats):
        y = param_eval["x"]
    _ = np.asarray(y).shape
    speed = (time.perf_counter() - start) * 1000.0 / args.speed_repeats
    two_err = (rollout_data["x"] - rollout_data["two_step_target"]) ** 2
    two = float(jnp.sqrt(jnp.mean(two_err)))
    two_by_channel = np.sqrt(np.asarray(jnp.mean(two_err, axis=(0, 1, 2))))
    two_pred = denormalize_fields(np.asarray(rollout_data["x"]), mean, std)
    two_target = denormalize_fields(np.asarray(rollout_data["two_step_target"]), mean, std)
    train_rel, train_rel_by_channel = rel_l2(train)
    param_rel, param_rel_by_channel = rel_l2(param_eval)
    time_rel, time_rel_by_channel = rel_l2(time_eval)
    two_rel = relative_l2_pct(two_pred, two_target)
    two_rel_by_channel = relative_l2_pct_by_channel(
        two_pred,
        two_target,
        channel_names,
    )
    return MetricRow(
        rmse(train),
        rmse(param_eval),
        rmse(time_eval),
        two,
        speed,
        rmse_by_channel(train),
        rmse_by_channel(param_eval),
        rmse_by_channel(time_eval),
        {name: float(value) for name, value in zip(channel_names, two_by_channel)},
        None,
        None,
        None,
        None,
        None,
        param_rel,
        param_rel_by_channel,
        two_rel,
        two_rel_by_channel,
        train_rel,
        train_rel_by_channel,
        time_rel,
        time_rel_by_channel,
    )


PPDONET_CHANNEL_DIRS = {
    "log_sigma": "single_log_sigma",
    "v_r": "single_v_r",
    "v_theta": "single_v_theta",
}


def load_pretrained_ppdonet_predictions(raw_params, r, theta, channel_names):
    sys.path.insert(0, PPDONET_ROOT.as_posix())
    import onet_disk2D.run  # noqa: PLC0415

    name_to_col = {
        "alpha": 0,
        "aspectratio": 1,
        "aspect_ratio": 1,
        "planetmass": 2,
        "planet_mass": 2,
    }
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    coords = jnp.asarray(np.stack([rr, tt], axis=-1).reshape((-1, 2)), dtype=jnp.float32)
    jobs = {}
    predictions = []
    for channel in channel_names:
        run_dir = PPDONET_ROOT / "trained_network" / PPDONET_CHANNEL_DIRS[channel]
        job_args = onet_disk2D.run.load_job_args(
            run_dir,
            args_file="args.yml",
            arg_groups_file="arg_groups.yml",
            fargo_setup_file="fargo_setups.yml",
        )
        job = onet_disk2D.run.JOB(job_args)
        with contextlib.redirect_stdout(io.StringIO()):
            job.load_model(run_dir)
        cols = [name_to_col[p.lower()] for p in sorted(job_args["parameter"])]
        pred_fn = jax.jit(
            lambda u_batch, job=job: job.s_pred_fn(
                job.model.params,
                job.state,
                {"u_net": u_batch, "y_net": coords},
            )
        )
        u = jnp.asarray(raw_params[:, cols], dtype=jnp.float32)
        pred = pred_fn(u).block_until_ready()
        predictions.append(np.asarray(pred, dtype=np.float32).reshape((raw_params.shape[0], len(r), len(theta))))
        jobs[channel] = (pred_fn, cols)
    pred = np.stack(predictions, axis=-1).astype(np.float32)

    def speed(param_ids, repeats):
        batches = {
            channel: jnp.asarray(raw_params[param_ids][:, cols], dtype=jnp.float32)
            for channel, (_, cols) in jobs.items()
        }
        for channel, (pred_fn, _) in jobs.items():
            pred_fn(batches[channel]).block_until_ready()
        start = time.perf_counter()
        for _ in range(repeats):
            outputs = [pred_fn(batches[channel]) for channel, (pred_fn, _) in jobs.items()]
        for output in outputs:
            output.block_until_ready()
        return (time.perf_counter() - start) * 1000.0 / repeats

    return pred, speed


def pretrained_ppdonet_metrics(
    pred,
    speed_fn,
    fields,
    train_case_ids,
    heldout_case_ids,
    train_starts,
    heldout_time_starts,
    mean,
    std,
    args,
    channel_names,
):
    pred_norm = normalize_fields(pred, mean, std)

    def rmse(param_ids, start_ids):
        ys = []
        ps = []
        for p in param_ids:
            for sid in start_ids:
                ys.append(normalize_fields(fields[p, sid + 1], mean, std))
                ps.append(pred_norm[p])
        return float(np.sqrt(np.mean((np.stack(ps) - np.stack(ys)) ** 2)))

    def rmse_by_channel(param_ids, start_ids):
        ys = []
        ps = []
        for p in param_ids:
            for sid in start_ids:
                ys.append(normalize_fields(fields[p, sid + 1], mean, std))
                ps.append(pred_norm[p])
        values = np.sqrt(np.mean((np.stack(ps) - np.stack(ys)) ** 2, axis=(0, 1, 2)))
        return {name: float(value) for name, value in zip(channel_names, values)}

    def rel_l2(param_ids, start_ids):
        ys = []
        ps = []
        for p in param_ids:
            for sid in start_ids:
                ys.append(fields[p, sid + 1])
                ps.append(pred[p])
        return relative_l2_pct(np.stack(ps), np.stack(ys))

    def rel_l2_by_channel(param_ids, start_ids):
        ys = []
        ps = []
        for p in param_ids:
            for sid in start_ids:
                ys.append(fields[p, sid + 1])
                ps.append(pred[p])
        return relative_l2_pct_by_channel(np.stack(ps), np.stack(ys), channel_names)

    two_target = normalize_fields(fields[heldout_case_ids, -1], mean, std)
    two_pred = pred_norm[heldout_case_ids]
    two_values = np.sqrt(np.mean((two_pred - two_target) ** 2, axis=(0, 1, 2)))
    two_rel = relative_l2_pct(pred[heldout_case_ids], fields[heldout_case_ids, -1])
    two_rel_by_channel = relative_l2_pct_by_channel(pred[heldout_case_ids], fields[heldout_case_ids, -1], channel_names)
    param_eval_ids = np.repeat(heldout_case_ids, len(train_starts))
    return MetricRow(
        train_rmse=rmse(train_case_ids, train_starts),
        heldout_param_rmse=rmse(heldout_case_ids, train_starts),
        heldout_time_rmse=rmse(train_case_ids, heldout_time_starts),
        two_step_rollout_rmse=float(np.sqrt(np.mean((two_pred - two_target) ** 2))),
        speed_ms_per_batch=speed_fn(param_eval_ids, args.speed_repeats),
        train_rmse_by_channel=rmse_by_channel(train_case_ids, train_starts),
        heldout_param_rmse_by_channel=rmse_by_channel(heldout_case_ids, train_starts),
        heldout_time_rmse_by_channel=rmse_by_channel(train_case_ids, heldout_time_starts),
        two_step_rollout_rmse_by_channel={
            name: float(value) for name, value in zip(channel_names, two_values)
        },
        heldout_param_rel_l2_pct=rel_l2(heldout_case_ids, train_starts),
        heldout_param_rel_l2_pct_by_channel=rel_l2_by_channel(heldout_case_ids, train_starts),
        two_step_rollout_rel_l2_pct=two_rel,
        two_step_rollout_rel_l2_pct_by_channel=two_rel_by_channel,
        train_rel_l2_pct=rel_l2(train_case_ids, train_starts),
        train_rel_l2_pct_by_channel=rel_l2_by_channel(train_case_ids, train_starts),
        heldout_time_rel_l2_pct=rel_l2(train_case_ids, heldout_time_starts),
        heldout_time_rel_l2_pct_by_channel=rel_l2_by_channel(train_case_ids, heldout_time_starts),
    )


def predict_one_step(model, params, data):
    return np.asarray(model.apply(params, {k: v for k, v in data.items() if k != "two_step_target"}))


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size < 2:
        delta = 0.5
        return np.asarray([centers[0] - delta, centers[0] + delta], dtype=np.float32)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]]).astype(np.float32)


def polar_mesh(r: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_edges = cell_edges(r)
    theta_edges = cell_edges(theta)
    rr, tt = np.meshgrid(r_edges, theta_edges, indexing="ij")
    return rr * np.cos(tt), rr * np.sin(tt)


def plot_snapshot_comparison(
    fields: np.ndarray,
    predictions: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    channel_names: list[str],
    heldout_case_ids: np.ndarray,
    rollout_start: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    case_offset = 0
    case_id = int(heldout_case_ids[case_offset])
    input_field = fields[case_id, rollout_start]
    truth = fields[case_id, rollout_start + 1]
    denorm_predictions = {
        name: denormalize_fields(pred[[case_offset]], mean, std)[0]
        for name, pred in predictions.items()
    }
    xx, yy = polar_mesh(r, theta)

    for channel_id, channel in enumerate(channel_names):
        panels = [("input", input_field[..., channel_id]), ("ground truth", truth[..., channel_id])]
        panels.extend((name, pred[..., channel_id]) for name, pred in denorm_predictions.items())
        values = [panel[1] for panel in panels]
        vmin = min(float(np.nanpercentile(value, 2)) for value in values)
        vmax = max(float(np.nanpercentile(value, 98)) for value in values)
        err_values = [(name, value - truth[..., channel_id]) for name, value in panels[2:]]
        err_abs = max(float(np.nanpercentile(np.abs(err), 98)) for _, err in err_values)
        fig, axes = plt.subplots(2, len(panels), figsize=(2.7 * len(panels), 5.2), constrained_layout=True)
        for ax, (name, value) in zip(axes[0], panels):
            im = ax.pcolormesh(xx, yy, value, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[1, :2]:
            ax.axis("off")
        for ax, (name, err) in zip(axes[1, 2:], err_values):
            im_err = ax.pcolormesh(xx, yy, err, shading="auto", cmap="coolwarm", vmin=-err_abs, vmax=err_abs)
            ax.set_title(f"{name} error", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[0, :], shrink=0.75, location="right")
        fig.colorbar(im_err, ax=axes[1, 2:], shrink=0.75, location="right")
        fig.suptitle(f"Held-out case {case_id}, {channel}, frame {rollout_start} to {rollout_start + 1}", fontsize=11)
        fig.savefig(output_dir / f"snapshot_{channel}.png", dpi=180)
        plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields, params, raw_params, r, theta, times, meta = load_dataset(args.dataset, args.channels)
    coords = coordinate_grid(r, theta)
    output_channels = len(args.channels)

    n_cases = fields.shape[0]
    raw_heldout_cases = args.heldout_cases if args.heldout_cases is not None else [args.heldout_case]
    heldout_cases = [case if case >= 0 else n_cases + case for case in raw_heldout_cases]
    if any(case < 0 or case >= n_cases for case in heldout_cases):
        raise ValueError(f"heldout cases {raw_heldout_cases} are outside 0..{n_cases - 1}")
    heldout_case_ids = np.asarray(sorted(set(heldout_cases)))
    train_case_ids = np.asarray([i for i in range(n_cases) if i not in set(heldout_case_ids.tolist())])
    if train_case_ids.size == 0:
        raise ValueError("At least one training case is required")
    max_required_frames = max(4, max(args.train_spans) + 2, sum(args.consistency_spans) + 2)
    if fields.shape[1] < max_required_frames:
        raise ValueError(f"At least {max_required_frames} frames are required for the requested spans")
    heldout_time_index = fields.shape[1] - 1
    train_starts = np.arange(0, fields.shape[1] - 1)
    one_step_train_starts = np.arange(0, fields.shape[1] - 2)
    heldout_time_starts = np.asarray([fields.shape[1] - 2])

    train_window_fields = fields[train_case_ids][:, train_starts]
    mean = train_window_fields.mean(axis=(0, 1, 2, 3)).astype(np.float32)
    std = (train_window_fields.std(axis=(0, 1, 2, 3)) + 1.0e-6).astype(np.float32)

    step_train = make_step_data(
        fields,
        params,
        times,
        train_case_ids,
        train_starts,
        mean,
        std,
        spans=args.train_spans,
        max_target_index=heldout_time_index - 1,
    )
    rollout_train_starts = one_step_train_starts[one_step_train_starts + 2 <= heldout_time_index - 1]
    rollout_train = add_two_step_target(
        make_step_data(fields, params, times, train_case_ids, rollout_train_starts, mean, std, spans=(1,)),
        fields,
        train_case_ids,
        rollout_train_starts,
        mean,
        std,
    )
    consistency_train = None
    consistency_eval = make_semigroup_data(
        fields,
        params,
        times,
        heldout_case_ids,
        np.asarray([fields.shape[1] - sum(args.consistency_spans) - 1]),
        args.consistency_spans[0],
        args.consistency_spans[1],
        mean,
        std,
    )
    if args.consistency_weight > 0.0:
        consistency_starts = np.arange(0, fields.shape[1] - 1 - sum(args.consistency_spans))
        consistency_train = make_semigroup_data(
            fields,
            params,
            times,
            train_case_ids,
            consistency_starts,
            args.consistency_spans[0],
            args.consistency_spans[1],
            mean,
            std,
        )
    step_param = make_step_data(fields, params, times, heldout_case_ids, one_step_train_starts, mean, std, spans=(1,))
    step_time = make_step_data(fields, params, times, train_case_ids, heldout_time_starts, mean, std, spans=(1,))
    rollout_start = fields.shape[1] - 3
    rollout_data = make_step_data(fields, params, times, heldout_case_ids, np.asarray([rollout_start]), mean, std)
    two_target = normalize_fields(fields[heldout_case_ids, -1], mean, std)
    rollout_data["two_step_target"] = jnp.asarray(two_target)

    model_specs = {
        "time_deeponet": (
            "time_deeponet_param_t_to_next_frame",
            make_param_deeponet(
                coords,
                args.latent,
                args.width,
                output_channels,
                use_time=True,
                use_state=False,
                residual=False,
            ),
            step_train,
            step_param,
            step_time,
            None,
        ),
        "state_deeponet": (
            "state_conditioned_deeponet_step",
            make_param_deeponet(
                coords,
                args.latent,
                args.width,
                output_channels,
                use_time=True,
                use_state=True,
                residual=True,
            ),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
        "pointwise": (
            "pointwise_residual_step",
            make_pointwise(coords, args.width, args.depth, output_channels),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
        "fno": (
            "fno_residual_step",
            make_fno(coords, args.width, args.modes_r, args.modes_theta, args.depth, output_channels),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
        "fno_flow": (
            "fno_flow_map_residual" if args.flow_residual_output else "fno_flow_map",
            make_fno(
                coords,
                args.width,
                args.modes_r,
                args.modes_theta,
                args.depth,
                output_channels,
                residual=args.flow_residual_output,
            ),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
    }

    results = {
        "persistence_xn_as_xnp1": asdict(
            persistence_metrics(step_train, step_param, step_time, rollout_data, args, args.channels, mean, std)
        )
    }
    plot_predictions = {
        "persistence": np.asarray(rollout_data["x"]),
    }
    if not args.no_pretrained_ppdonet:
        ppdonet_pred, ppdonet_speed = load_pretrained_ppdonet_predictions(raw_params, r, theta, args.channels)
        plot_predictions["ppdonet_steady"] = normalize_fields(ppdonet_pred[heldout_case_ids], mean, std)
        results["pretrained_ppdonet_ss_time_independent"] = asdict(
            pretrained_ppdonet_metrics(
                ppdonet_pred,
                ppdonet_speed,
                fields,
                train_case_ids,
                heldout_case_ids,
                train_starts,
                heldout_time_starts,
                mean,
                std,
                args,
                args.channels,
            )
        )
    for i, key in enumerate(args.models):
        name, model, train, param_eval, time_eval, rollout_eval = model_specs[key]
        print(f"Training {name}...")
        use_rollout_train = rollout_eval is not None and args.rollout_weight > 0.0
        trained_params, rmse, rmse_by_channel, rel_l2_pct, speed, train_info = train_and_eval(
            model,
            train,
            {},
            args,
            args.channels,
            seed_offset=10 * i,
            rollout_train=rollout_train if use_rollout_train else None,
            consistency_train=consistency_train if key in {"state_deeponet", "pointwise", "fno", "fno_flow"} else None,
            use_identity_loss=key in {"state_deeponet", "pointwise", "fno", "fno_flow"},
            mean=mean,
            std=std,
        )
        two_step = None
        two_step_by_channel = None
        two_step_rel = None
        two_step_rel_by_channel = None
        direct_two = None
        direct_two_by_channel = None
        direct_two_rel = None
        direct_two_rel_by_channel = None
        sg = None
        sg_by_channel = None
        if rollout_eval is not None:
            two_step, two_step_by_channel, two_step_rel, two_step_rel_by_channel = rollout_two_steps(
                model,
                trained_params,
                rollout_eval,
                args.channels,
                mean,
                std,
            )
            if key == "fno_flow":
                direct_two, direct_two_by_channel, direct_two_rel, direct_two_rel_by_channel = direct_two_step(
                    model,
                    trained_params,
                    rollout_eval,
                    args.channels,
                    mean,
                    std,
                )
            plot_predictions[key] = predict_one_step(model, trained_params, rollout_eval)
        elif key == "time_deeponet":
            plot_predictions[key] = predict_one_step(model, trained_params, rollout_data)
        param_rel, param_rel_by_channel = rel_l2_pct(param_eval)
        train_rel, train_rel_by_channel = rel_l2_pct(train)
        time_rel, time_rel_by_channel = rel_l2_pct(time_eval)
        if key in {"state_deeponet", "pointwise", "fno", "fno_flow"}:
            sg, sg_by_channel = semigroup_error(model, trained_params, consistency_eval, args.channels)
        results[name] = asdict(
            MetricRow(
                train_rmse=rmse(train),
                heldout_param_rmse=rmse(param_eval),
                heldout_time_rmse=rmse(time_eval),
                two_step_rollout_rmse=two_step,
                speed_ms_per_batch=speed(param_eval),
                train_rmse_by_channel=rmse_by_channel(train),
                heldout_param_rmse_by_channel=rmse_by_channel(param_eval),
                heldout_time_rmse_by_channel=rmse_by_channel(time_eval),
                two_step_rollout_rmse_by_channel=two_step_by_channel,
                semigroup_rmse=sg,
                semigroup_rmse_by_channel=sg_by_channel,
                trained_steps=train_info["trained_steps"],
                best_step=train_info["best_step"],
                best_monitor_rmse=train_info["best_monitor_rmse"],
                heldout_param_rel_l2_pct=param_rel,
                heldout_param_rel_l2_pct_by_channel=param_rel_by_channel,
                two_step_rollout_rel_l2_pct=two_step_rel,
                two_step_rollout_rel_l2_pct_by_channel=two_step_rel_by_channel,
                train_rel_l2_pct=train_rel,
                train_rel_l2_pct_by_channel=train_rel_by_channel,
                heldout_time_rel_l2_pct=time_rel,
                heldout_time_rel_l2_pct_by_channel=time_rel_by_channel,
                direct_two_step_rmse=direct_two,
                direct_two_step_rmse_by_channel=direct_two_by_channel,
                direct_two_step_rel_l2_pct=direct_two_rel,
                direct_two_step_rel_l2_pct_by_channel=direct_two_rel_by_channel,
            )
        )

    if args.save_plots:
        plot_snapshot_comparison(
            fields,
            plot_predictions,
            mean,
            std,
            r,
            theta,
            args.channels,
            heldout_case_ids,
            rollout_start,
            args.output_dir,
        )

    summary = {
        "setup": {
            "dataset": args.dataset.as_posix(),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "latent": args.latent,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "train_case_ids": train_case_ids.tolist(),
            "heldout_case_ids": heldout_case_ids.tolist(),
            "channels": args.channels,
            "normalization_mean": {name: float(value) for name, value in zip(args.channels, mean)},
            "normalization_std": {name: float(value) for name, value in zip(args.channels, std)},
            "rollout_weight": args.rollout_weight,
            "train_spans": args.train_spans,
            "consistency_weight": args.consistency_weight,
            "consistency_spans": args.consistency_spans,
            "identity_weight": args.identity_weight,
            "loss_channel_weights": args.loss_channel_weights,
            "flow_residual_output": args.flow_residual_output,
            "grad_clip_norm": args.grad_clip_norm,
            "eval_every": args.eval_every,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "metric_units": "RMSE after per-channel standardization on the training windows",
            "primary_metric": (
                "Mean per-sample physical-unit relative L2 percentage error, computed after undoing "
                "per-channel standardization: mean_i 100 * ||prediction_i - target_i||_2 / ||target_i||_2."
            ),
            "note": (
                f"Tiny FARGO3D sanity benchmark: {fields.shape[0]} cases, "
                f"{fields.shape[1]} frames, {fields.shape[2]}x{fields.shape[3]} grid, "
                f"{output_channels} channels. "
                "Held-out parameter cases are excluded from training; held-out time is "
                "the last one-step window on train cases."
            ),
            "dataset_meta": meta,
        },
        "results": results,
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()

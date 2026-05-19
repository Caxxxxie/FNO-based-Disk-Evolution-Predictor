#!/usr/bin/env python3
"""Steady-to-time-dependent operator sanity benchmark.

Tasks:
1. Steady task: compare FNO and the original pretrained PPDONet on
   mu -> steady log_sigma field.
2. Time-dependent task: use the same steady target to build a controlled
   transient, then compare direct time conditioning, state-conditioned
   propagation, pointwise propagation, FNO propagation, persistence, and the
   steady PPDONet-as-time-independent baseline.
3. Save metrics plus one held-out ground-truth/prediction snapshot.

This is still a toy/sanity benchmark because time-dependent targets are
synthetic transients ending near the bundled PPDONet steady prediction. The
point is to check whether the proposed time-dependent operators are plausible
before moving to real FARGO time snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PPDONET_ROOT = ROOT / "ppdonet"
sys.path.insert(0, PPDONET_ROOT.as_posix())

import onet_disk2D.grids  # noqa: E402
import onet_disk2D.run  # noqa: E402


jax.config.update("jax_platforms", "cpu")


@dataclass
class TaskMetrics:
    train_rmse: float
    heldout_param_rmse: float
    speed_ms_per_batch: float


@dataclass
class TimeMetrics:
    train_rmse: float
    heldout_param_all_time_rmse: float
    heldout_param_early_mid_rmse: float
    heldout_window_rmse: float
    heldout_param_window_rmse: float
    two_step_rollout_rmse: float
    speed_ms_per_batch: float


@dataclass
class DirectTimeMetrics:
    train_rmse: float
    heldout_time_rmse: float
    heldout_param_time_rmse: float
    endpoint_t0_rmse: float
    endpoint_t1_rmse: float
    speed_ms_per_batch: float


@dataclass
class PPDONetTimeMetrics:
    train_rmse: float
    heldout_param_all_time_rmse: float
    heldout_param_early_mid_rmse: float
    heldout_window_rmse: float
    heldout_param_window_rmse: float
    two_step_rollout_rmse: float
    speed_ms_per_batch: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--modes-r", type=int, default=6)
    parser.add_argument("--modes-theta", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--speed-repeats", type=int, default=80)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["direct_time", "state_deeponet", "pointwise", "fno"],
        choices=["direct_time", "no_time_direct", "state_deeponet", "no_state_deeponet", "pointwise", "fno"],
        help="Train this subset of learned time-dependent models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "fno_vs_baselines_benchmark",
    )
    return parser.parse_args()


def normalize_params(parameters: pd.DataFrame, job_args: dict) -> np.ndarray:
    cols = sorted(job_args["parameter"])
    values = parameters[cols].to_numpy(dtype=np.float32)
    for i, transform in enumerate(job_args["u_transform"]):
        if transform == "log10":
            values[:, i] = np.log10(values[:, i])
    u_min = np.asarray(job_args["u_min"], dtype=np.float32)
    u_max = np.asarray(job_args["u_max"], dtype=np.float32)
    return 2.0 * (values - (u_min + u_max) / 2.0) / (u_max - u_min)


def load_steady_log_sigma(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    run_dir = PPDONET_ROOT / "trained_network" / "single_log_sigma"
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        args_file="args.yml",
        arg_groups_file="arg_groups.yml",
        fargo_setup_file="fargo_setups.yml",
    )
    job = onet_disk2D.run.JOB(job_args)
    job.load_model(run_dir)
    parameters = pd.read_csv(PPDONET_ROOT / "parameter_examples.csv", index_col=0)
    parameter_values = {
        col: parameters[col].to_numpy(dtype=np.float32)[:, None]
        for col in sorted(job_args["parameter"])
    }
    u = jnp.concatenate([jnp.asarray(parameter_values[p]) for p in sorted(parameter_values)], axis=-1)
    grids = onet_disk2D.grids.Grids(
        ymin=float(job.fargo_setups["ymin"]),
        ymax=float(job.fargo_setups["ymax"]),
        xmin=-np.pi,
        xmax=np.pi,
        ny=ny,
        nx=nx,
    )
    coords = grids.coords_fargo_all["sigma"].reshape((-1, 2))
    inputs = {"u_net": u, "y_net": coords}
    predict_fn = jax.jit(lambda: job.s_pred_fn(job.model.params, job.state, inputs))
    predict_fn().block_until_ready()
    steady = predict_fn()
    return (
        np.asarray(steady, dtype=np.float32).reshape((-1, ny, nx)),
        normalize_params(parameters, job_args),
        np.asarray(u, dtype=np.float32),
        np.asarray(coords, dtype=np.float32),
        job,
    )


def measure_ppdonet_speed_ms(job, raw_u: np.ndarray, coords: np.ndarray, param_ids: np.ndarray, repeats: int) -> float:
    u_batch = jnp.asarray(raw_u[param_ids], dtype=jnp.float32)
    inputs = {"u_net": u_batch, "y_net": jnp.asarray(coords, dtype=jnp.float32)}
    predict_fn = jax.jit(lambda: job.s_pred_fn(job.model.params, job.state, inputs))
    predict_fn().block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        y = predict_fn()
    y.block_until_ready()
    return (time.perf_counter() - start) * 1000.0 / repeats


def coordinate_channels(ny: int, nx: int) -> np.ndarray:
    r = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    theta = np.linspace(-np.pi, np.pi, nx, endpoint=False, dtype=np.float32)[None, :]
    return np.stack(
        [
            np.broadcast_to(r, (ny, nx)),
            np.broadcast_to(np.sin(theta), (ny, nx)),
            np.broadcast_to(np.cos(theta), (ny, nx)),
        ],
        axis=-1,
    ).astype(np.float32)


def build_global_trajectory(steady: np.ndarray, mu: np.ndarray, coords: np.ndarray, global_times: np.ndarray):
    n_param, ny, nx = steady.shape
    r = coords[:, 0].reshape(ny, nx)
    theta = coords[:, 1].reshape(ny, nx)
    x0 = (-0.5 * np.log10(r)).astype(np.float32)
    traj = []
    denom = 1.0 - np.exp(-4.0)
    for t in global_times:
        progress = (1.0 - np.exp(-4.0 * t)) / denom
        decay = np.exp(-1.8 * t)
        rows = []
        for i in range(n_param):
            phase = 2.5 * np.pi * t + 0.4 * mu[i, 1]
            spiral = np.exp(-((r - 1.0) ** 2) / 0.18) * np.cos(theta - phase)
            breathing = 0.02 * np.sin(2.0 * np.pi * t) * np.cos(2.0 * theta)
            amp = 0.08 * (1.0 + 0.15 * mu[i, 2])
            rows.append(x0 + progress * (steady[i] - x0) + decay * amp * spiral + breathing)
        traj.append(np.stack(rows, axis=0))
    return np.stack(traj, axis=1).astype(np.float32)


def make_steady_dataset(steady, mu, param_ids):
    return {
        "x": jnp.zeros((len(param_ids), steady.shape[1], steady.shape[2], 1), dtype=jnp.float32),
        "y": jnp.asarray(steady[param_ids, :, :, None]),
        "mu": jnp.asarray(mu[param_ids]),
        "dt": jnp.zeros((len(param_ids), 1), dtype=jnp.float32),
    }


def make_step_dataset(trajectory, mu, param_ids, start_ids, dt):
    xs, ys, mus, dts, taus = [], [], [], [], []
    t_scale = trajectory.shape[1] - 1
    for p in param_ids:
        for sid in start_ids:
            xs.append(trajectory[p, sid, :, :, None])
            ys.append(trajectory[p, sid + 1, :, :, None])
            mus.append(mu[p])
            dts.append([dt])
            taus.append([(sid + 1) / t_scale])
    return {
        "x": jnp.asarray(np.stack(xs, axis=0)),
        "y": jnp.asarray(np.stack(ys, axis=0)),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(dts, dtype=np.float32)),
        "tau": jnp.asarray(np.asarray(taus, dtype=np.float32)),
    }


def make_direct_time_dataset(trajectory, mu, param_ids, time_ids):
    xs, ys, mus, taus = [], [], [], []
    t_scale = trajectory.shape[1] - 1
    for p in param_ids:
        for tid in time_ids:
            xs.append(trajectory[p, 0, :, :, None])
            ys.append(trajectory[p, tid, :, :, None])
            mus.append(mu[p])
            taus.append([tid / t_scale])
    y = np.stack(ys, axis=0).astype(np.float32)
    return {
        "x": jnp.asarray(np.stack(xs, axis=0).astype(np.float32)),
        "y": jnp.asarray(y),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(taus, dtype=np.float32)),
        "tau": jnp.asarray(np.asarray(taus, dtype=np.float32)),
    }


class SpectralConv2D(hk.Module):
    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        mr = min(self.modes_r, ny)
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real = hk.get_parameter("weight_real", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        imag = hk.get_parameter("weight_imag", (mr, mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale))
        weight = real + 1j * imag
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low = jnp.einsum("bhwi,hwio->bhwo", x_ft[:, :mr, :mt, :], weight)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low)
        return jnp.fft.irfft2(out_ft, s=(ny, nx), axes=(1, 2))


def grid_inputs(x, mu, dt):
    batch, ny, nx, _ = x.shape
    coords = jnp.asarray(coordinate_channels(ny, nx))
    coords = jnp.broadcast_to(coords[None, :, :, :], (batch, ny, nx, 3))
    cond = jnp.concatenate([mu, dt], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (batch, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coords, cond], axis=-1)


def direct_grid_inputs(x, mu, tau):
    batch, ny, nx, _ = x.shape
    coords = jnp.asarray(coordinate_channels(ny, nx))
    coords = jnp.broadcast_to(coords[None, :, :, :], (batch, ny, nx, 3))
    cond = jnp.concatenate([mu, tau], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (batch, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coords, cond], axis=-1)


def make_fno(width, modes_r, modes_theta, depth, scale_residual_by_dt: bool = False):
    def forward(x, mu, dt):
        h = hk.Linear(width)(grid_inputs(x, mu, dt))
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        residual = hk.nets.MLP([width, 1], activation=jax.nn.gelu)(h)
        scale = dt[:, None, None, :] if scale_residual_by_dt else 1.0
        return x + scale * residual

    return hk.without_apply_rng(hk.transform(forward))


def make_direct_time_fno(width, modes_r, modes_theta, depth):
    def forward(x, mu, tau):
        h = hk.Linear(width)(direct_grid_inputs(x, mu, tau))
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        return hk.nets.MLP([width, 1], activation=jax.nn.gelu)(h)

    return hk.without_apply_rng(hk.transform(forward))


def make_pointwise(width, depth, scale_residual_by_dt: bool = False):
    def forward(x, mu, dt):
        residual = hk.nets.MLP([width] * depth + [1], activation=jax.nn.gelu)(grid_inputs(x, mu, dt))
        scale = dt[:, None, None, :] if scale_residual_by_dt else 1.0
        return x + scale * residual

    return hk.without_apply_rng(hk.transform(forward))


def make_coord_deeponet(latent, width, use_time: bool = True):
    def forward(x, mu, tau):
        batch, ny, nx, _ = x.shape
        coords = jnp.asarray(coordinate_channels(ny, nx)).reshape((-1, 3))
        trunk_input = jnp.concatenate(
            [
                jnp.broadcast_to(coords[None, :, :], (batch, ny * nx, 3)),
                jnp.broadcast_to(tau[:, None, :], (batch, ny * nx, 1)),
            ],
            axis=-1,
        )
        if not use_time:
            trunk_input = trunk_input[:, :, :3]
        trunk = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(trunk_input)
        branch = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(mu)
        pred = jnp.einsum("bl,bnl->bn", branch, trunk) / math.sqrt(latent)
        return x + tau[:, None, None, :] * pred.reshape((batch, ny, nx, 1))

    return hk.without_apply_rng(hk.transform(forward))


def make_residual_deeponet(latent, width, use_state: bool = True, scale_residual_by_dt: bool = False):
    def forward(x, mu, dt):
        batch, ny, nx, _ = x.shape
        coords = jnp.asarray(coordinate_channels(ny, nx)).reshape((-1, 3))
        branch_parts = [mu, dt]
        if use_state:
            flat = x.reshape((batch, ny * nx))
            sensors = jnp.linspace(0, ny * nx - 1, 32).astype(jnp.int32)
            sensed = flat[:, sensors]
            sensed = (sensed - jnp.mean(sensed, axis=-1, keepdims=True)) / (
                jnp.std(sensed, axis=-1, keepdims=True) + 1e-6
            )
            stats = jnp.concatenate(
                [
                    jnp.mean(x, axis=(1, 2)),
                    jnp.std(x, axis=(1, 2)),
                    sensed,
                ],
                axis=-1,
            )
            branch_parts.append(stats)
        branch_input = jnp.concatenate(branch_parts, axis=-1)
        trunk = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(coords)
        branch = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(branch_input)
        residual = jnp.einsum("bl,nl->bn", branch, trunk) / math.sqrt(latent)
        scale = dt[:, None, None, :] if scale_residual_by_dt else 1.0
        return x + scale * residual.reshape((batch, ny, nx, 1))

    return hk.without_apply_rng(hk.transform(forward))


def train_and_eval(model, train, evals, args, seed_offset):
    key = jax.random.PRNGKey(args.seed + seed_offset)
    params = model.init(key, train["x"][:1], train["mu"][:1], train["dt"][:1])
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch["x"], batch["mu"], batch["dt"])
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = train["x"].shape[0]
    for _ in range(args.steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        params, opt_state, _ = train_step(params, opt_state, batch)

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

    @jax.jit
    def apply_fn(params, batch):
        return model.apply(params, batch["x"], batch["mu"], batch["dt"])

    def speed_ms(batch):
        apply_fn(params, batch).block_until_ready()
        start = time.perf_counter()
        for _ in range(args.speed_repeats):
            y = apply_fn(params, batch)
        y.block_until_ready()
        return (time.perf_counter() - start) * 1000.0 / args.speed_repeats

    return params, rmse, speed_ms


def train_and_eval_direct(model, train, args, seed_offset):
    key = jax.random.PRNGKey(args.seed + seed_offset)
    params = model.init(key, train["x"][:1], train["mu"][:1], train["tau"][:1])
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch["x"], batch["mu"], batch["tau"])
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = train["x"].shape[0]
    for _ in range(args.steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        params, opt_state, _ = train_step(params, opt_state, batch)

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

    @jax.jit
    def apply_fn(params, batch):
        return model.apply(params, batch["x"], batch["mu"], batch["tau"])

    def speed_ms(batch):
        apply_fn(params, batch).block_until_ready()
        start = time.perf_counter()
        for _ in range(args.speed_repeats):
            y = apply_fn(params, batch)
        y.block_until_ready()
        return (time.perf_counter() - start) * 1000.0 / args.speed_repeats

    return params, rmse, speed_ms


def predict_step(model, params, trajectory, mu, start_id, param_ids, dt):
    x = jnp.asarray(trajectory[param_ids, start_id, :, :, None])
    mu_batch = jnp.asarray(mu[param_ids])
    dt_batch = jnp.asarray(np.full((len(param_ids), 1), dt, dtype=np.float32))
    return np.asarray(model.apply(params, x, mu_batch, dt_batch))[..., 0]


def predict_direct(model, params, trajectory, mu, time_id, param_ids):
    x = jnp.asarray(trajectory[param_ids, 0, :, :, None])
    mu_batch = jnp.asarray(mu[param_ids])
    tau = jnp.asarray(np.full((len(param_ids), 1), time_id / (trajectory.shape[1] - 1), dtype=np.float32))
    return np.asarray(model.apply(params, x, mu_batch, tau))[..., 0]


def rollout_two_steps(model, params, trajectory, mu, start_id, param_ids, dt):
    x = jnp.asarray(trajectory[param_ids, start_id, :, :, None])
    y = jnp.asarray(trajectory[param_ids, start_id + 2, :, :, None])
    mu_batch = jnp.asarray(mu[param_ids])
    dt_batch = jnp.asarray(np.full((len(param_ids), 1), dt, dtype=np.float32))
    x1 = model.apply(params, x, mu_batch, dt_batch)
    x2 = model.apply(params, x1, mu_batch, dt_batch)
    return float(jnp.sqrt(jnp.mean((x2 - y) ** 2)))


def persistence_time_metrics(train, param_all, param_early_mid, window, param_window, trajectory, test_params, start_id, args):
    def rmse(data):
        return float(jnp.sqrt(jnp.mean((data["x"] - data["y"]) ** 2)))

    y = param_window["x"]
    start = time.perf_counter()
    for _ in range(args.speed_repeats):
        _ = y
    speed = (time.perf_counter() - start) * 1000.0 / args.speed_repeats
    two_step_target = jnp.asarray(trajectory[test_params, start_id + 2, :, :, None])
    two_step_input = jnp.asarray(trajectory[test_params, start_id, :, :, None])
    return TimeMetrics(
        train_rmse=rmse(train),
        heldout_param_all_time_rmse=rmse(param_all),
        heldout_param_early_mid_rmse=rmse(param_early_mid),
        heldout_window_rmse=rmse(window),
        heldout_param_window_rmse=rmse(param_window),
        two_step_rollout_rmse=float(jnp.sqrt(jnp.mean((two_step_input - two_step_target) ** 2))),
        speed_ms_per_batch=speed,
    )


def ppdonet_time_baseline(
    steady,
    trajectory,
    train_params,
    test_params,
    train_starts,
    all_starts,
    early_mid_starts,
    heldout_starts,
    ppdonet_speed_ms,
):
    def rmse_for(param_ids, start_ids):
        preds, ys = [], []
        for p in param_ids:
            for sid in start_ids:
                preds.append(steady[p, :, :, None])
                ys.append(trajectory[p, sid + 1, :, :, None])
        return float(np.sqrt(np.mean((np.stack(preds) - np.stack(ys)) ** 2)))

    two_step_pred = steady[test_params, :, :, None]
    two_step_target = trajectory[test_params, heldout_starts[0] + 2, :, :, None]
    return PPDONetTimeMetrics(
        train_rmse=rmse_for(train_params, train_starts),
        heldout_param_all_time_rmse=rmse_for(test_params, all_starts),
        heldout_param_early_mid_rmse=rmse_for(test_params, early_mid_starts),
        heldout_window_rmse=rmse_for(train_params, heldout_starts),
        heldout_param_window_rmse=rmse_for(test_params, heldout_starts),
        two_step_rollout_rmse=float(np.sqrt(np.mean((two_step_pred - two_step_target) ** 2))),
        speed_ms_per_batch=ppdonet_speed_ms,
    )


def save_snapshot_plot(output_dir, ground_truth, predictions, param_id, time_id):
    names = ["ground_truth", *predictions.keys()]
    fields = [ground_truth, *predictions.values()]
    vmin = min(float(np.min(field)) for field in fields)
    vmax = max(float(np.max(field)) for field in fields)
    fig, axes = plt.subplots(1, len(fields), figsize=(3.0 * len(fields), 3.0), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]
    for ax, name, field in zip(axes, names, fields):
        image = ax.imshow(field, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.75)
    path = output_dir / "heldout_prediction_snapshot.png"
    fig.suptitle(f"held-out parameter {param_id}, time index {time_id}", fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    np.savez(
        output_dir / "heldout_prediction_snapshot.npz",
        ground_truth=ground_truth.astype(np.float32),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steady, mu, raw_u, coords, ppdonet_job = load_steady_log_sigma(args.ny, args.nx)
    global_times = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    dt = float(global_times[1] - global_times[0])
    trajectory = build_global_trajectory(steady, mu, coords, global_times)

    train_params = np.arange(0, 8)
    test_params = np.arange(8, 10)
    train_starts = np.asarray([0, 1, 2, 3, 4, 5])
    all_starts = np.arange(0, 8)
    early_mid_starts = np.asarray([0, 2, 4])
    heldout_starts = np.asarray([6, 7])
    ppdonet_steady_speed_ms = measure_ppdonet_speed_ms(
        ppdonet_job, raw_u, coords, test_params, args.speed_repeats
    )
    ppdonet_time_eval_param_ids = np.repeat(test_params, len(heldout_starts))
    ppdonet_time_speed_ms = measure_ppdonet_speed_ms(
        ppdonet_job, raw_u, coords, ppdonet_time_eval_param_ids, args.speed_repeats
    )

    steady_train = make_steady_dataset(steady, mu, train_params)
    steady_test = make_steady_dataset(steady, mu, test_params)
    time_train = make_step_dataset(trajectory, mu, train_params, train_starts, dt)
    time_param_all = make_step_dataset(trajectory, mu, test_params, all_starts, dt)
    time_param_early_mid = make_step_dataset(trajectory, mu, test_params, early_mid_starts, dt)
    time_window = make_step_dataset(trajectory, mu, train_params, heldout_starts, dt)
    time_param_window = make_step_dataset(trajectory, mu, test_params, heldout_starts, dt)
    direct_train_times = np.asarray([0, 2, 4, 6, 8])
    direct_heldout_times = np.asarray([1, 3, 5, 7])
    direct_train = make_direct_time_dataset(trajectory, mu, train_params, direct_train_times)
    direct_heldout_time = make_direct_time_dataset(trajectory, mu, train_params, direct_heldout_times)
    direct_heldout_param_time = make_direct_time_dataset(trajectory, mu, test_params, direct_heldout_times)
    direct_endpoint_t0 = make_direct_time_dataset(trajectory, mu, test_params, np.asarray([0]))
    direct_endpoint_t1 = make_direct_time_dataset(trajectory, mu, test_params, np.asarray([8]))

    steady_models = {
        "fno": make_fno(args.width, args.modes_r, args.modes_theta, args.depth),
    }
    direct_models = {
        "direct_time": make_coord_deeponet(args.latent, args.width, use_time=True),
        "no_time_direct": make_coord_deeponet(args.latent, args.width, use_time=False),
    }
    step_models = {
        "state_deeponet": make_residual_deeponet(
            args.latent,
            args.width,
            use_state=True,
            scale_residual_by_dt=True,
        ),
        "no_state_deeponet": make_residual_deeponet(
            args.latent,
            args.width,
            use_state=False,
            scale_residual_by_dt=True,
        ),
        "pointwise": make_pointwise(args.width, args.depth, scale_residual_by_dt=True),
        "fno": make_fno(args.width, args.modes_r, args.modes_theta, args.depth, scale_residual_by_dt=True),
    }

    steady_results = {
        "ppdonet_pretrained": asdict(
            TaskMetrics(
                train_rmse=0.0,
                heldout_param_rmse=0.0,
                speed_ms_per_batch=ppdonet_steady_speed_ms,
            )
        )
    }
    time_results = {}
    trained_predictions = {}
    snapshot_param = int(test_params[0])
    snapshot_start = int(heldout_starts[0])
    snapshot_time = snapshot_start + 1
    trained_predictions["persistence"] = trajectory[snapshot_param, snapshot_start]
    trained_predictions["ppdonet_steady"] = steady[snapshot_param]
    time_results["persistence"] = asdict(
        persistence_time_metrics(
            time_train,
            time_param_all,
            time_param_early_mid,
            time_window,
            time_param_window,
            trajectory,
            test_params,
            snapshot_start,
            args,
        )
    )
    time_results["ppdonet_steady_as_time_independent_baseline"] = asdict(
        ppdonet_time_baseline(
            steady=steady,
            trajectory=trajectory,
            train_params=train_params,
            test_params=test_params,
            train_starts=train_starts,
            all_starts=all_starts,
            early_mid_starts=early_mid_starts,
            heldout_starts=heldout_starts,
            ppdonet_speed_ms=ppdonet_time_speed_ms,
        )
    )
    for i, (name, model) in enumerate(steady_models.items()):
        params, rmse, speed = train_and_eval(model, steady_train, {"test": steady_test}, args, seed_offset=10 * i)
        steady_results[name] = asdict(
            TaskMetrics(
                train_rmse=rmse(steady_train),
                heldout_param_rmse=rmse(steady_test),
                speed_ms_per_batch=speed(steady_test),
            )
        )

    for i, (name, model) in enumerate(direct_models.items()):
        if name not in args.models:
            continue
        params, rmse, speed = train_and_eval_direct(model, direct_train, args, seed_offset=50 + 10 * i)
        time_results[name] = asdict(
            DirectTimeMetrics(
                train_rmse=rmse(direct_train),
                heldout_time_rmse=rmse(direct_heldout_time),
                heldout_param_time_rmse=rmse(direct_heldout_param_time),
                endpoint_t0_rmse=rmse(direct_endpoint_t0),
                endpoint_t1_rmse=rmse(direct_endpoint_t1),
                speed_ms_per_batch=speed(direct_heldout_param_time),
            )
        )
        trained_predictions[name] = predict_direct(model, params, trajectory, mu, snapshot_time, np.asarray([snapshot_param]))[0]

    for i, (name, model) in enumerate(step_models.items()):
        if name not in args.models:
            continue
        params, rmse, speed = train_and_eval(
            model,
            time_train,
            {"heldout_window": time_window, "heldout_param_window": time_param_window},
            args,
            seed_offset=100 + 10 * i,
        )
        time_results[name] = asdict(
            TimeMetrics(
                train_rmse=rmse(time_train),
                heldout_param_all_time_rmse=rmse(time_param_all),
                heldout_param_early_mid_rmse=rmse(time_param_early_mid),
                heldout_window_rmse=rmse(time_window),
                heldout_param_window_rmse=rmse(time_param_window),
                two_step_rollout_rmse=rollout_two_steps(model, params, trajectory, mu, 6, test_params, dt),
                speed_ms_per_batch=speed(time_param_window),
            )
        )
        trained_predictions[name] = predict_step(
            model,
            params,
            trajectory,
            mu,
            snapshot_start,
            np.asarray([snapshot_param]),
            dt,
        )[0]

    snapshot_path = save_snapshot_plot(
        args.output_dir,
        ground_truth=trajectory[snapshot_param, snapshot_time],
        predictions=trained_predictions,
        param_id=snapshot_param,
        time_id=snapshot_time,
    )

    summary = {
        "setup": {
            "ny": args.ny,
            "nx": args.nx,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "latent": args.latent,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "depth": args.depth,
            "speed_repeats": args.speed_repeats,
            "trained_time_models": args.models,
            "snapshot_png": str(snapshot_path),
            "note": "Steady target is the bundled pretrained PPDONet output, so PPDONet has zero target error by construction. Time target is synthetic/PPDONet-derived, so this is a sanity benchmark, not a real FARGO benchmark.",
        },
        "steady_task": steady_results,
        "time_dependent_task": time_results,
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

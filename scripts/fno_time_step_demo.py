#!/usr/bin/env python3
"""Small FNO-style time-step sanity check.

This keeps the synthetic trajectory setup from the DeepONet demos, but changes
the backbone to a grid-to-grid propagator:

    (x_n(r, theta), mu, dt) -> x_{n+1}(r, theta).

This is the natural FNO use case. Unlike coordinate DeepONet, the model sees the
whole current field at once and predicts the whole next field at once.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
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
class Metrics:
    train_rmse: float
    heldout_window_rmse: float
    heldout_param_window_rmse: float
    two_step_rollout_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes-r", type=int, default=8)
    parser.add_argument("--modes-theta", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "fno_time_step_demo",
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


def load_steady_log_sigma(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    steady = job.s_pred_fn(job.model.params, job.state, {"u_net": u, "y_net": coords})
    return np.asarray(steady, dtype=np.float32).reshape((-1, ny, nx)), normalize_params(parameters, job_args), np.asarray(coords, dtype=np.float32)


def build_global_trajectory(
    steady: np.ndarray,
    mu: np.ndarray,
    coords: np.ndarray,
    ny: int,
    nx: int,
    global_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = coords[:, 0].reshape(ny, nx)
    theta = coords[:, 1].reshape(ny, nx)
    x0 = (-0.5 * np.log10(r)).astype(np.float32)

    traj = []
    denom = 1.0 - np.exp(-4.0)
    for t in global_times:
        progress = (1.0 - np.exp(-4.0 * t)) / denom
        decay = np.exp(-1.8 * t)
        rows = []
        for i in range(steady.shape[0]):
            phase = 2.5 * np.pi * t + 0.4 * mu[i, 1]
            spiral = np.exp(-((r - 1.0) ** 2) / 0.18) * np.cos(theta - phase)
            breathing = 0.02 * np.sin(2.0 * np.pi * t) * np.cos(2.0 * theta)
            amp = 0.08 * (1.0 + 0.15 * mu[i, 2])
            rows.append(x0 + progress * (steady[i] - x0) + decay * amp * spiral + breathing)
        traj.append(np.stack(rows, axis=0))
    return np.stack(traj, axis=1).astype(np.float32), x0


def make_step_dataset(
    trajectory: np.ndarray,
    mu: np.ndarray,
    param_ids: np.ndarray,
    start_ids: np.ndarray,
    dt: float,
) -> dict[str, jnp.ndarray]:
    xs, ys, mus, dts = [], [], [], []
    for p in param_ids:
        for sid in start_ids:
            xs.append(trajectory[p, sid, :, :, None])
            ys.append(trajectory[p, sid + 1, :, :, None])
            mus.append(mu[p])
            dts.append([dt])
    return {
        "x": jnp.asarray(np.stack(xs, axis=0)),
        "y": jnp.asarray(np.stack(ys, axis=0)),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(dts, dtype=np.float32)),
    }


class SpectralConv2D(hk.Module):
    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        mr = min(self.modes_r, ny)
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real = hk.get_parameter(
            "weight_real",
            shape=(mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        imag = hk.get_parameter(
            "weight_imag",
            shape=(mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        weight = real + 1j * imag
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low = jnp.einsum("bhwi,hwio->bhwo", x_ft[:, :mr, :mt, :], weight)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low)
        return jnp.fft.irfft2(out_ft, s=(ny, nx), axes=(1, 2))


def make_fno(width: int, modes_r: int, modes_theta: int, depth: int):
    def forward(x, mu, dt):
        batch, ny, nx, _ = x.shape
        r_grid = jnp.linspace(-1.0, 1.0, ny)[None, :, None, None]
        theta = jnp.linspace(-jnp.pi, jnp.pi, nx, endpoint=False)
        sin_theta = jnp.sin(theta)[None, None, :, None]
        cos_theta = jnp.cos(theta)[None, None, :, None]
        r_grid = jnp.broadcast_to(r_grid, (batch, ny, nx, 1))
        sin_theta = jnp.broadcast_to(sin_theta, (batch, ny, nx, 1))
        cos_theta = jnp.broadcast_to(cos_theta, (batch, ny, nx, 1))
        cond = jnp.concatenate([mu, dt], axis=-1)
        cond = jnp.broadcast_to(cond[:, None, None, :], (batch, ny, nx, cond.shape[-1]))
        h = jnp.concatenate([x, r_grid, sin_theta, cos_theta, cond], axis=-1)
        h = hk.Linear(width)(h)
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        residual = hk.nets.MLP([width, 1], activation=jax.nn.gelu)(h)
        return x + residual

    return hk.without_apply_rng(hk.transform(forward))


def make_pointwise_baseline(width: int, depth: int):
    def forward(x, mu, dt):
        batch, ny, nx, _ = x.shape
        r_grid = jnp.linspace(-1.0, 1.0, ny)[None, :, None, None]
        theta = jnp.linspace(-jnp.pi, jnp.pi, nx, endpoint=False)
        sin_theta = jnp.sin(theta)[None, None, :, None]
        cos_theta = jnp.cos(theta)[None, None, :, None]
        r_grid = jnp.broadcast_to(r_grid, (batch, ny, nx, 1))
        sin_theta = jnp.broadcast_to(sin_theta, (batch, ny, nx, 1))
        cos_theta = jnp.broadcast_to(cos_theta, (batch, ny, nx, 1))
        cond = jnp.concatenate([mu, dt], axis=-1)
        cond = jnp.broadcast_to(cond[:, None, None, :], (batch, ny, nx, cond.shape[-1]))
        h = jnp.concatenate([x, r_grid, sin_theta, cos_theta, cond], axis=-1)
        h = hk.nets.MLP([width] * depth + [1], activation=jax.nn.gelu)(h)
        return x + h

    return hk.without_apply_rng(hk.transform(forward))


def train_model(
    model,
    train: dict[str, jnp.ndarray],
    eval_sets: dict[str, dict[str, jnp.ndarray]],
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[Metrics, list[dict[str, float]], hk.Params]:
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
    history = []
    for step in range(1, args.steps + 1):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        params, opt_state, loss = train_step(params, opt_state, batch)
        if step == 1 or step % 300 == 0 or step == args.steps:
            history.append({"step": step, "batch_rmse": float(jnp.sqrt(loss))})

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

    return (
        Metrics(
            train_rmse=rmse(train),
            heldout_window_rmse=rmse(eval_sets["heldout_window"]),
            heldout_param_window_rmse=rmse(eval_sets["heldout_param_window"]),
            two_step_rollout_rmse=0.0,
        ),
        history,
        params,
    )


def rollout_two_steps(model, params, trajectory, mu, start_id, param_ids, dt):
    x = jnp.asarray(trajectory[param_ids, start_id, :, :, None])
    y = jnp.asarray(trajectory[param_ids, start_id + 2, :, :, None])
    mu_batch = jnp.asarray(mu[param_ids])
    dt_batch = jnp.asarray(np.full((len(param_ids), 1), dt, dtype=np.float32))
    x1 = model.apply(params, x, mu_batch, dt_batch)
    x2 = model.apply(params, x1, mu_batch, dt_batch)
    return float(jnp.sqrt(jnp.mean((x2 - y) ** 2)))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steady, mu, coords = load_steady_log_sigma(args.ny, args.nx)
    global_times = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    dt = float(global_times[1] - global_times[0])
    trajectory, _ = build_global_trajectory(steady, mu, coords, args.ny, args.nx, global_times)

    train_params = np.arange(0, 8)
    test_params = np.arange(8, 10)
    train_starts = np.asarray([0, 1, 2, 3, 4, 5])
    heldout_starts = np.asarray([6, 7])
    train = make_step_dataset(trajectory, mu, train_params, train_starts, dt)
    eval_sets = {
        "heldout_window": make_step_dataset(trajectory, mu, train_params, heldout_starts, dt),
        "heldout_param_window": make_step_dataset(trajectory, mu, test_params, heldout_starts, dt),
    }

    fno = make_fno(args.width, args.modes_r, args.modes_theta, args.depth)
    pointwise = make_pointwise_baseline(args.width, args.depth)
    fno_metrics, fno_history, fno_params = train_model(fno, train, eval_sets, args, seed_offset=0)
    point_metrics, point_history, point_params = train_model(pointwise, train, eval_sets, args, seed_offset=100)

    fno_metrics.two_step_rollout_rmse = rollout_two_steps(
        fno, fno_params, trajectory, mu, start_id=6, param_ids=test_params, dt=dt
    )
    point_metrics.two_step_rollout_rmse = rollout_two_steps(
        pointwise, point_params, trajectory, mu, start_id=6, param_ids=test_params, dt=dt
    )

    summary = {
        "setup": {
            "ny": args.ny,
            "nx": args.nx,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "depth": args.depth,
            "dt": dt,
            "target": "synthetic grid-to-grid one-step transient based on bundled PPDONet steady log_sigma",
        },
        "fno": asdict(fno_metrics),
        "pointwise_baseline": asdict(point_metrics),
        "improvement_ratio_pointwise_over_fno": {
            key: getattr(point_metrics, key) / max(getattr(fno_metrics, key), 1e-12)
            for key in asdict(fno_metrics)
        },
        "history": {"fno": fno_history, "pointwise_baseline": point_history},
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

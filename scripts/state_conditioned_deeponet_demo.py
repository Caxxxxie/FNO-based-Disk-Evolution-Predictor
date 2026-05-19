#!/usr/bin/env python3
"""Sanity check for a state-conditioned time-window DeepONet.

This script is deliberately synthetic. It asks a focused question:

    Does a local operator need the current state x_n, or is time input enough?

We create many local windows from a transient trajectory. Every window uses the
same local tau grid in [0, 1], but each window starts from a different state.
The state-conditioned model sees sensor values from x_n. The no-state baseline
sees only (mu, r, theta, tau). If the task is really about evolution, the
state-conditioned model should win.
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
    local_tau0_rmse: float
    local_tau1_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--state-sensors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "state_conditioned_deeponet_demo",
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
    inputs = {"u_net": u, "y_net": coords}
    steady = job.s_pred_fn(job.model.params, job.state, inputs)
    return np.asarray(steady, dtype=np.float32), normalize_params(parameters, job_args), np.asarray(coords, dtype=np.float32)


def build_global_trajectory(
    steady: np.ndarray,
    mu: np.ndarray,
    coords: np.ndarray,
    global_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = coords[:, 0]
    theta = coords[:, 1]
    r_norm = 2.0 * (r - r.min()) / (r.max() - r.min()) - 1.0
    coord_features = np.stack([r_norm, np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)
    x0 = (-0.5 * np.log10(r)).astype(np.float32)

    # Smoothly approach the bundled steady field, with a decaying moving spiral.
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
            transient = decay * amp * spiral + breathing
            rows.append(x0 + progress * (steady[i] - x0) + transient)
        traj.append(np.stack(rows, axis=0))
    return np.stack(traj, axis=1).astype(np.float32), x0[:, None], coord_features


def make_windows(
    trajectory: np.ndarray,
    global_times: np.ndarray,
    window_start_ids: np.ndarray,
    local_tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return targets[p, w, tau, coord] and start_state[p, w, coord]."""
    window_targets = []
    start_states = []
    for start_id in window_start_ids:
        start_t = global_times[start_id]
        end_t = global_times[start_id + 2]
        # local tau points align with global grid because starts are two-index windows.
        wanted_times = start_t + local_tau * (end_t - start_t)
        idx = [int(np.argmin(np.abs(global_times - t))) for t in wanted_times]
        window_targets.append(trajectory[:, idx, :])
        start_states.append(trajectory[:, start_id, :])
    return np.stack(window_targets, axis=1), np.stack(start_states, axis=1)


def sensor_matrix(n_coord: int, n_sensors: int) -> np.ndarray:
    if n_sensors > n_coord:
        raise ValueError("n_sensors cannot exceed coordinate count")
    return np.linspace(0, n_coord - 1, n_sensors, dtype=np.int32)


def flatten_dataset(
    mu: np.ndarray,
    coord_features: np.ndarray,
    window_targets: np.ndarray,
    start_states: np.ndarray,
    param_ids: np.ndarray,
    window_ids: np.ndarray,
    tau_ids: np.ndarray,
    local_tau: np.ndarray,
    sensor_ids: np.ndarray,
) -> dict[str, jnp.ndarray]:
    parts = {"mu": [], "state": [], "coord": [], "tau": [], "x_start": [], "y": []}
    n_coord = coord_features.shape[0]
    for p in param_ids:
        for w in window_ids:
            state = start_states[p, w, sensor_ids]
            state = (state - state.mean()) / (state.std() + 1e-6)
            state_rep = np.repeat(state[None, :], n_coord, axis=0).astype(np.float32)
            x_start = start_states[p, w, :, None]
            for tid in tau_ids:
                parts["mu"].append(np.repeat(mu[p : p + 1], n_coord, axis=0))
                parts["state"].append(state_rep)
                parts["coord"].append(coord_features)
                parts["tau"].append(np.full((n_coord, 1), local_tau[tid], dtype=np.float32))
                parts["x_start"].append(x_start)
                parts["y"].append(window_targets[p, w, tid, :, None])
    return {key: jnp.asarray(np.concatenate(value, axis=0)) for key, value in parts.items()}


def make_model(use_state: bool, latent: int, width: int):
    def forward(mu, state, coord, tau, x_start):
        if use_state:
            branch_input = jnp.concatenate([mu, state], axis=-1)
        else:
            branch_input = mu
        branch = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(branch_input)
        trunk_input = jnp.concatenate([coord, tau], axis=-1)
        trunk = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(trunk_input)
        raw = jnp.sum(branch * trunk, axis=-1, keepdims=True) / math.sqrt(latent)
        return x_start + tau * raw

    return hk.without_apply_rng(hk.transform(forward))


def train_model(
    train: dict[str, jnp.ndarray],
    eval_sets: dict[str, dict[str, jnp.ndarray]],
    use_state: bool,
    args: argparse.Namespace,
) -> tuple[Metrics, list[dict[str, float]]]:
    model = make_model(use_state=use_state, latent=args.latent, width=args.width)
    key = jax.random.PRNGKey(args.seed + int(use_state))
    params = model.init(
        key,
        train["mu"][:1],
        train["state"][:1],
        train["coord"][:1],
        train["tau"][:1],
        train["x_start"][:1],
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(
            params,
            batch["mu"],
            batch["state"],
            batch["coord"],
            batch["tau"],
            batch["x_start"],
        )
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = train["y"].shape[0]
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
            local_tau0_rmse=rmse(eval_sets["local_tau0"]),
            local_tau1_rmse=rmse(eval_sets["local_tau1"]),
        ),
        history,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steady, mu, coords = load_steady_log_sigma(args.ny, args.nx)

    global_times = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    local_tau = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    trajectory, _, coord_features = build_global_trajectory(steady, mu, coords, global_times)
    window_start_ids = np.asarray([0, 2, 4, 6])
    window_targets, start_states = make_windows(
        trajectory,
        global_times,
        window_start_ids,
        local_tau,
    )

    sensor_ids = sensor_matrix(coord_features.shape[0], args.state_sensors)
    train_params = np.arange(0, 8)
    test_params = np.arange(8, 10)
    train_windows = np.asarray([0, 1, 2])
    heldout_windows = np.asarray([3])
    all_tau = np.asarray([0, 1, 2])

    train = flatten_dataset(
        mu,
        coord_features,
        window_targets,
        start_states,
        train_params,
        train_windows,
        all_tau,
        local_tau,
        sensor_ids,
    )
    eval_sets = {
        "heldout_window": flatten_dataset(
            mu, coord_features, window_targets, start_states, train_params, heldout_windows, all_tau, local_tau, sensor_ids
        ),
        "heldout_param_window": flatten_dataset(
            mu, coord_features, window_targets, start_states, test_params, heldout_windows, all_tau, local_tau, sensor_ids
        ),
        "local_tau0": flatten_dataset(
            mu, coord_features, window_targets, start_states, test_params, heldout_windows, np.asarray([0]), local_tau, sensor_ids
        ),
        "local_tau1": flatten_dataset(
            mu, coord_features, window_targets, start_states, test_params, heldout_windows, np.asarray([2]), local_tau, sensor_ids
        ),
    }

    with_state, with_history = train_model(train, eval_sets, use_state=True, args=args)
    no_state, no_history = train_model(train, eval_sets, use_state=False, args=args)

    summary = {
        "setup": {
            "ny": args.ny,
            "nx": args.nx,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "state_sensors": args.state_sensors,
            "train_windows_global_start": global_times[window_start_ids[train_windows]].tolist(),
            "heldout_windows_global_start": global_times[window_start_ids[heldout_windows]].tolist(),
            "local_tau": local_tau.tolist(),
            "target": "synthetic multi-window transient ending near bundled PPDONet steady log_sigma",
        },
        "with_state": asdict(with_state),
        "without_state": asdict(no_state),
        "improvement_ratio_without_over_with": {
            key: getattr(no_state, key) / max(getattr(with_state, key), 1e-12)
            for key in asdict(with_state)
        },
        "history": {"with_state": with_history, "without_state": no_history},
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

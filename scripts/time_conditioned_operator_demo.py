#!/usr/bin/env python3
"""Mini demo: add time to the operator input and test if it helps.

This is not a real time-dependent FARGO reproduction yet. It is a controlled
smoke test:

1. load the bundled steady PPDONet log_sigma model,
2. build a synthetic transient that starts from an analytic initial disk and
   ends at the steady PPDONet prediction,
3. train two tiny operator networks:
   - one sees local time tau in the coordinate branch,
   - one does not, except for the hard IC factor tau,
4. compare both on held-out parameters and held-out times.

If the time-conditioned model wins, the architecture is at least doing the
right basic thing before we spend effort on real FARGO time snapshots.
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
    heldout_time_rmse: float
    heldout_param_time_rmse: float
    endpoint_t0_rmse: float
    endpoint_t1_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "time_conditioned_operator_demo",
    )
    return parser.parse_args()


def normalize_params(parameters: pd.DataFrame, job_args: dict) -> np.ndarray:
    """Use the same log/linear normalization range as the bundled PPDONet."""
    cols = sorted(job_args["parameter"])
    values = parameters[cols].to_numpy(dtype=np.float32)
    transforms = job_args["u_transform"]
    for i, transform in enumerate(transforms):
        if transform == "log10":
            values[:, i] = np.log10(values[:, i])
    u_min = np.asarray(job_args["u_min"], dtype=np.float32)
    u_max = np.asarray(job_args["u_max"], dtype=np.float32)
    return 2.0 * (values - (u_min + u_max) / 2.0) / (u_max - u_min)


def load_steady_log_sigma(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    mu = normalize_params(parameters, job_args)
    return (
        np.asarray(steady, dtype=np.float32),
        mu.astype(np.float32),
        np.asarray(coords, dtype=np.float32),
        parameters.to_numpy(dtype=np.float32),
    )


def make_synthetic_transient(
    steady: np.ndarray,
    mu: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small transient with exact initial and final states."""
    r = coords[:, 0]
    theta = coords[:, 1]
    r_norm = 2.0 * (r - r.min()) / (r.max() - r.min()) - 1.0
    coord_features = np.stack([r_norm, np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)

    # The log of the original power-law density initial condition.
    x0 = (-0.5 * np.log10(r)).astype(np.float32)

    targets = []
    denom = 1.0 - np.exp(-4.0)
    for tau in times:
        progress = (1.0 - np.exp(-4.0 * tau)) / denom
        window = np.sin(np.pi * tau)
        rows = []
        for i in range(steady.shape[0]):
            h_feature = mu[i, 1]
            q_feature = mu[i, 2]
            spiral = np.exp(-((r - 1.0) ** 2) / 0.25) * np.cos(
                theta - 2.0 * np.pi * tau + 0.7 * h_feature
            )
            amp = 0.04 * (1.0 + 0.2 * q_feature)
            transient = amp * window * spiral
            rows.append(x0 + progress * (steady[i] - x0) + transient)
        targets.append(np.stack(rows, axis=0))
    return np.stack(targets, axis=1).astype(np.float32), x0[:, None], coord_features


def flatten_dataset(
    mu: np.ndarray,
    coord_features: np.ndarray,
    x0: np.ndarray,
    targets: np.ndarray,
    param_ids: np.ndarray,
    time_ids: np.ndarray,
    times: np.ndarray,
) -> dict[str, jnp.ndarray]:
    parts = {"mu": [], "coord": [], "tau": [], "x0": [], "y": []}
    n_coord = coord_features.shape[0]
    for p in param_ids:
        for tid in time_ids:
            parts["mu"].append(np.repeat(mu[p : p + 1], n_coord, axis=0))
            parts["coord"].append(coord_features)
            parts["tau"].append(np.full((n_coord, 1), times[tid], dtype=np.float32))
            parts["x0"].append(x0)
            parts["y"].append(targets[p, tid, :, None])
    return {key: jnp.asarray(np.concatenate(value, axis=0)) for key, value in parts.items()}


def make_model(include_time: bool, latent: int, width: int):
    def forward(mu, coord, tau, x0):
        branch = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(mu)
        if include_time:
            trunk_input = jnp.concatenate([coord, tau], axis=-1)
        else:
            trunk_input = coord
        trunk = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(trunk_input)
        raw = jnp.sum(branch * trunk, axis=-1, keepdims=True) / math.sqrt(latent)
        return x0 + tau * raw

    return hk.without_apply_rng(hk.transform(forward))


def train_model(
    train: dict[str, jnp.ndarray],
    eval_sets: dict[str, dict[str, jnp.ndarray]],
    include_time: bool,
    args: argparse.Namespace,
) -> tuple[Metrics, list[dict[str, float]]]:
    model = make_model(include_time=include_time, latent=args.latent, width=args.width)
    key = jax.random.PRNGKey(args.seed + int(include_time))
    params = model.init(
        key,
        train["mu"][:1],
        train["coord"][:1],
        train["tau"][:1],
        train["x0"][:1],
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch["mu"], batch["coord"], batch["tau"], batch["x0"])
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n = train["y"].shape[0]
    history = []
    for step in range(1, args.steps + 1):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        params, opt_state, loss = train_step(params, opt_state, batch)
        if step == 1 or step % 200 == 0 or step == args.steps:
            history.append({"step": step, "batch_rmse": float(jnp.sqrt(loss))})

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

    metrics = Metrics(
        train_rmse=rmse(train),
        heldout_time_rmse=rmse(eval_sets["heldout_time"]),
        heldout_param_time_rmse=rmse(eval_sets["heldout_param_time"]),
        endpoint_t0_rmse=rmse(eval_sets["endpoint_t0"]),
        endpoint_t1_rmse=rmse(eval_sets["endpoint_t1"]),
    )
    return metrics, history


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steady, mu, coords, _ = load_steady_log_sigma(args.ny, args.nx)
    times = np.asarray([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float32)
    targets, x0, coord_features = make_synthetic_transient(steady, mu, coords, times)

    train_params = np.arange(0, 8)
    test_params = np.arange(8, 10)
    train_time_ids = np.asarray([0, 2, 4, 6, 8])
    heldout_time_ids = np.asarray([1, 3, 5, 7])
    endpoint_t0_id = np.asarray([0])
    endpoint_t1_id = np.asarray([8])

    train = flatten_dataset(mu, coord_features, x0, targets, train_params, train_time_ids, times)
    eval_sets = {
        "heldout_time": flatten_dataset(mu, coord_features, x0, targets, train_params, heldout_time_ids, times),
        "heldout_param_time": flatten_dataset(mu, coord_features, x0, targets, test_params, heldout_time_ids, times),
        "endpoint_t0": flatten_dataset(mu, coord_features, x0, targets, test_params, endpoint_t0_id, times),
        "endpoint_t1": flatten_dataset(mu, coord_features, x0, targets, test_params, endpoint_t1_id, times),
    }

    with_time, with_history = train_model(train, eval_sets, include_time=True, args=args)
    no_time, no_history = train_model(train, eval_sets, include_time=False, args=args)

    summary = {
        "setup": {
            "ny": args.ny,
            "nx": args.nx,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "train_parameters": train_params.tolist(),
            "heldout_parameters": test_params.tolist(),
            "train_times": times[train_time_ids].tolist(),
            "heldout_times": times[heldout_time_ids].tolist(),
            "target": "synthetic transient from analytic initial profile to bundled PPDONet steady log_sigma",
        },
        "with_time": asdict(with_time),
        "without_time": asdict(no_time),
        "improvement_ratio_without_over_with": {
            key: getattr(no_time, key) / max(getattr(with_time, key), 1e-12)
            for key in asdict(with_time)
        },
        "history": {
            "with_time": with_history,
            "without_time": no_history,
        },
    }

    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

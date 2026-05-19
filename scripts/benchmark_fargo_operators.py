#!/usr/bin/env python3
"""Compare small operator variants on a real tiny FARGO3D time series.

This is intentionally a sanity benchmark. The bundled dataset is tiny, coarse,
and short; it is useful for checking whether an implementation can learn a
real solver-produced transient signal before we spend time on larger sweeps.
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
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ppdonet_style", "time_deeponet", "state_deeponet", "fno"],
        choices=["ppdonet_style", "time_deeponet", "state_deeponet", "pointwise", "fno"],
        help="Model subset to train. Persistence is always reported.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "fargo_operator_benchmark")
    return parser.parse_args()


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
    x0 = x[..., 0]
    pooled = x0.reshape((x0.shape[0], 4, x0.shape[1] // 4, 8, x0.shape[2] // 8)).mean(axis=(2, 4))
    stats = np.stack([x0.mean(axis=(1, 2)), x0.std(axis=(1, 2))], axis=-1)
    return np.concatenate([stats, pooled.reshape((x0.shape[0], -1))], axis=-1).astype(np.float32)


def state_features_jax(x: jnp.ndarray) -> jnp.ndarray:
    x0 = x[..., 0]
    pooled = x0.reshape((x0.shape[0], 4, x0.shape[1] // 4, 8, x0.shape[2] // 8)).mean(axis=(2, 4))
    stats = jnp.stack([x0.mean(axis=(1, 2)), x0.std(axis=(1, 2))], axis=-1)
    return jnp.concatenate([stats, pooled.reshape((x0.shape[0], -1))], axis=-1)


def load_dataset(path: Path):
    data = np.load(path)
    x = data["log_sigma"].astype(np.float32)
    raw_params = data["params"].astype(np.float32)
    params = normalize_params(raw_params)
    r = data["r"].astype(np.float32)
    theta = data["theta"].astype(np.float32)
    times = data["times"].astype(np.float32)
    meta = json.loads(data["meta"].item())
    return x, params, raw_params, r, theta, times, meta


def make_step_data(x, params, times, param_ids, start_ids, mean, std):
    xs, ys, mus, ts, dts = [], [], [], [], []
    t_scale = float(times[-1] - times[0])
    for p in param_ids:
        for sid in start_ids:
            xs.append(((x[p, sid] - mean) / std)[..., None])
            ys.append(((x[p, sid + 1] - mean) / std)[..., None])
            mus.append(params[p])
            ts.append([(times[sid + 1] - times[0]) / t_scale])
            dts.append([(times[sid + 1] - times[sid]) / t_scale])
    xs = np.stack(xs, axis=0).astype(np.float32)
    return {
        "x": jnp.asarray(xs),
        "y": jnp.asarray(np.stack(ys, axis=0).astype(np.float32)),
        "mu": jnp.asarray(np.asarray(mus, dtype=np.float32)),
        "t": jnp.asarray(np.asarray(ts, dtype=np.float32)),
        "dt": jnp.asarray(np.asarray(dts, dtype=np.float32)),
        "state": jnp.asarray(state_features(xs)),
    }


def make_final_data(x, params, times, param_ids, mean, std):
    y = ((x[param_ids, -1] - mean) / std)[..., None]
    t_final = np.full((len(param_ids), 1), 1.0, dtype=np.float32)
    zeros = np.zeros_like(y, dtype=np.float32)
    return {
        "x": jnp.asarray(zeros),
        "y": jnp.asarray(y.astype(np.float32)),
        "mu": jnp.asarray(params[param_ids].astype(np.float32)),
        "t": jnp.asarray(t_final),
        "dt": jnp.asarray(t_final),
        "state": jnp.asarray(state_features(zeros)),
    }


def add_two_step_target(data, fields, param_ids, start_ids, mean, std):
    targets = []
    for p in param_ids:
        for sid in start_ids:
            targets.append(((fields[p, sid + 2] - mean) / std)[..., None])
    data = dict(data)
    data["two_step_target"] = jnp.asarray(np.stack(targets, axis=0).astype(np.float32))
    return data


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


def make_param_deeponet(coords, latent, width, use_time: bool, use_state: bool, residual: bool):
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
        trunk = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(coords_flat)
        branch = hk.nets.MLP([width, width, latent], activation=jax.nn.tanh)(branch_input)
        y = jnp.einsum("bl,nl->bn", branch, trunk) / math.sqrt(latent)
        y = y.reshape((b, ny, nx, 1))
        return x + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_fno(coords, width, modes_r, modes_theta, depth):
    def forward(batch):
        h = hk.Linear(width)(grid_inputs(batch, coords))
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        residual = hk.nets.MLP([width, 1], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * residual

    return hk.without_apply_rng(hk.transform(forward))


def make_pointwise(coords, width, depth):
    def forward(batch):
        residual = hk.nets.MLP([width] * depth + [1], activation=jax.nn.gelu)(grid_inputs(batch, coords))
        return batch["x"] + batch["dt"][:, None, None, :] * residual

    return hk.without_apply_rng(hk.transform(forward))


def train_and_eval(model, train, evals, args, seed_offset=0, rollout_train=None):
    key = jax.random.PRNGKey(args.seed + seed_offset)
    params = model.init(key, {k: v[:1] for k, v in train.items()})
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.lr))
    opt_state = opt.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch)
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def rollout_loss_fn(params, batch, rollout_batch):
        pred = model.apply(params, batch)
        one_step = jnp.mean((pred - batch["y"]) ** 2)
        x1 = model.apply(params, rollout_batch)
        batch2 = dict(rollout_batch)
        batch2["x"] = x1
        batch2["state"] = state_features_jax(x1)
        x2 = model.apply(params, batch2)
        rollout = jnp.mean((x2 - rollout_batch["two_step_target"]) ** 2)
        return one_step + args.rollout_weight * rollout

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def rollout_train_step(params, opt_state, batch, rollout_batch):
        loss, grads = jax.value_and_grad(rollout_loss_fn)(params, batch, rollout_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = train["x"].shape[0]
    n_rollout = 0 if rollout_train is None else rollout_train["x"].shape[0]
    for _ in range(args.steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (args.batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        if rollout_train is None:
            params, opt_state, _ = train_step(params, opt_state, batch)
        else:
            key, rollout_key = jax.random.split(key)
            ridx = jax.random.randint(rollout_key, (args.batch_size,), 0, n_rollout)
            rollout_batch = {k: v[ridx] for k, v in rollout_train.items()}
            params, opt_state, _ = rollout_train_step(params, opt_state, batch, rollout_batch)

    def rmse(data):
        return float(jnp.sqrt(loss_fn(params, data)))

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

    return params, rmse, speed


def rollout_two_steps(model, params, data, test_case_index=3):
    y2 = data["two_step_target"]
    batch = {k: v for k, v in data.items() if k != "two_step_target"}
    x1 = model.apply(params, batch)
    batch2 = dict(batch)
    batch2["x"] = x1
    batch2["state"] = state_features_jax(x1)
    x2 = model.apply(params, batch2)
    return float(jnp.sqrt(jnp.mean((x2 - y2) ** 2)))


def persistence_metrics(train, param_eval, time_eval, rollout_data, args):
    def rmse(data):
        return float(jnp.sqrt(jnp.mean((data["x"] - data["y"]) ** 2)))

    start = time.perf_counter()
    for _ in range(args.speed_repeats):
        y = param_eval["x"]
    _ = np.asarray(y).shape
    speed = (time.perf_counter() - start) * 1000.0 / args.speed_repeats
    two = float(jnp.sqrt(jnp.mean((rollout_data["x"] - rollout_data["two_step_target"]) ** 2)))
    return MetricRow(rmse(train), rmse(param_eval), rmse(time_eval), two, speed)


def load_pretrained_ppdonet_predictions(raw_params, r, theta):
    sys.path.insert(0, PPDONET_ROOT.as_posix())
    import onet_disk2D.run  # noqa: PLC0415

    run_dir = PPDONET_ROOT / "trained_network" / "single_log_sigma"
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        args_file="args.yml",
        arg_groups_file="arg_groups.yml",
        fargo_setup_file="fargo_setups.yml",
    )
    job = onet_disk2D.run.JOB(job_args)
    job.load_model(run_dir)

    name_to_col = {
        "alpha": 0,
        "aspectratio": 1,
        "aspect_ratio": 1,
        "planetmass": 2,
        "planet_mass": 2,
    }
    cols = [name_to_col[p.lower()] for p in sorted(job_args["parameter"])]
    u = jnp.asarray(raw_params[:, cols], dtype=jnp.float32)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    coords = jnp.asarray(np.stack([rr, tt], axis=-1).reshape((-1, 2)), dtype=jnp.float32)
    pred_fn = jax.jit(lambda u_batch: job.s_pred_fn(job.model.params, job.state, {"u_net": u_batch, "y_net": coords}))
    pred = pred_fn(u).block_until_ready()
    pred = np.asarray(pred, dtype=np.float32).reshape((raw_params.shape[0], len(r), len(theta)))

    def speed(param_ids, repeats):
        u_batch = jnp.asarray(raw_params[param_ids][:, cols], dtype=jnp.float32)
        pred_fn(u_batch).block_until_ready()
        start = time.perf_counter()
        for _ in range(repeats):
            y = pred_fn(u_batch)
        y.block_until_ready()
        return (time.perf_counter() - start) * 1000.0 / repeats

    return pred, speed


def pretrained_ppdonet_metrics(pred, speed_fn, fields, train_case_ids, heldout_case_ids, train_starts, heldout_time_starts, mean, std, args):
    pred_norm = (pred - mean) / std

    def rmse(param_ids, start_ids):
        ys = []
        ps = []
        for p in param_ids:
            for sid in start_ids:
                ys.append((fields[p, sid + 1] - mean) / std)
                ps.append(pred_norm[p])
        return float(np.sqrt(np.mean((np.stack(ps) - np.stack(ys)) ** 2)))

    two_target = (fields[heldout_case_ids, -1] - mean) / std
    two_pred = pred_norm[heldout_case_ids]
    param_eval_ids = np.repeat(heldout_case_ids, len(train_starts))
    return MetricRow(
        train_rmse=rmse(train_case_ids, train_starts),
        heldout_param_rmse=rmse(heldout_case_ids, train_starts),
        heldout_time_rmse=rmse(train_case_ids, heldout_time_starts),
        two_step_rollout_rmse=float(np.sqrt(np.mean((two_pred - two_target) ** 2))),
        speed_ms_per_batch=speed_fn(param_eval_ids, args.speed_repeats),
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields, params, raw_params, r, theta, times, meta = load_dataset(args.dataset)
    coords = coordinate_grid(r, theta)

    n_cases = fields.shape[0]
    raw_heldout_cases = args.heldout_cases if args.heldout_cases is not None else [args.heldout_case]
    heldout_cases = [case if case >= 0 else n_cases + case for case in raw_heldout_cases]
    if any(case < 0 or case >= n_cases for case in heldout_cases):
        raise ValueError(f"heldout cases {raw_heldout_cases} are outside 0..{n_cases - 1}")
    heldout_case_ids = np.asarray(sorted(set(heldout_cases)))
    train_case_ids = np.asarray([i for i in range(n_cases) if i not in set(heldout_case_ids.tolist())])
    if train_case_ids.size == 0:
        raise ValueError("At least one training case is required")
    if fields.shape[1] < 4:
        raise ValueError("At least four frames are required for one-step and two-step tests")
    train_starts = np.arange(0, fields.shape[1] - 2)
    heldout_time_starts = np.asarray([fields.shape[1] - 2])

    mean = float(fields[train_case_ids][:, train_starts].mean())
    std = float(fields[train_case_ids][:, train_starts].std() + 1.0e-6)

    step_train = make_step_data(fields, params, times, train_case_ids, train_starts, mean, std)
    rollout_train_starts = train_starts[train_starts + 2 < fields.shape[1]]
    rollout_train = add_two_step_target(
        make_step_data(fields, params, times, train_case_ids, rollout_train_starts, mean, std),
        fields,
        train_case_ids,
        rollout_train_starts,
        mean,
        std,
    )
    step_param = make_step_data(fields, params, times, heldout_case_ids, train_starts, mean, std)
    step_time = make_step_data(fields, params, times, train_case_ids, heldout_time_starts, mean, std)
    final_train = make_final_data(fields, params, times, train_case_ids, mean, std)
    final_param = make_final_data(fields, params, times, heldout_case_ids, mean, std)

    rollout_start = fields.shape[1] - 3
    rollout_data = make_step_data(fields, params, times, heldout_case_ids, np.asarray([rollout_start]), mean, std)
    two_target = ((fields[heldout_case_ids, -1] - mean) / std)[..., None].astype(np.float32)
    rollout_data["two_step_target"] = jnp.asarray(two_target)

    model_specs = {
        "ppdonet_style": (
            "ppdonet_style_final_param_to_last_frame",
            make_param_deeponet(coords, args.latent, args.width, use_time=False, use_state=False, residual=False),
            final_train,
            final_param,
            final_param,
            None,
        ),
        "time_deeponet": (
            "time_deeponet_param_t_to_next_frame",
            make_param_deeponet(coords, args.latent, args.width, use_time=True, use_state=False, residual=False),
            step_train,
            step_param,
            step_time,
            None,
        ),
        "state_deeponet": (
            "state_conditioned_deeponet_step",
            make_param_deeponet(coords, args.latent, args.width, use_time=True, use_state=True, residual=True),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
        "pointwise": (
            "pointwise_residual_step",
            make_pointwise(coords, args.width, args.depth),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
        "fno": (
            "fno_residual_step",
            make_fno(coords, args.width, args.modes_r, args.modes_theta, args.depth),
            step_train,
            step_param,
            step_time,
            rollout_data,
        ),
    }

    results = {
        "persistence_xn_as_xnp1": asdict(
            persistence_metrics(step_train, step_param, step_time, rollout_data, args)
        )
    }
    if not args.no_pretrained_ppdonet:
        ppdonet_pred, ppdonet_speed = load_pretrained_ppdonet_predictions(raw_params, r, theta)
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
            )
        )
    for i, key in enumerate(args.models):
        name, model, train, param_eval, time_eval, rollout_eval = model_specs[key]
        print(f"Training {name}...")
        use_rollout_train = rollout_eval is not None and args.rollout_weight > 0.0
        trained_params, rmse, speed = train_and_eval(
            model,
            train,
            {},
            args,
            seed_offset=10 * i,
            rollout_train=rollout_train if use_rollout_train else None,
        )
        two_step = None
        if rollout_eval is not None:
            two_step = rollout_two_steps(model, trained_params, rollout_eval)
        results[name] = asdict(
            MetricRow(
                train_rmse=rmse(train),
                heldout_param_rmse=rmse(param_eval),
                heldout_time_rmse=rmse(time_eval),
                two_step_rollout_rmse=two_step,
                speed_ms_per_batch=speed(param_eval),
            )
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
            "normalization_mean": mean,
            "normalization_std": std,
            "rollout_weight": args.rollout_weight,
            "grad_clip_norm": args.grad_clip_norm,
            "metric_units": "RMSE in normalized log_sigma units",
            "note": (
                f"Tiny FARGO3D sanity benchmark: {fields.shape[0]} cases, "
                f"{fields.shape[1]} frames, {fields.shape[2]}x{fields.shape[3]} grid. "
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

#!/usr/bin/env python3
"""Long-term trajectory benchmark scaffold for transient FARGO operators.

This script follows the recurrent 2D+RNN setup used in Li et al. section 5.3:

    previous H saved states -> next saved state

During training, each batch rolls out several future steps autoregressively:
the prediction from step k is appended to the input window before predicting
step k + 1, and every predicted step is compared with the corresponding ground
truth frame. At evaluation time, the same recurrent update is extended to a
chosen physical horizon.

The FNO-3D branch in Li et al. predicts a whole future trajectory in one pass.
That requires a separate trajectory-output wrapper, so this scaffold focuses on
the recurrent 2D models and leaves the actual model choices blank.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
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
ORBIT_PERIOD = 2.0 * math.pi
MODEL_CHOICES = (
    "geometry_aware_fno",
    "geometry_radial_fno",
    "fno_reflect2d",
    "plain_fno",
    "unet",
    "periodic_unet",
    "convlstm",
)
MODEL_ALIASES = {
    "fno": "geometry_aware_fno",
    "fno_radial": "geometry_radial_fno",
    "radial_fno": "geometry_radial_fno",
    "geometry-radial-fno": "geometry_radial_fno",
    "reflect2d_fno": "fno_reflect2d",
    "reflect2d-fno": "fno_reflect2d",
    "fno-reflect2d": "fno_reflect2d",
    "plain-fno": "plain_fno",
    "geometry-aware-fno": "geometry_aware_fno",
    "geometry-fno": "geometry_aware_fno",
    "geometry_fno": "geometry_aware_fno",
    "periodic-unet": "periodic_unet",
}
MODEL_CLI_CHOICES = tuple(sorted(set(MODEL_CHOICES + tuple(MODEL_ALIASES))))


def load_v3_module():
    candidates = [
        ROOT / "scripts" / "benchmark_fargo_operators_v3.1.py",
        ROOT / "scripts" / "benchmark_fargo_operators_v3.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("benchmark_fargo_operators_v3_current", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load module spec from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.__file_path__ = path
            return module
    raise FileNotFoundError("Could not find benchmark_fargo_operators_v3.1.py or benchmark_fargo_operators_v3.py")


V3 = load_v3_module()


@dataclass
class LongTermModelResult:
    train_rollout_validation: dict
    next_step_validation: dict
    rollout_horizons: dict
    wall_time_sec: dict
    trained_steps: int
    best_step: int
    best_validation_rollout_rmse: float
    checkpoint: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "smoke_10orbits_128f")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "long_term_rollout_v3")
    parser.add_argument("--channels", nargs="+", default=["log_sigma", "delta_v_r", "delta_v_theta"])
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        choices=MODEL_CLI_CHOICES,
        help="Intentionally blank by default. Supports aliases such as plain-fno and geometry-aware-fno.",
    )
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--step-span", type=int, default=1, help="Number of saved frames advanced by one recurrent step.")
    parser.add_argument("--history-stride", type=int, default=None, help="Frame gap inside the history window; defaults to --step-span.")
    parser.add_argument(
        "--train-rollout-steps",
        type=int,
        default=10,
        help="Number of future recurrent steps included in each training-batch autoregressive loss.",
    )
    parser.add_argument(
        "--rollout-gradient",
        choices=["full", "truncated"],
        default="full",
        help="Use truncated to stop gradients through predicted states between rollout-loss steps.",
    )
    parser.add_argument(
        "--remat-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute model activations during backprop to reduce memory use.",
    )
    parser.add_argument(
        "--rollout-end-orbits",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0],
        help="One or more recursive rollout durations in orbits, e.g. --rollout-end-orbits 1.0 2.0 5.0.",
    )
    parser.add_argument(
        "--rollout-start-mode",
        choices=["sampled", "initial"],
        default="sampled",
        help="sampled uses all valid 0-10 orbit windows; initial reproduces rollout from the first history window only.",
    )
    parser.add_argument("--rollout-batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch-style training. If set, overrides --steps using ceil(samples_per_epoch / batch_size) * epochs.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=1000,
        help="Number of randomly sampled training windows that define one epoch when --epochs is used.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes-r", type=int, default=12)
    parser.add_argument("--modes-theta", type=int, default=24)
    parser.add_argument("--radial-kernels", type=int, nargs="+", default=[3, 5, 5])
    parser.add_argument("--radial-dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--radial-padding", choices=["zero", "edge", "reflect"], default="edge")
    parser.add_argument("--unet-levels", type=int, default=3)
    parser.add_argument("--convlstm-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "step"], default="step")
    parser.add_argument("--lr-step-epochs", type=int, default=100)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--normalization-samples", type=int, default=256)
    parser.add_argument("--loss-weighting", choices=["uniform", "area"], default="area")
    parser.add_argument("--rel-l2-floor", type=float, default=1.0e-6)
    parser.add_argument("--physics-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planet-radius", type=float, default=1.0)
    parser.add_argument("--gap-r-min", type=float, default=0.6)
    parser.add_argument("--gap-r-max", type=float, default=1.4)
    parser.add_argument("--ring-r-min", type=float, default=0.6)
    parser.add_argument("--ring-r-max", type=float, default=2.2)
    parser.add_argument("--spiral-r-min", type=float, default=0.5)
    parser.add_argument("--spiral-r-max", type=float, default=2.0)
    parser.add_argument("--sigma-ref", type=float, default=1.0)
    parser.add_argument("--sigma-ref-slope", type=float, default=0.5)
    parser.add_argument("--gap-detection-fraction", type=float, default=0.9)
    parser.add_argument("--ring-detection-factor", type=float, default=1.05)
    parser.add_argument("--diagnostic-smoothing", type=int, default=5)
    parser.add_argument("--spiral-modes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-platform", choices=["default", "cpu", "gpu", "cuda"], default="default")
    args = parser.parse_args()
    if args.history_stride is None:
        args.history_stride = args.step_span
    args.models = canonicalize_models(args.models)
    if args.jax_platform != "default":
        platform = "cuda" if args.jax_platform == "gpu" else args.jax_platform
        jax.config.update("jax_platforms", platform)
        args.jax_platform = platform
    validate_args(args)
    resolve_training_length(args)
    return args


def canonicalize_models(models: list[str]) -> list[str]:
    canonical = []
    seen = set()
    for name in models:
        resolved = MODEL_ALIASES.get(name, name)
        if resolved not in seen:
            canonical.append(resolved)
            seen.add(resolved)
    return canonical


def validate_args(args: argparse.Namespace) -> None:
    if args.history_length < 2:
        raise ValueError("--history-length must be at least 2")
    if args.step_span <= 0 or args.history_stride <= 0:
        raise ValueError("--step-span and --history-stride must be positive")
    if args.history_stride != args.step_span:
        raise ValueError("This recurrent rollout scaffold requires --history-stride == --step-span")
    if args.train_rollout_steps <= 0:
        raise ValueError("--train-rollout-steps must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive when provided")
    if args.samples_per_epoch <= 0:
        raise ValueError("--samples-per-epoch must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.lr_step_epochs <= 0:
        raise ValueError("--lr-step-epochs must be positive")
    if not (0.0 < args.lr_decay_factor <= 1.0):
        raise ValueError("--lr-decay-factor must be in (0, 1]")
    if any(orbit <= 0.0 for orbit in args.rollout_end_orbits):
        raise ValueError("All --rollout-end-orbits values must be positive")
    if len(args.radial_kernels) != len(args.radial_dilations):
        raise ValueError("--radial-kernels and --radial-dilations must have the same length")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in args.radial_kernels):
        raise ValueError("All radial kernels must be positive odd integers")
    if any(dilation <= 0 for dilation in args.radial_dilations):
        raise ValueError("All radial dilations must be positive")
    if args.sigma_ref <= 0.0:
        raise ValueError("--sigma-ref must be positive")
    if args.diagnostic_smoothing <= 0:
        raise ValueError("--diagnostic-smoothing must be positive")
    if any(mode <= 0 for mode in args.spiral_modes):
        raise ValueError("--spiral-modes must be positive")


def steps_per_epoch(args: argparse.Namespace) -> int:
    return int(math.ceil(args.samples_per_epoch / args.batch_size))


def resolve_training_length(args: argparse.Namespace) -> None:
    args.steps_per_epoch = steps_per_epoch(args)
    if args.epochs is not None:
        args.steps = int(args.epochs * args.steps_per_epoch)


def make_learning_rate_schedule(args: argparse.Namespace):
    if args.lr_schedule == "constant":
        return args.lr, {}
    boundary_interval = args.lr_step_epochs * args.steps_per_epoch
    max_decays = max(0, (args.steps - 1) // boundary_interval)
    boundaries = {
        boundary_interval * i: args.lr_decay_factor
        for i in range(1, max_decays + 1)
    }
    schedule = optax.piecewise_constant_schedule(init_value=args.lr, boundaries_and_scales=boundaries)
    metadata = {
        "type": "step",
        "initial_lr": args.lr,
        "decay_factor": args.lr_decay_factor,
        "lr_step_epochs": args.lr_step_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "boundary_steps": sorted(boundaries.keys()),
    }
    return schedule, metadata


def history_end_min(args: argparse.Namespace) -> int:
    return (args.history_length - 1) * args.history_stride


def frame_for_orbit(ds, orbit: float) -> int:
    orbits = np.asarray(ds.times[0], dtype=np.float64) / ORBIT_PERIOD
    return int(np.argmin(np.abs(orbits - orbit)))


def rollout_targets_for_orbits(ds, requested_orbits: list[float]) -> list[dict]:
    targets = []
    orbits = np.asarray(ds.times[0], dtype=np.float64) / ORBIT_PERIOD
    for orbit in requested_orbits:
        span_steps = max(1, frame_for_orbit(ds, orbit))
        targets.append(
            {
                "requested_orbits": float(orbit),
                "span_steps": int(span_steps),
                "actual_duration_orbits": float(orbits[span_steps] - orbits[0]),
            }
        )
    return targets


def rollout_target_key(target: dict) -> str:
    return f"{target['requested_orbits']:.3f}_orbits"


def valid_rollout_start_ends(ds, args: argparse.Namespace, span_steps: int) -> np.ndarray:
    min_end = history_end_min(args)
    last_end = ds.n_frames - 1 - span_steps
    if last_end < min_end:
        raise ValueError(
            f"Rollout duration of {span_steps} saved-frame steps is too long for this dataset and history window"
        )
    return np.arange(min_end, last_end + 1, dtype=np.int32)


def rollout_case_source(ds) -> np.ndarray:
    if ds.test_cases.size:
        return ds.test_cases
    if ds.validation_cases.size:
        return ds.validation_cases
    if ds.train_cases.size:
        return ds.train_cases
    raise ValueError("No cases available for rollout evaluation: test, validation, and train splits are all empty.")


def valid_history_ends(
    ds,
    args: argparse.Namespace,
    max_target_frame: int | None = None,
    target_steps: int = 1,
) -> np.ndarray:
    min_end = history_end_min(args)
    last_end = ds.n_frames - 1 - target_steps * args.step_span
    if max_target_frame is not None:
        last_end = min(last_end, max_target_frame - target_steps * args.step_span)
    if last_end < min_end:
        raise ValueError("Not enough frames for the requested history length, stride, and rollout horizon")
    return np.arange(min_end, last_end + 1, dtype=np.int32)


def history_frame_ids(end_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    offsets = np.arange(args.history_length - 1, -1, -1, dtype=np.int32) * args.history_stride
    return end_ids[:, None] - offsets[None, :]


def read_history_raw(ds, cases: np.ndarray, end_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    case_grid = np.broadcast_to(cases[:, None], (cases.size, args.history_length))
    frame_grid = history_frame_ids(end_ids, args)
    return ds.read_state(case_grid, frame_grid)


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean.reshape((1,) * (values.ndim - 1) + (-1,))) / std.reshape((1,) * (values.ndim - 1) + (-1,))


def flatten_history(history: np.ndarray) -> np.ndarray:
    batch, history_len, ny, nx, channels = history.shape
    return np.transpose(history, (0, 2, 3, 1, 4)).reshape(batch, ny, nx, history_len * channels)


def flatten_history_jax(history: jnp.ndarray) -> jnp.ndarray:
    batch, history_len, ny, nx, channels = history.shape
    return jnp.transpose(history, (0, 2, 3, 1, 4)).reshape((batch, ny, nx, history_len * channels))


def make_history_batch(
    ds,
    rng: np.random.Generator,
    case_ids: np.ndarray,
    end_rows: np.ndarray,
    batch_size: int,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
    rollout_steps: int = 1,
) -> dict[str, jnp.ndarray]:
    cases = rng.choice(case_ids, size=batch_size, replace=True).astype(np.int32)
    ends = rng.choice(end_rows, size=batch_size, replace=True).astype(np.int32)
    history_raw = read_history_raw(ds, cases, ends, args)
    future_offsets = np.arange(1, rollout_steps + 1, dtype=np.int32) * args.step_span
    target_ids = ends[:, None] + future_offsets[None, :]
    case_grid = np.broadcast_to(cases[:, None], target_ids.shape)
    target_raw = ds.read_state(case_grid, target_ids)
    history = normalize(history_raw, mean, std).astype(np.float32)
    target_rollout = normalize(target_raw, mean, std).astype(np.float32)
    spans = np.full(batch_size, args.step_span, dtype=np.int32)
    t_steps = []
    dt_steps = []
    for step in range(rollout_steps):
        current_ends = ends + step * args.step_span
        t_step, dt_step = ds.read_time(cases, current_ends, spans)
        t_steps.append(t_step)
        dt_steps.append(dt_step)
    t_rollout = np.stack(t_steps, axis=1).astype(np.float32)
    dt_rollout = np.stack(dt_steps, axis=1).astype(np.float32)
    return {
        "x_history": jnp.asarray(history),
        "x_context": jnp.asarray(flatten_history(history)),
        "x_last": jnp.asarray(history[:, -1]),
        "y": jnp.asarray(target_rollout[:, 0]),
        "y_rollout": jnp.asarray(target_rollout),
        "mu": jnp.asarray(ds.read_params(cases)),
        "t": jnp.asarray(t_rollout[:, 0]),
        "dt": jnp.asarray(dt_rollout[:, 0]),
        "t_rollout": jnp.asarray(t_rollout),
        "dt_rollout": jnp.asarray(dt_rollout),
        "end_ids": jnp.asarray(ends),
        "case_ids": jnp.asarray(cases),
    }


def estimate_history_normalization(ds, rng: np.random.Generator, end_rows: np.ndarray, args: argparse.Namespace):
    remaining = args.normalization_samples
    sum_values = np.zeros(ds.n_channels, dtype=np.float64)
    sum_sq_values = np.zeros(ds.n_channels, dtype=np.float64)
    count = 0
    while remaining > 0:
        take = min(remaining, 8)
        cases = rng.choice(ds.train_cases, size=take, replace=True).astype(np.int32)
        ends = rng.choice(end_rows, size=take, replace=True).astype(np.int32)
        history = read_history_raw(ds, cases, ends, args)
        future_offsets = np.arange(1, args.train_rollout_steps + 1, dtype=np.int32) * args.step_span
        target_ids = ends[:, None] + future_offsets[None, :]
        case_grid = np.broadcast_to(cases[:, None], target_ids.shape)
        target = ds.read_state(case_grid, target_ids)
        values = np.concatenate([history.reshape((-1, ds.n_channels)), target.reshape((-1, ds.n_channels))], axis=0)
        sum_values += values.sum(axis=0)
        sum_sq_values += np.square(values, dtype=np.float64).sum(axis=0)
        count += values.shape[0]
        remaining -= take
    mean = (sum_values / count).astype(np.float32)
    var = np.maximum(sum_sq_values / count - np.square(mean.astype(np.float64)), 0.0)
    std = (np.sqrt(var) + 1.0e-6).astype(np.float32)
    return mean, std


def history_grid_inputs(batch: dict[str, jnp.ndarray], coords: np.ndarray) -> jnp.ndarray:
    x = batch["x_context"]
    b, ny, nx, _ = x.shape
    coord = jnp.broadcast_to(jnp.asarray(coords)[None, :, :, :], (b, ny, nx, coords.shape[-1]))
    cond = jnp.concatenate([batch["mu"], batch["t"], batch["dt"]], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (b, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coord, cond], axis=-1)


def make_history_model(name: str, coords: np.ndarray, args: argparse.Namespace, output_channels: int):
    def residual_output(y, batch):
        return batch["x_last"] + batch["dt"][:, None, None, :] * y

    if name == "geometry_aware_fno":
        def forward(batch):
            h = hk.Linear(args.width)(history_grid_inputs(batch, coords))
            for i in range(args.depth):
                spectral = V3.ThetaSpectralRadialConv(
                    args.width,
                    args.modes_theta,
                    args.radial_kernels,
                    args.radial_dilations,
                    args.radial_padding,
                    name=f"theta_spectral_radial_{i}",
                )(h)
                pointwise = hk.Linear(args.width, name=f"pointwise_{i}")(h)
                h = jax.nn.gelu(spectral + pointwise)
            y = hk.nets.MLP([args.width, output_channels], activation=jax.nn.gelu)(h)
            return residual_output(y, batch)

    elif name == "geometry_radial_fno":
        def forward(batch):
            h = hk.Linear(args.width)(history_grid_inputs(batch, coords))
            for i in range(args.depth):
                spectral = V3.ThetaSpectralRadialConv(
                    args.width,
                    args.modes_theta,
                    args.radial_kernels,
                    args.radial_dilations,
                    args.radial_padding,
                    modes_r=args.modes_r,
                    radial_global_mixing="reflect_fft",
                    radial_global_weight=1.0,
                    name=f"theta_spectral_radial_{i}",
                )(h)
                pointwise = hk.Linear(args.width, name=f"pointwise_{i}")(h)
                h = jax.nn.gelu(spectral + pointwise)
            y = hk.nets.MLP([args.width, output_channels], activation=jax.nn.gelu)(h)
            return residual_output(y, batch)

    elif name == "plain_fno":
        def forward(batch):
            h = hk.Linear(args.width)(history_grid_inputs(batch, coords))
            for i in range(args.depth):
                spectral = V3.PlainSpectralConv2D(args.width, args.modes_r, args.modes_theta, name=f"plain_spectral_{i}")(h)
                pointwise = hk.Linear(args.width, name=f"pointwise_{i}")(h)
                h = jax.nn.gelu(spectral + pointwise)
            y = hk.nets.MLP([args.width, output_channels], activation=jax.nn.gelu)(h)
            return residual_output(y, batch)

    elif name == "fno_reflect2d":
        def forward(batch):
            h = hk.Linear(args.width)(history_grid_inputs(batch, coords))
            for i in range(args.depth):
                spectral = V3.ReflectSpectralConv2D(
                    args.width,
                    args.modes_r,
                    args.modes_theta,
                    name=f"reflect_spectral_{i}",
                )(h)
                pointwise = hk.Linear(args.width, name=f"pointwise_{i}")(h)
                h = jax.nn.gelu(spectral + pointwise)
            y = hk.nets.MLP([args.width, output_channels], activation=jax.nn.gelu)(h)
            return residual_output(y, batch)

    elif name == "unet":
        def forward(batch):
            h = V3.conv_block(history_grid_inputs(batch, coords), args.width, "input")
            skips = []
            for level in range(args.unet_levels):
                skips.append(h)
                channels = args.width * (2 ** min(level + 1, 3))
                h = hk.Conv2D(channels, kernel_shape=3, stride=2, padding="SAME", name=f"down_{level}")(h)
                h = V3.conv_block(jax.nn.gelu(h), channels, f"down_block_{level}")
            h = V3.conv_block(h, h.shape[-1], "bottleneck")
            for level, skip in reversed(list(enumerate(skips))):
                h = jax.image.resize(h, skip.shape, method="nearest")
                h = jnp.concatenate([h, skip], axis=-1)
                h = V3.conv_block(h, skip.shape[-1], f"up_block_{level}")
            y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
            return residual_output(y, batch)

    elif name == "periodic_unet":
        def forward(batch):
            h = V3.periodic_conv_block(history_grid_inputs(batch, coords), args.width, "input", args.radial_padding)
            skips = []
            for level in range(args.unet_levels):
                skips.append(h)
                channels = args.width * (2 ** min(level + 1, 3))
                h = V3.polar_conv2d(h, channels, 3, f"down_{level}", args.radial_padding, stride=2)
                h = V3.periodic_conv_block(jax.nn.gelu(h), channels, f"down_block_{level}", args.radial_padding)
            h = V3.periodic_conv_block(h, h.shape[-1], "bottleneck", args.radial_padding)
            for level, skip in reversed(list(enumerate(skips))):
                h = jax.image.resize(h, skip.shape, method="nearest")
                h = jnp.concatenate([h, skip], axis=-1)
                h = V3.periodic_conv_block(h, skip.shape[-1], f"up_block_{level}", args.radial_padding)
            y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
            return residual_output(y, batch)

    elif name == "convlstm":
        def forward(batch):
            history = batch["x_history"]
            b, _, ny, nx, _ = history.shape
            h = jnp.zeros((b, ny, nx, args.width), dtype=history.dtype)
            c = jnp.zeros_like(h)
            cell = V3.ConvLSTMCell(args.width, name="convlstm_cell")
            for step in range(args.history_length):
                frame = history[:, step]
                context = history_grid_inputs(dict(batch, x_context=frame), coords)
                h, c = cell(context, h, c)
            for i in range(max(0, args.depth - 1)):
                h = hk.Conv2D(args.width, kernel_shape=3, padding="SAME", name=f"post_conv_{i}")(h)
                h = jax.nn.gelu(h)
            y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
            return residual_output(y, batch)

    else:
        raise ValueError(f"Unknown model: {name}")

    return hk.without_apply_rng(hk.transform(forward))


def metric_accumulator(channels: int) -> dict:
    return {
        "sum_sq": 0.0,
        "sum_sq_ch": np.zeros(channels, dtype=np.float64),
        "num_sq": 0.0,
        "den_sq": 0.0,
        "num_sq_ch": np.zeros(channels, dtype=np.float64),
        "den_sq_ch": np.zeros(channels, dtype=np.float64),
        "physics_sums": {},
        "samples": 0,
        "batches": 0,
    }


def update_metrics(acc: dict, pred_norm, target_norm, ds, args, mean, std, loss_weights) -> None:
    rmse, rmse_ch, num, den, num_ch, den_ch = V3.evaluate_predictions(
        pred_norm, target_norm, mean, std, ds.channels, loss_weights
    )
    n = pred_norm.shape[0]
    acc["sum_sq"] += (rmse**2) * n
    for i, channel in enumerate(ds.channels):
        acc["sum_sq_ch"][i] += (rmse_ch[channel] ** 2) * n
    acc["num_sq"] += num
    acc["den_sq"] += den
    acc["num_sq_ch"] += num_ch
    acc["den_sq_ch"] += den_ch
    V3.accumulate_physics_metrics(acc["physics_sums"], pred_norm, target_norm, mean, std, ds, args)
    acc["samples"] += n
    acc["batches"] += 1


def finalize_metrics(acc: dict, ds, args) -> dict:
    metrics = V3.finalize_eval_metrics(
        acc["sum_sq"],
        acc["sum_sq_ch"],
        acc["samples"],
        acc["num_sq"],
        acc["den_sq"],
        acc["num_sq_ch"],
        acc["den_sq_ch"],
        ds.channels,
        acc["batches"],
        args.rel_l2_floor,
        acc["physics_sums"],
    )
    return V3.to_jsonable(metrics)


def evaluate_next_step(model, params, ds, case_ids, end_rows, args, mean, std, loss_weights, batches: int) -> dict:
    rng = np.random.default_rng(args.seed + 910)
    acc = metric_accumulator(ds.n_channels)
    for _ in range(batches):
        batch = make_history_batch(ds, rng, case_ids, end_rows, args.batch_size, args, mean, std, rollout_steps=1)
        pred_norm = np.asarray(model.apply(params, batch))
        update_metrics(acc, pred_norm, np.asarray(batch["y"]), ds, args, mean, std, loss_weights)
    return finalize_metrics(acc, ds, args)


def predict_short_rollout(model, params, batch, args) -> np.ndarray:
    history = np.asarray(batch["x_history"])
    steps = int(np.asarray(batch["y_rollout"]).shape[1])
    preds = []
    for step in range(steps):
        step_batch = {
            "x_history": jnp.asarray(history),
            "x_context": jnp.asarray(flatten_history(history)),
            "x_last": jnp.asarray(history[:, -1]),
            "mu": batch["mu"],
            "t": batch["t_rollout"][:, step],
            "dt": batch["dt_rollout"][:, step],
            "case_ids": batch["case_ids"],
            "end_ids": batch["end_ids"],
        }
        pred = np.asarray(model.apply(params, step_batch))
        preds.append(pred)
        history = np.concatenate([history[:, 1:], pred[:, None]], axis=1)
    return np.stack(preds, axis=1)


def evaluate_train_rollout(model, params, ds, case_ids, end_rows, args, mean, std, loss_weights, batches: int) -> dict:
    rng = np.random.default_rng(args.seed + 915)
    acc = metric_accumulator(ds.n_channels)
    for _ in range(batches):
        batch = make_history_batch(
            ds,
            rng,
            case_ids,
            end_rows,
            args.batch_size,
            args,
            mean,
            std,
            rollout_steps=args.train_rollout_steps,
        )
        pred_rollout = predict_short_rollout(model, params, batch, args)
        target_rollout = np.asarray(batch["y_rollout"])
        b, steps, ny, nx, channels = pred_rollout.shape
        update_metrics(
            acc,
            pred_rollout.reshape(b * steps, ny, nx, channels),
            target_rollout.reshape(b * steps, ny, nx, channels),
            ds,
            args,
            mean,
            std,
            loss_weights,
        )
    return finalize_metrics(acc, ds, args)


def train_history_model(name, model, ds, train_ends, val_ends, args, mean, std, loss_weights):
    rng = np.random.default_rng(args.seed + 101)
    key = jax.random.PRNGKey(args.seed + 101)
    init_batch = make_history_batch(
        ds,
        rng,
        ds.train_cases,
        train_ends,
        min(args.batch_size, 2),
        args,
        mean,
        std,
        rollout_steps=args.train_rollout_steps,
    )
    params = model.init(key, init_batch)
    apply_model = jax.checkpoint(model.apply) if args.remat_model else model.apply
    lr_schedule, _ = make_learning_rate_schedule(args)
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(lr_schedule))
    opt_state = opt.init(params)
    weights = jnp.asarray(loss_weights)

    @jax.jit
    def loss_fn(params, batch):
        targets = jnp.swapaxes(batch["y_rollout"], 0, 1)
        t_steps = jnp.swapaxes(batch["t_rollout"], 0, 1)
        dt_steps = jnp.swapaxes(batch["dt_rollout"], 0, 1)

        def rollout_step(history, inputs):
            target, t_step, dt_step = inputs
            step_batch = {
                "x_history": history,
                "x_context": flatten_history_jax(history),
                "x_last": history[:, -1],
                "mu": batch["mu"],
                "t": t_step,
                "dt": dt_step,
                "case_ids": batch["case_ids"],
                "end_ids": batch["end_ids"],
            }
            pred = apply_model(params, step_batch)
            step_loss = jnp.mean(((pred - target) ** 2) * weights)
            next_state = jax.lax.stop_gradient(pred) if args.rollout_gradient == "truncated" else pred
            next_history = jnp.concatenate([history[:, 1:], next_state[:, None]], axis=1)
            return next_history, step_loss

        _, step_losses = jax.lax.scan(rollout_step, batch["x_history"], (targets, t_steps, dt_steps))
        return jnp.mean(step_losses)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = []
    best_params = params
    best_step = 0
    best_val = math.inf
    stale = 0
    for step in range(1, args.steps + 1):
        batch = make_history_batch(
            ds,
            rng,
            ds.train_cases,
            train_ends,
            args.batch_size,
            args,
            mean,
            std,
            rollout_steps=args.train_rollout_steps,
        )
        params, opt_state, loss = train_step(params, opt_state, batch)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate_train_rollout(
                model,
                params,
                ds,
                ds.validation_cases if ds.validation_cases.size else ds.train_cases,
                val_ends if val_ends.size else train_ends,
                args,
                mean,
                std,
                loss_weights,
                max(2, args.eval_batches // 2),
            )
            val_rmse = float(val["rmse"])
            history.append(
                {
                    "step": step,
                    "epoch": float(step / args.steps_per_epoch),
                    "learning_rate": float(lr_schedule(step - 1)) if callable(lr_schedule) else float(lr_schedule),
                    "batch_rollout_rmse": float(jnp.sqrt(loss)),
                    "validation_rollout_rmse": val_rmse,
                    "validation_rollout_steps": args.train_rollout_steps,
                }
            )
            current_lr = float(lr_schedule(step - 1)) if callable(lr_schedule) else float(lr_schedule)
            print(f"{name} step {step} epoch {step / args.steps_per_epoch:.2f}: rollout_val={val_rmse:.5f} lr={current_lr:.3e}")
            if val_rmse < best_val - args.early_stop_min_delta:
                best_val = val_rmse
                best_params = params
                best_step = step
                stale = 0
            else:
                stale += 1
            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(f"{name} early stopped at step {step}")
                break
    return best_params, history, best_step, best_val


def rollout_from_history_ends(model, params, ds, cases, start_ends, args, mean, std, loss_weights, span_steps: int):
    history_raw = read_history_raw(ds, cases, start_ends, args)
    history = normalize(history_raw, mean, std).astype(np.float32)
    curve = []
    trajectory_acc = metric_accumulator(ds.n_channels)
    final_acc = None
    step_offset = 0
    step_index = 0
    while step_offset + args.step_span <= span_steps:
        current_ends = start_ends + step_offset
        target_ids = current_ends + args.step_span
        spans = np.full(cases.size, args.step_span, dtype=np.int32)
        t, dt = ds.read_time(cases, current_ends, spans)
        batch = {
            "x_history": jnp.asarray(history),
            "x_context": jnp.asarray(flatten_history(history)),
            "x_last": jnp.asarray(history[:, -1]),
            "mu": jnp.asarray(ds.read_params(cases)),
            "t": jnp.asarray(t),
            "dt": jnp.asarray(dt),
        }
        pred = np.asarray(model.apply(params, batch))
        target = normalize(ds.read_state(cases, target_ids), mean, std).astype(np.float32)
        step_acc = metric_accumulator(ds.n_channels)
        update_metrics(step_acc, pred, target, ds, args, mean, std, loss_weights)
        update_metrics(trajectory_acc, pred, target, ds, args, mean, std, loss_weights)
        duration = float(np.asarray(ds.times[0, step_offset + args.step_span]) / ORBIT_PERIOD)
        row = {
            "step": step_index + 1,
            "span_steps": int(step_offset + args.step_span),
            "duration_orbits": duration,
            "mean_start_frame": float(np.mean(start_ends)),
            "mean_target_frame": float(np.mean(target_ids)),
        }
        row.update(finalize_metrics(step_acc, ds, args))
        curve.append(row)
        final_acc = step_acc
        history = np.concatenate([history[:, 1:], pred[:, None]], axis=1)
        step_offset += args.step_span
        step_index += 1
    if final_acc is None:
        raise ValueError("Rollout horizon produced no recurrent steps")
    return finalize_metrics(trajectory_acc, ds, args), finalize_metrics(final_acc, ds, args), curve


def evaluate_rollout(model, params, ds, args, mean, std, loss_weights, target: dict):
    rng = np.random.default_rng(args.seed + 920)
    cases_source = rollout_case_source(ds)
    valid_start_ends = valid_rollout_start_ends(ds, args, target["span_steps"])
    if args.rollout_start_mode == "initial":
        valid_start_ends = valid_start_ends[:1]
    summary_acc = metric_accumulator(ds.n_channels)
    final_acc = metric_accumulator(ds.n_channels)
    curve_accs: list[dict] | None = None
    curve_meta: list[dict] | None = None
    for _ in range(args.rollout_batches):
        cases = rng.choice(cases_source, size=args.batch_size, replace=True).astype(np.int32)
        start_ends = rng.choice(valid_start_ends, size=args.batch_size, replace=True).astype(np.int32)
        trajectory_metrics, final_metrics, curve = rollout_from_history_ends(
            model, params, ds, cases, start_ends, args, mean, std, loss_weights, target["span_steps"]
        )
        # Re-run accumulation from finalized rows is awkward, so store batch-weighted scalar summaries for the two top-line
        # views and use curve rows as averaged diagnostic records below.
        for source, acc in [(trajectory_metrics, summary_acc), (final_metrics, final_acc)]:
            n = int(source.get("samples", args.batch_size))
            acc["sum_sq"] += float(source["rmse"]) ** 2 * n
            for i, channel in enumerate(ds.channels):
                acc["sum_sq_ch"][i] += float(source["rmse_by_channel"][channel]) ** 2 * n
            acc["num_sq"] += 0.0
            acc["den_sq"] += 0.0
            acc["samples"] += n
            acc["batches"] += 1
            for key, value in (source.get("physics") or {}).items():
                acc["physics_sums"][key] = acc["physics_sums"].get(key, 0.0) + float(value) * n
        if curve_accs is None:
            curve_accs = [metric_accumulator(ds.n_channels) for _ in curve]
            curve_meta = [
                {"step": row["step"], "span_steps": row["span_steps"], "duration_orbits": row["duration_orbits"]}
                for row in curve
            ]
        for row, acc in zip(curve, curve_accs):
            n = args.batch_size
            acc["sum_sq"] += float(row["rmse"]) ** 2 * n
            for i, channel in enumerate(ds.channels):
                acc["sum_sq_ch"][i] += float(row["rmse_by_channel"][channel]) ** 2 * n
            acc["samples"] += n
            acc["batches"] += 1
            for key, value in (row.get("physics") or {}).items():
                acc["physics_sums"][key] = acc["physics_sums"].get(key, 0.0) + float(value) * n

    def finalize_rollout_acc(acc):
        physics = None
        if acc["physics_sums"]:
            physics = {key: float(value / acc["samples"]) for key, value in sorted(acc["physics_sums"].items())}
        return {
            "rmse": float(math.sqrt(acc["sum_sq"] / acc["samples"])),
            "rmse_by_channel": {
                channel: float(math.sqrt(acc["sum_sq_ch"][i] / acc["samples"])) for i, channel in enumerate(ds.channels)
            },
            "batches": acc["batches"],
            "samples": acc["samples"],
            "physics": physics,
        }

    curve_out = []
    for meta, acc in zip(curve_meta or [], curve_accs or []):
        row = dict(meta)
        row.update(finalize_rollout_acc(acc))
        curve_out.append(row)
    return finalize_rollout_acc(summary_acc), finalize_rollout_acc(final_acc), curve_out


def save_checkpoint(path: Path, name: str, params, args, mean, std) -> None:
    payload = {
        "format": "fargo_long_term_rollout_v3_checkpoint",
        "model": name,
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), jax.device_get(params)),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "config": vars(args),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_empty_model_plan(args, ds, rollout_targets: list[dict]) -> None:
    print("No models selected. Long-term scaffold is ready.")
    print("Fill --models with the selected models after short-term validation, for example:")
    print("  --models plain_fno geometry_aware_fno convlstm")
    target_text = ", ".join(
        f"{target['requested_orbits']:.3f} orbit -> {target['span_steps']} steps (actual {target['actual_duration_orbits']:.3f})"
        for target in rollout_targets
    )
    print(
        "Plan: history_length=%d, train_rollout_steps=%d, step_span=%d frame(s), rollout_targets=[%s]"
        % (args.history_length, args.train_rollout_steps, args.step_span, target_text)
    )
    print(
        "Training length: steps=%d, epochs=%s, samples_per_epoch=%d, batch_size=%d, steps_per_epoch=%d"
        % (
            args.steps,
            "None" if args.epochs is None else str(args.epochs),
            args.samples_per_epoch,
            args.batch_size,
            args.steps_per_epoch,
        )
    )
    print(
        "Optimizer: Adam, lr=%.3e, lr_schedule=%s, lr_step_epochs=%d, lr_decay_factor=%.3f"
        % (args.lr, args.lr_schedule, args.lr_step_epochs, args.lr_decay_factor)
    )
    print("Training mode: in-batch autoregressive rollout loss over all future training steps.")
    print("FNO-3D one-shot trajectory prediction should be added as a separate trajectory-output wrapper.")
    print(f"Dataset: cases={ds.n_cases}, frames={ds.n_frames}, grid={ds.ny}x{ds.nx}, channels={ds.channels}")


def main() -> None:
    args = parse_args()
    ds = V3.FargoMemmapDataset(args.dataset, args.channels, time_input_units="normalized", dt_units="orbits")
    coords = V3.coordinate_grid(ds.r, ds.theta)
    loss_weights = V3.spatial_loss_weights(ds.r, args.loss_weighting)
    rollout_targets = rollout_targets_for_orbits(ds, args.rollout_end_orbits)
    for target in rollout_targets:
        target["valid_start_count"] = int(valid_rollout_start_ends(ds, args, target["span_steps"]).size)
    max_rollout_span_steps = max(target["span_steps"] for target in rollout_targets)
    train_ends = valid_history_ends(ds, args, target_steps=args.train_rollout_steps)
    val_ends = train_ends
    if not args.models:
        print_empty_model_plan(args, ds, rollout_targets)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    norm_rng = np.random.default_rng(args.seed + 111)
    mean, std = estimate_history_normalization(ds, norm_rng, train_ends, args)
    _, lr_schedule_metadata = make_learning_rate_schedule(args)
    print("Loaded long-term dataset:", args.dataset)
    print("Normalization mean:", dict(zip(args.channels, mean.tolist())))
    print("Normalization std:", dict(zip(args.channels, std.tolist())))
    print("Rollout targets:")
    for target in rollout_targets:
        print(
            "  requested=%.4f orbit, span_steps=%d, actual_duration=%.4f orbit, valid_starts=%d"
            % (
                target["requested_orbits"],
                target["span_steps"],
                target["actual_duration_orbits"],
                target["valid_start_count"],
            )
        )

    results = {}
    histories = {}
    for name in args.models:
        model_start = time.perf_counter()
        model = make_history_model(name, coords, args, ds.n_channels)
        train_start = time.perf_counter()
        params, history, best_step, best_val = train_history_model(
            name, model, ds, train_ends, val_ends, args, mean, std, loss_weights
        )
        train_wall_time = time.perf_counter() - train_start
        checkpoint_path = None
        if args.save_checkpoints:
            checkpoint_path = args.output_dir / f"{name}_long_term_checkpoint.pkl"
            save_checkpoint(checkpoint_path, name, params, args, mean, std)
        next_step_start = time.perf_counter()
        next_step_validation = evaluate_next_step(
            model,
            params,
            ds,
            ds.validation_cases if ds.validation_cases.size else ds.train_cases,
            val_ends,
            args,
            mean,
            std,
            loss_weights,
            args.eval_batches,
        )
        next_step_wall_time = time.perf_counter() - next_step_start
        train_rollout_val_start = time.perf_counter()
        train_rollout_validation = evaluate_train_rollout(
            model,
            params,
            ds,
            ds.validation_cases if ds.validation_cases.size else ds.train_cases,
            val_ends if val_ends.size else train_ends,
            args,
            mean,
            std,
            loss_weights,
            args.eval_batches,
        )
        train_rollout_validation_wall_time = time.perf_counter() - train_rollout_val_start
        rollout_horizons = {}
        rollout_wall_times = {}
        for target in rollout_targets:
            rollout_start = time.perf_counter()
            rollout_summary, rollout_final, rollout_curve = evaluate_rollout(
                model, params, ds, args, mean, std, loss_weights, target
            )
            rollout_wall_times[rollout_target_key(target)] = time.perf_counter() - rollout_start
            rollout_horizons[rollout_target_key(target)] = {
                "requested_orbits": target["requested_orbits"],
                "actual_duration_orbits": target["actual_duration_orbits"],
                "span_steps": target["span_steps"],
                "valid_start_count": target["valid_start_count"],
                "trajectory": rollout_summary,
                "final_frame": rollout_final,
                "curve": rollout_curve,
            }
        total_wall_time = time.perf_counter() - model_start
        results[name] = V3.to_jsonable(
            LongTermModelResult(
                train_rollout_validation=train_rollout_validation,
                next_step_validation=next_step_validation,
                rollout_horizons=rollout_horizons,
                wall_time_sec={
                    "training": train_wall_time,
                    "next_step_validation": next_step_wall_time,
                    "train_rollout_validation": train_rollout_validation_wall_time,
                    "rollout_by_horizon": rollout_wall_times,
                    "total_model": total_wall_time,
                },
                trained_steps=max(row.get("step", 0) for row in history),
                best_step=best_step,
                best_validation_rollout_rmse=best_val,
                checkpoint=str(checkpoint_path) if checkpoint_path else None,
            )
        )
        histories[name] = history
        print(f"{name} long-term benchmark finished in {total_wall_time / 60.0:.2f} min")

    summary = {
        "setup": {
            "source_v3_module": str(getattr(V3, "__file_path__", "")),
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "models": args.models,
            "channels": args.channels,
            "history_length": args.history_length,
            "step_span": args.step_span,
            "history_stride": args.history_stride,
            "train_rollout_steps": args.train_rollout_steps,
            "training_mode": "autoregressive_rollout_loss",
            "rollout_gradient": args.rollout_gradient,
            "remat_model": args.remat_model,
            "optimizer": "adam",
            "learning_rate": {
                "initial_lr": args.lr,
                "schedule": args.lr_schedule,
                "lr_step_epochs": args.lr_step_epochs,
                "lr_decay_factor": args.lr_decay_factor,
                "schedule_metadata": lr_schedule_metadata,
            },
            "training_length": {
                "steps": args.steps,
                "epochs": args.epochs,
                "samples_per_epoch": args.samples_per_epoch,
                "batch_size": args.batch_size,
                "steps_per_epoch": args.steps_per_epoch,
            },
            "model_family_scope": "recurrent_2d; FNO-3D one-shot trajectory output requires a separate wrapper",
            "rollout_start_mode": args.rollout_start_mode,
            "rollout_horizon_semantics": "duration from each sampled history-window end",
            "rollout_end_orbits": args.rollout_end_orbits,
            "rollout_targets": rollout_targets,
            "rollout_span_steps": [target["span_steps"] for target in rollout_targets],
            "max_rollout_span_steps": max_rollout_span_steps,
            "rollout_steps_by_horizon": {
                rollout_target_key(target): int(target["span_steps"] // args.step_span)
                for target in rollout_targets
            },
            "width": args.width,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "radial_kernels": args.radial_kernels,
            "radial_dilations": args.radial_dilations,
            "radial_padding": args.radial_padding,
            "unet_levels": args.unet_levels,
            "convlstm_steps": args.convlstm_steps,
            "physics_metrics": {
                "enabled": args.physics_metrics,
                "sigma_ref": args.sigma_ref,
                "sigma_ref_slope": args.sigma_ref_slope,
                "sigma_ref_profile": "sigma_ref * (r / planet_radius) ** (-sigma_ref_slope)",
                "spiral_modes": args.spiral_modes,
                "gap_r_range": [args.gap_r_min, args.gap_r_max],
                "ring_r_range": [args.ring_r_min, args.ring_r_max],
                "spiral_r_range": [args.spiral_r_min, args.spiral_r_max],
            },
        },
        "results": results,
    }
    (args.output_dir / "long_term_metrics.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "long_term_loss_history.json").write_text(json.dumps(V3.to_jsonable(histories), indent=2))
    print(f"Saved long-term benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

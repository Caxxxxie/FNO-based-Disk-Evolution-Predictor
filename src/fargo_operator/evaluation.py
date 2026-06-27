"""Numerical, rollout, and physics diagnostics for FARGO operators."""

from __future__ import annotations

import argparse
import math

import jax.numpy as jnp
import numpy as np

from .constants import EvalMetrics
from .data import FargoMemmapDataset, make_batch, make_consistency_batch, sample_pairs
from .models import set_time_span


def evaluate_predictions(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    channels: list[str],
    loss_weights: np.ndarray,
):
    err = pred_norm - target_norm
    weighted_err2 = (err**2) * loss_weights
    rmse_by_channel_values = np.sqrt(np.mean(weighted_err2, axis=(0, 1, 2)))
    pred = pred_norm * std.reshape((1, 1, 1, -1)) + mean.reshape((1, 1, 1, -1))
    target = target_norm * std.reshape((1, 1, 1, -1)) + mean.reshape((1, 1, 1, -1))
    phys_err2 = ((pred - target) ** 2) * loss_weights
    phys_target2 = (target**2) * loss_weights
    return (
        float(np.sqrt(np.mean(weighted_err2))),
        {channel: float(value) for channel, value in zip(channels, rmse_by_channel_values)},
        float(np.sum(phys_err2)),
        float(np.sum(phys_target2)),
        np.sum(phys_err2, axis=(0, 1, 2)),
        np.sum(phys_target2, axis=(0, 1, 2)),
    )


def physical_fields(values_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values_norm * std.reshape((1, 1, 1, -1)) + mean.reshape((1, 1, 1, -1))


def radial_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded], axis=0)


def sigma_reference_profile(r: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    radius = np.maximum(np.asarray(r, dtype=np.float64), 1.0e-12)
    planet_radius = max(float(args.planet_radius), 1.0e-12)
    return float(args.sigma_ref) * (radius / planet_radius) ** (-float(args.sigma_ref_slope))


def count_local_extrema(values: np.ndarray, kind: str, threshold: float) -> int:
    if values.size < 3:
        return 0
    center = values[1:-1]
    if kind == "min":
        mask = (center < values[:-2]) & (center < values[2:]) & (center < threshold)
    elif kind == "max":
        mask = (center > values[:-2]) & (center > values[2:]) & (center > threshold)
    else:
        raise ValueError(f"Unknown extremum kind: {kind}")
    return int(np.count_nonzero(mask))


def gap_ring_features(sigma: np.ndarray, r: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    sigma_bar = sigma.mean(axis=2)
    sigma_bar = radial_smooth(sigma_bar, args.diagnostic_smoothing)
    sigma_ref = sigma_reference_profile(r, args)
    gap_mask = (r >= args.gap_r_min) & (r <= args.gap_r_max)
    ring_mask = (r >= args.ring_r_min) & (r <= args.ring_r_max)
    if not np.any(gap_mask):
        raise ValueError("Gap diagnostic radial window is empty")
    if not np.any(ring_mask):
        raise ValueError("Ring diagnostic radial window is empty")
    r_gap = r[gap_mask]
    r_ring = r[ring_mask]
    gap_profiles = sigma_bar[:, gap_mask] / np.maximum(sigma_ref[gap_mask][None, :], 1.0e-12)
    ring_profiles = sigma_bar[:, ring_mask] / np.maximum(sigma_ref[ring_mask][None, :], 1.0e-12)

    gap_min_idx = np.argmin(gap_profiles, axis=1)
    gap_min = np.take_along_axis(gap_profiles, gap_min_idx[:, None], axis=1)[:, 0]
    gap_center = r_gap[gap_min_idx]
    gap_depth = 1.0 / np.maximum(gap_min, 1.0e-12)
    gap_threshold = np.sqrt(np.maximum(gap_min, 1.0e-24))
    gap_width = np.zeros(gap_profiles.shape[0], dtype=np.float64)
    gap_count = np.zeros(gap_profiles.shape[0], dtype=np.float64)
    for i, profile in enumerate(gap_profiles):
        below = profile <= gap_threshold[i]
        gap_width[i] = float(r_gap[below][-1] - r_gap[below][0]) if np.any(below) else 0.0
        gap_count[i] = count_local_extrema(profile, "min", args.gap_detection_fraction)

    ring_peak_idx = np.argmax(ring_profiles, axis=1)
    ring_peak = np.take_along_axis(ring_profiles, ring_peak_idx[:, None], axis=1)[:, 0]
    ring_peak_radius = r_ring[ring_peak_idx]
    ring_contrast = ring_peak
    ring_count = np.zeros(ring_profiles.shape[0], dtype=np.float64)
    for i, profile in enumerate(ring_profiles):
        ring_count[i] = count_local_extrema(profile, "max", args.ring_detection_factor)

    return {
        "gap_depth": gap_depth,
        "gap_center": gap_center,
        "gap_width": gap_width,
        "gap_count": gap_count,
        "ring_peak_radius": ring_peak_radius,
        "ring_contrast": ring_contrast,
        "ring_count": ring_count,
    }


def wrap_phase(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def add_gap_ring_metrics(metrics: dict[str, float], pred_sigma: np.ndarray, target_sigma: np.ndarray, r: np.ndarray, args):
    pred = gap_ring_features(pred_sigma, r, args)
    target = gap_ring_features(target_sigma, r, args)
    for name in ["gap_depth", "gap_center", "gap_width", "ring_peak_radius", "ring_contrast"]:
        metrics[f"{name}_abs_error"] = float(np.mean(np.abs(pred[name] - target[name])))
        metrics[f"target_{name}_mean"] = float(np.mean(target[name]))
    for name in ["gap_depth", "ring_contrast"]:
        denom = np.maximum(np.abs(target[name]), 1.0e-12)
        metrics[f"{name}_rel_error_pct"] = float(100.0 * np.mean(np.abs(pred[name] - target[name]) / denom))
    for name in ["gap_count", "ring_count"]:
        metrics[f"{name}_abs_error"] = float(np.mean(np.abs(pred[name] - target[name])))
        metrics[f"{name}_match_fraction"] = float(np.mean(pred[name] == target[name]))
        metrics[f"target_{name}_mean"] = float(np.mean(target[name]))


def add_spiral_metrics(
    metrics: dict[str, float],
    pred_sigma: np.ndarray,
    target_sigma: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    args: argparse.Namespace,
):
    spiral_mask = (r >= args.spiral_r_min) & (r <= args.spiral_r_max)
    if not np.any(spiral_mask):
        raise ValueError("Spiral diagnostic radial window is empty")
    pred_delta = pred_sigma - pred_sigma.mean(axis=2, keepdims=True)
    target_delta = target_sigma - target_sigma.mean(axis=2, keepdims=True)
    r_spiral = r[spiral_mask]
    for mode in args.spiral_modes:
        basis = np.exp(-1j * mode * theta).astype(np.complex64)
        pred_coeff = np.mean(pred_delta * basis[None, None, :], axis=2)[:, spiral_mask]
        target_coeff = np.mean(target_delta * basis[None, None, :], axis=2)[:, spiral_mask]
        phase_diff = wrap_phase(np.angle(pred_coeff) - np.angle(target_coeff))
        target_amp = np.abs(target_coeff)
        pred_amp = np.abs(pred_coeff)
        amp_weight = target_amp / np.maximum(np.mean(target_amp), 1.0e-12)
        metrics[f"spiral_phase_mae_m{mode}_rad"] = float(np.mean(np.abs(phase_diff)))
        metrics[f"spiral_phase_amp_weighted_mae_m{mode}_rad"] = float(
            np.sum(np.abs(phase_diff) * target_amp) / np.maximum(np.sum(target_amp), 1.0e-12)
        )
        amp_num = np.sum((pred_amp - target_amp) ** 2)
        amp_den = np.sum(target_amp**2)
        metrics[f"spiral_amplitude_rel_l2_m{mode}_pct"] = float(100.0 * np.sqrt(amp_num / max(amp_den, 1.0e-24)))
        if r_spiral.size >= 3:
            pred_phase = np.unwrap(np.angle(pred_coeff), axis=1)
            target_phase = np.unwrap(np.angle(target_coeff), axis=1)
            pred_theta = -pred_phase / mode
            target_theta = -target_phase / mode
            pred_dtheta = np.gradient(pred_theta, r_spiral, axis=1)
            target_dtheta = np.gradient(target_theta, r_spiral, axis=1)
            slope_diff = pred_dtheta - target_dtheta
            pred_pitch = np.arctan2(1.0, np.maximum(np.abs(r_spiral[None, :] * pred_dtheta), 1.0e-12))
            target_pitch = np.arctan2(1.0, np.maximum(np.abs(r_spiral[None, :] * target_dtheta), 1.0e-12))
            metrics[f"spiral_phase_slope_amp_weighted_mae_m{mode}"] = float(
                np.sum(np.abs(slope_diff) * amp_weight) / np.maximum(np.sum(amp_weight), 1.0e-12)
            )
            metrics[f"spiral_pitch_amp_weighted_mae_m{mode}_rad"] = float(
                np.sum(np.abs(pred_pitch - target_pitch) * amp_weight) / np.maximum(np.sum(amp_weight), 1.0e-12)
            )


def compute_physics_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    channels: list[str],
    r: np.ndarray,
    theta: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    if not args.physics_metrics or "log_sigma" not in channels:
        return {}
    log_sigma_idx = channels.index("log_sigma")
    pred = physical_fields(pred_norm, mean, std)
    target = physical_fields(target_norm, mean, std)
    pred_sigma = np.power(10.0, np.clip(pred[..., log_sigma_idx], -12.0, 12.0))
    target_sigma = np.power(10.0, np.clip(target[..., log_sigma_idx], -12.0, 12.0))
    metrics: dict[str, float] = {}
    add_gap_ring_metrics(metrics, pred_sigma, target_sigma, np.asarray(r), args)
    add_spiral_metrics(metrics, pred_sigma, target_sigma, np.asarray(r), np.asarray(theta), args)
    return metrics


def accumulate_physics_metrics(
    sums: dict[str, float],
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ds: FargoMemmapDataset,
    args: argparse.Namespace,
) -> None:
    metrics = compute_physics_metrics(pred_norm, target_norm, mean, std, ds.channels, ds.r, ds.theta, args)
    n = pred_norm.shape[0]
    for key, value in metrics.items():
        if np.isfinite(value):
            sums[key] = sums.get(key, 0.0) + float(value) * n


def finalize_eval_metrics(
    sum_sq: float,
    sum_sq_ch: np.ndarray,
    samples: int,
    num_sq: float,
    den_sq: float,
    num_sq_ch: np.ndarray,
    den_sq_ch: np.ndarray,
    channels: list[str],
    batches: int,
    rel_l2_floor: float,
    physics_sums: dict[str, float] | None = None,
) -> EvalMetrics:
    floor_sq = rel_l2_floor**2
    physics = None
    if physics_sums:
        physics = {key: float(value / samples) for key, value in sorted(physics_sums.items())}
    return EvalMetrics(
        rmse=float(math.sqrt(sum_sq / samples)),
        rmse_by_channel={channel: float(math.sqrt(sum_sq_ch[i] / samples)) for i, channel in enumerate(channels)},
        rel_l2_pct=float(100.0 * math.sqrt(num_sq / max(den_sq, floor_sq))),
        rel_l2_pct_by_channel={
            channel: float(100.0 * math.sqrt(num_sq_ch[i] / max(den_sq_ch[i], floor_sq)))
            for i, channel in enumerate(channels)
        },
        batches=batches,
        samples=samples,
        physics=physics,
    )


def evaluate_model(
    model,
    params,
    ds,
    case_ids,
    pair_rows,
    args,
    mean,
    std,
    loss_weights,
    max_batches=None,
    include_physics: bool = True,
) -> EvalMetrics:
    batches = args.eval_batches if max_batches is None else max_batches
    rng = np.random.default_rng(args.seed + 991)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(batches):
        batch = make_batch(ds, rng, case_ids, pair_rows, args.batch_size, mean, std)
        pred_norm = np.asarray(model.apply(params, batch))
        target_norm = np.asarray(batch["y"])
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            pred_norm, target_norm, mean, std, ds.channels, loss_weights
        )
        if include_physics:
            accumulate_physics_metrics(physics_sums, pred_norm, target_norm, mean, std, ds, args)
        n = pred_norm.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        batches,
        args.rel_l2_floor,
        physics_sums,
    )


def evaluate_persistence(ds, case_ids, pair_rows, args, mean, std, loss_weights, max_batches=None) -> EvalMetrics:
    batches = args.eval_batches if max_batches is None else max_batches
    rng = np.random.default_rng(args.seed + 882)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(batches):
        batch = make_batch(ds, rng, case_ids, pair_rows, args.batch_size, mean, std)
        pred_norm = np.asarray(batch["x"])
        target_norm = np.asarray(batch["y"])
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            pred_norm, target_norm, mean, std, ds.channels, loss_weights
        )
        accumulate_physics_metrics(physics_sums, pred_norm, target_norm, mean, std, ds, args)
        n = pred_norm.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        batches,
        args.rel_l2_floor,
        physics_sums,
    )


def linear_extrapolate_raw(ds: FargoMemmapDataset, cases: np.ndarray, starts: np.ndarray, spans: np.ndarray) -> np.ndarray:
    prev_starts = np.maximum(starts - 1, 0)
    x_prev = ds.read_state(cases, prev_starts)
    x = ds.read_state(cases, starts)
    t_prev = ds.read_time_values("code", cases, prev_starts)
    t0 = ds.read_time_values("code", cases, starts)
    t1 = ds.read_time_values("code", cases, starts + spans)
    dt_prev = np.maximum(t0 - t_prev, 1.0e-12)
    ratio = np.where(starts > 0, (t1 - t0) / dt_prev, 0.0).astype(np.float32)
    return x + ratio.reshape((-1, 1, 1, 1)) * (x - x_prev)


def evaluate_linear_extrapolation(ds, case_ids, pair_rows, args, mean, std, loss_weights, max_batches=None) -> EvalMetrics:
    batches = args.eval_batches if max_batches is None else max_batches
    rng = np.random.default_rng(args.seed + 883)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(batches):
        cases, starts, spans = sample_pairs(rng, case_ids, pair_rows, args.batch_size)
        pred_raw = linear_extrapolate_raw(ds, cases, starts, spans)
        target_raw = ds.read_state(cases, starts + spans)
        pred_norm = (pred_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        target_norm = (target_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            pred_norm, target_norm, mean, std, ds.channels, loss_weights
        )
        accumulate_physics_metrics(physics_sums, pred_norm, target_norm, mean, std, ds, args)
        n = pred_norm.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        batches,
        args.rel_l2_floor,
        physics_sums,
    )


def evaluate_rollout(model, params, ds, case_ids, args, mean, std, loss_weights, horizon: int) -> EvalMetrics:
    rng = np.random.default_rng(args.seed + 773)
    starts = np.arange(0, ds.n_frames - horizon, dtype=np.int32)
    rows = np.asarray([(int(start), 1, 0) for start in starts], dtype=np.int32)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(args.eval_batches):
        cases, start_ids, _ = sample_pairs(rng, case_ids, rows, args.batch_size)
        prev_ids = np.maximum(start_ids - 1, 0)
        x_prev_raw = ds.read_state(cases, prev_ids)
        x_raw = ds.read_state(cases, start_ids)
        target_raw = ds.read_state(cases, start_ids + horizon)
        previous = (x_prev_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        current = (x_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        for step in range(horizon):
            spans = np.ones(args.batch_size, dtype=np.int32)
            t, dt = ds.read_time(cases, start_ids + step, spans)
            batch = {
                "x_prev": jnp.asarray(previous),
                "x": jnp.asarray(current),
                "mu": jnp.asarray(ds.read_params(cases)),
                "t": jnp.asarray(t),
                "dt": jnp.asarray(dt),
            }
            next_state = np.asarray(model.apply(params, batch))
            previous = current
            current = next_state
        target = (target_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            current, target, mean, std, ds.channels, loss_weights
        )
        accumulate_physics_metrics(physics_sums, current, target, mean, std, ds, args)
        n = current.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        args.eval_batches,
        args.rel_l2_floor,
        physics_sums,
    )


def evaluate_persistence_rollout(ds, case_ids, args, mean, std, loss_weights, horizon: int) -> EvalMetrics:
    rng = np.random.default_rng(args.seed + 774)
    starts = np.arange(0, ds.n_frames - horizon, dtype=np.int32)
    rows = np.asarray([(int(start), 1, 0) for start in starts], dtype=np.int32)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(args.eval_batches):
        cases, start_ids, _ = sample_pairs(rng, case_ids, rows, args.batch_size)
        x_raw = ds.read_state(cases, start_ids)
        target_raw = ds.read_state(cases, start_ids + horizon)
        pred_norm = (x_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        target_norm = (target_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            pred_norm, target_norm, mean, std, ds.channels, loss_weights
        )
        accumulate_physics_metrics(physics_sums, pred_norm, target_norm, mean, std, ds, args)
        n = pred_norm.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        args.eval_batches,
        args.rel_l2_floor,
        physics_sums,
    )


def evaluate_linear_extrapolation_rollout(ds, case_ids, args, mean, std, loss_weights, horizon: int) -> EvalMetrics:
    rng = np.random.default_rng(args.seed + 775)
    starts = np.arange(0, ds.n_frames - horizon, dtype=np.int32)
    rows = np.asarray([(int(start), 1, 0) for start in starts], dtype=np.int32)
    sum_sq = 0.0
    sum_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    num_sq = 0.0
    den_sq = 0.0
    num_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    den_sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    physics_sums: dict[str, float] = {}
    samples = 0
    for _ in range(args.eval_batches):
        cases, start_ids, _ = sample_pairs(rng, case_ids, rows, args.batch_size)
        prev_ids = np.maximum(start_ids - 1, 0)
        previous = ds.read_state(cases, prev_ids)
        current = ds.read_state(cases, start_ids)
        target_raw = ds.read_state(cases, start_ids + horizon)
        for step in range(horizon):
            step_ids = start_ids + step
            next_ids = step_ids + 1
            prev_time_ids = np.maximum(step_ids - 1, 0)
            t_prev = ds.read_time_values("code", cases, prev_time_ids)
            t0 = ds.read_time_values("code", cases, step_ids)
            t1 = ds.read_time_values("code", cases, next_ids)
            dt_prev = np.maximum(t0 - t_prev, 1.0e-12)
            ratio = np.where(step_ids > 0, (t1 - t0) / dt_prev, 0.0).astype(np.float32)
            predicted = current + ratio.reshape((-1, 1, 1, 1)) * (current - previous)
            previous = current
            current = predicted
        pred_norm = (current - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        target_norm = (target_raw - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
            pred_norm, target_norm, mean, std, ds.channels, loss_weights
        )
        accumulate_physics_metrics(physics_sums, pred_norm, target_norm, mean, std, ds, args)
        n = pred_norm.shape[0]
        sum_sq += (rmse**2) * n
        for i, channel in enumerate(ds.channels):
            sum_sq_ch[i] += (rmse_ch[channel] ** 2) * n
        num_sq += batch_num
        den_sq += batch_den
        num_sq_ch += batch_num_ch
        den_sq_ch += batch_den_ch
        samples += n
    return finalize_eval_metrics(
        sum_sq,
        sum_sq_ch,
        samples,
        num_sq,
        den_sq,
        num_sq_ch,
        den_sq_ch,
        ds.channels,
        args.eval_batches,
        args.rel_l2_floor,
        physics_sums,
    )


def evaluate_semigroup(model, params, ds, case_ids, args, mean, std, loss_weights) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(args.seed + 664)
    span_a, span_b = args.consistency_spans
    sq = 0.0
    sq_ch = np.zeros(ds.n_channels, dtype=np.float64)
    samples = 0
    for _ in range(args.eval_batches):
        batch = make_consistency_batch(ds, rng, case_ids, span_a, span_b, args.batch_size, mean, std)
        direct = np.asarray(model.apply(params, set_time_span(batch, batch["t_ab"], batch["dt_ab"])))
        first = model.apply(params, batch)
        second_batch = dict(batch)
        second_batch["x"] = first
        second_batch["t"] = batch["t_b"]
        second_batch["dt"] = batch["dt_b"]
        second = np.asarray(model.apply(params, second_batch))
        err = direct - second
        weighted_err2 = (err**2) * loss_weights
        n = err.shape[0]
        sq += float(np.mean(weighted_err2)) * n
        values = np.mean(weighted_err2, axis=(0, 1, 2))
        sq_ch += np.asarray(values) * n
        samples += n
    return float(math.sqrt(sq / samples)), {
        channel: float(math.sqrt(sq_ch[i] / samples)) for i, channel in enumerate(ds.channels)
    }

"""Evaluation metrics for FARGO operator models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from fargo_data import FargoMemmapDataset, make_batch, make_consistency_batch, normalize_state, sample_pairs, set_time_span


@dataclass
class EvalMetrics:
    rmse: float
    rmse_by_channel: dict[str, float]
    rel_l2_pct: float
    rel_l2_pct_by_channel: dict[str, float]
    batches: int
    samples: int


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
) -> EvalMetrics:
    floor_sq = rel_l2_floor**2
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
    )


def _empty_accumulators(ds: FargoMemmapDataset):
    return {
        "sum_sq": 0.0,
        "sum_sq_ch": np.zeros(ds.n_channels, dtype=np.float64),
        "num_sq": 0.0,
        "den_sq": 0.0,
        "num_sq_ch": np.zeros(ds.n_channels, dtype=np.float64),
        "den_sq_ch": np.zeros(ds.n_channels, dtype=np.float64),
        "samples": 0,
    }


def _accumulate_prediction(
    acc: dict,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    ds: FargoMemmapDataset,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
) -> None:
    rmse, rmse_ch, batch_num, batch_den, batch_num_ch, batch_den_ch = evaluate_predictions(
        pred_norm, target_norm, mean, std, ds.channels, loss_weights
    )
    n = pred_norm.shape[0]
    acc["sum_sq"] += (rmse**2) * n
    for i, channel in enumerate(ds.channels):
        acc["sum_sq_ch"][i] += (rmse_ch[channel] ** 2) * n
    acc["num_sq"] += batch_num
    acc["den_sq"] += batch_den
    acc["num_sq_ch"] += batch_num_ch
    acc["den_sq_ch"] += batch_den_ch
    acc["samples"] += n


def _finalize_accumulator(acc: dict, ds: FargoMemmapDataset, batches: int, rel_l2_floor: float) -> EvalMetrics:
    return finalize_eval_metrics(
        acc["sum_sq"],
        acc["sum_sq_ch"],
        acc["samples"],
        acc["num_sq"],
        acc["den_sq"],
        acc["num_sq_ch"],
        acc["den_sq_ch"],
        ds.channels,
        batches,
        rel_l2_floor,
    )


def evaluate_model(model, params, ds, case_ids, pair_rows, args, mean, std, loss_weights, max_batches=None) -> EvalMetrics:
    batches = args.eval_batches if max_batches is None else max_batches
    rng = np.random.default_rng(args.seed + 991)
    acc = _empty_accumulators(ds)
    for _ in range(batches):
        batch = make_batch(ds, rng, case_ids, pair_rows, args.batch_size, mean, std)
        pred_norm = np.asarray(model.apply(params, batch))
        target_norm = np.asarray(batch["y"])
        _accumulate_prediction(acc, pred_norm, target_norm, ds, mean, std, loss_weights)
    return _finalize_accumulator(acc, ds, batches, args.rel_l2_floor)


def evaluate_persistence(ds, case_ids, pair_rows, args, mean, std, loss_weights, max_batches=None) -> EvalMetrics:
    batches = args.eval_batches if max_batches is None else max_batches
    rng = np.random.default_rng(args.seed + 882)
    acc = _empty_accumulators(ds)
    for _ in range(batches):
        batch = make_batch(ds, rng, case_ids, pair_rows, args.batch_size, mean, std)
        pred_norm = np.asarray(batch["x"])
        target_norm = np.asarray(batch["y"])
        _accumulate_prediction(acc, pred_norm, target_norm, ds, mean, std, loss_weights)
    return _finalize_accumulator(acc, ds, batches, args.rel_l2_floor)


def evaluate_model_by_span(model, params, ds, case_ids, pair_rows, args, mean, std, loss_weights) -> dict[str, EvalMetrics]:
    """Evaluate an operator separately for each temporal span in ``pair_rows``."""
    metrics = {}
    for span in sorted(set(int(row[1]) for row in pair_rows)):
        rows = pair_rows[pair_rows[:, 1] == span]
        if rows.shape[0] == 0:
            continue
        metrics[str(span)] = evaluate_model(model, params, ds, case_ids, rows, args, mean, std, loss_weights)
    return metrics


def evaluate_persistence_by_span(ds, case_ids, pair_rows, args, mean, std, loss_weights) -> dict[str, EvalMetrics]:
    """Evaluate the persistence baseline separately for each temporal span."""
    metrics = {}
    for span in sorted(set(int(row[1]) for row in pair_rows)):
        rows = pair_rows[pair_rows[:, 1] == span]
        if rows.shape[0] == 0:
            continue
        metrics[str(span)] = evaluate_persistence(ds, case_ids, rows, args, mean, std, loss_weights)
    return metrics


def evaluate_rollout(model, params, ds, case_ids, args, mean, std, loss_weights, horizon: int) -> EvalMetrics:
    rng = np.random.default_rng(args.seed + 773)
    starts = np.arange(0, ds.n_frames - horizon, dtype=np.int32)
    rows = np.asarray([(int(start), 1, 0) for start in starts], dtype=np.int32)
    acc = _empty_accumulators(ds)
    for _ in range(args.eval_batches):
        cases, start_ids, _ = sample_pairs(rng, case_ids, rows, args.batch_size)
        target_raw = ds.read_state(cases, start_ids + horizon)
        current = normalize_state(ds.read_state(cases, start_ids), mean, std)
        for step in range(horizon):
            spans = np.ones(args.batch_size, dtype=np.int32)
            t, dt = ds.read_time(cases, start_ids + step, spans)
            batch = {
                "x": jnp.asarray(current),
                "mu": jnp.asarray(ds.read_params(cases)),
                "t": jnp.asarray(t),
                "dt": jnp.asarray(dt),
            }
            current = np.asarray(model.apply(params, batch))
        target = normalize_state(target_raw, mean, std)
        _accumulate_prediction(acc, current, target, ds, mean, std, loss_weights)
    return _finalize_accumulator(acc, ds, args.eval_batches, args.rel_l2_floor)


def evaluate_rollouts(model, params, ds, case_ids, args, mean, std, loss_weights, horizons: list[int]) -> dict[str, EvalMetrics]:
    """Evaluate autoregressive rollouts at multiple horizons."""
    return {
        str(int(horizon)): evaluate_rollout(model, params, ds, case_ids, args, mean, std, loss_weights, int(horizon))
        for horizon in horizons
    }


def evaluate_persistence_rollout(ds, case_ids, args, mean, std, loss_weights, horizon: int) -> EvalMetrics:
    rng = np.random.default_rng(args.seed + 774)
    starts = np.arange(0, ds.n_frames - horizon, dtype=np.int32)
    rows = np.asarray([(int(start), 1, 0) for start in starts], dtype=np.int32)
    acc = _empty_accumulators(ds)
    for _ in range(args.eval_batches):
        cases, start_ids, _ = sample_pairs(rng, case_ids, rows, args.batch_size)
        pred_norm = normalize_state(ds.read_state(cases, start_ids), mean, std)
        target_norm = normalize_state(ds.read_state(cases, start_ids + horizon), mean, std)
        _accumulate_prediction(acc, pred_norm, target_norm, ds, mean, std, loss_weights)
    return _finalize_accumulator(acc, ds, args.eval_batches, args.rel_l2_floor)


def evaluate_persistence_rollouts(ds, case_ids, args, mean, std, loss_weights, horizons: list[int]) -> dict[str, EvalMetrics]:
    """Evaluate persistence rollouts at multiple horizons."""
    return {
        str(int(horizon)): evaluate_persistence_rollout(ds, case_ids, args, mean, std, loss_weights, int(horizon))
        for horizon in horizons
    }


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

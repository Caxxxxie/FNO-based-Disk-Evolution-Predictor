"""Serialization, checkpointing, and plotting helpers for FARGO benchmarks."""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np

from fargo_data import FargoMemmapDataset
from fargo_metrics import EvalMetrics


@dataclass
class ModelResult:
    train: EvalMetrics
    validation: EvalMetrics
    heldout_time: EvalMetrics
    heldout_parameter: EvalMetrics
    heldout_parameter_time: EvalMetrics
    rollout: EvalMetrics | None
    speed_ms_per_batch: float | None
    trained_steps: int | None = None
    best_step: int | None = None
    best_validation_rmse: float | None = None
    heldout_parameter_time_by_span: dict[str, EvalMetrics] | None = None
    rollout_by_horizon: dict[str, EvalMetrics] | None = None
    semigroup_rmse: float | None = None
    semigroup_rmse_by_channel: dict[str, float] | None = None


def to_jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def speed_ms_per_batch(model, params, batch, repeats: int) -> float:
    y = model.apply(params, batch)
    y.block_until_ready()
    import time

    start = time.perf_counter()
    for _ in range(repeats):
        y = model.apply(params, batch)
    y.block_until_ready()
    return 1000.0 * (time.perf_counter() - start) / repeats


def save_loss_plot(history_by_model: dict[str, list[dict]], output_dir: Path) -> None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Skipping loss plot: {exc}")
        return
    for name, history in history_by_model.items():
        rows = [row for row in history if "step" in row]
        if not rows:
            continue
        steps = [row["step"] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        batch_values = np.asarray([row.get("batch_rmse", 0.0) for row in rows], dtype=np.float64)
        train_values = np.asarray([row.get("train_rmse", np.nan) for row in rows], dtype=np.float64)
        val_values = np.asarray([row.get("validation_rmse", np.nan) for row in rows], dtype=np.float64)
        for key, label in [
            ("batch_rmse", "batch"),
            ("train_rmse", "train"),
            ("validation_rmse", "validation"),
            ("consistency_rmse", "consistency"),
        ]:
            values = [row.get(key, 0.0) for row in rows]
            if any(value != 0.0 for value in values):
                ax.plot(steps, values, marker="o", linewidth=1.5, label=label)
        if batch_values.size >= 5 and np.any(batch_values != 0.0):
            window = min(7, batch_values.size)
            kernel = np.ones(window, dtype=np.float64) / window
            smooth = np.convolve(batch_values, kernel, mode="same")
            ax.plot(steps, smooth, linewidth=2.0, label=f"batch ma{window}")
        ax.set_xlabel("step")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{name} loss curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_{name}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        if np.any(np.isfinite(train_values)):
            ax.plot(steps, train_values, marker="o", linewidth=1.8, label="train")
        if np.any(np.isfinite(val_values)):
            ax.plot(steps, val_values, marker="o", linewidth=1.8, label="validation")
        ax.set_xlabel("step")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{name} train/validation loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_{name}_train_val.png", dpi=180)
        plt.close(fig)


def summarize_splits(pair_splits: dict[str, np.ndarray]) -> dict:
    summary = {}
    for split, rows in pair_splits.items():
        spans = {}
        bins = {}
        for _, span, bin_id in rows:
            spans[str(int(span))] = spans.get(str(int(span)), 0) + 1
            bins[str(int(bin_id))] = bins.get(str(int(bin_id)), 0) + 1
        summary[split] = {"pairs": int(rows.shape[0]), "spans": spans, "bins": bins}
    return summary


def save_model_checkpoint(
    output_dir: Path,
    name: str,
    params,
    ds: FargoMemmapDataset,
    args,
    mean: np.ndarray,
    std: np.ndarray,
    pair_splits: dict[str, np.ndarray],
) -> Path:
    checkpoint = {
        "format": "fargo_operator_v3_fno_checkpoint",
        "model": name,
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x), jax.device_get(params)),
        "channels": list(ds.channels),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "dataset": str(args.dataset),
        "model_config": {
            "width": args.width,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "radial_kernels": list(args.radial_kernels),
            "radial_dilations": list(args.radial_dilations),
            "radial_padding": args.radial_padding,
            "residual": True,
            "time_input_units": args.time_input_units,
            "dt_units": args.dt_units,
        },
        "training_config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "grad_clip_norm": args.grad_clip_norm,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "seed": args.seed,
            "loss_weighting": args.loss_weighting,
            "temporal_bins": args.temporal_bins,
            "temporal_train_frac": args.temporal_train_frac,
            "temporal_val_frac": args.temporal_val_frac,
            "fno_spans": list(args.fno_spans),
            "fno_flow_spans": list(args.fno_flow_spans),
        },
        "case_splits": {
            "train": np.asarray(ds.train_cases, dtype=np.int32),
            "validation": np.asarray(ds.validation_cases, dtype=np.int32),
            "test": np.asarray(ds.test_cases, dtype=np.int32),
        },
        "temporal_pair_splits": {split: np.asarray(rows, dtype=np.int32) for split, rows in pair_splits.items()},
    }
    path = output_dir / f"{name}_checkpoint.pkl"
    with path.open("wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

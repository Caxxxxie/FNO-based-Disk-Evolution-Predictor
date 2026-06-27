#!/usr/bin/env python3
"""Diagnose rollout error growth from an existing FARGO operator checkpoint.

The script loads a saved checkpoint, rolls it out autoregressively, and
decomposes the error energy into low/mid/high Fourier bands at each rollout step.
It also reports simple mass-drift and radial-flux proxy errors.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORBIT_PERIOD = 2.0 * math.pi


def load_v3_module():
    candidates = [
        ROOT / "scripts" / "train_fargo_operator.py",
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
            return module
    raise FileNotFoundError("Could not find train_fargo_operator.py or benchmark_fargo_operators_v3.py")


V3 = load_v3_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=64, help="Autoregressive rollout horizon in saved-frame steps.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--case-split",
        choices=["test", "validation", "train", "all"],
        default="test",
        help="Which case split to sample for diagnostics. Use all for external validation datasets.",
    )
    parser.add_argument("--loss-weighting", choices=["uniform", "area"], default="area")
    parser.add_argument("--rel-l2-floor", type=float, default=1.0e-6)
    parser.add_argument("--low-frac", type=float, default=0.15, help="Normalized radial wavenumber cutoff for low band.")
    parser.add_argument("--mid-frac", type=float, default=0.45, help="Normalized radial wavenumber cutoff for mid band.")
    parser.add_argument("--theta-low", type=int, default=3, help="Azimuthal mode cutoff for low band.")
    parser.add_argument("--theta-mid", type=int, default=16, help="Azimuthal mode cutoff for mid band.")
    parser.add_argument("--channels", nargs="+", default=None, help="Override checkpoint channel list.")
    parser.add_argument("--jax-platform", choices=["default", "cpu", "gpu", "cuda"], default="default")
    args = parser.parse_args()
    if args.jax_platform != "default":
        platform = "cuda" if args.jax_platform == "gpu" else args.jax_platform
        jax.config.update("jax_platforms", platform)
        args.jax_platform = platform
    return args


def checkpoint_args(payload: dict) -> SimpleNamespace:
    model_config = payload.get("model_config", {})
    training_config = payload.get("training_config", {})
    return SimpleNamespace(
        width=int(model_config.get("width", 64)),
        depth=int(model_config.get("depth", 4)),
        modes_r=int(model_config.get("modes_r", 24)),
        modes_theta=int(model_config.get("modes_theta", 32)),
        radial_kernels=list(model_config.get("radial_kernels", [3, 5, 5])),
        radial_dilations=list(model_config.get("radial_dilations", [1, 2, 4])),
        radial_padding=model_config.get("radial_padding", "edge"),
        radial_global_mixing=model_config.get("radial_global_mixing", "none"),
        radial_global_weight=float(model_config.get("radial_global_weight", 1.0)),
        unet_levels=int(model_config.get("unet_levels", 3)),
        convlstm_steps=int(model_config.get("convlstm_steps", 10)),
        time_input_units=model_config.get("time_input_units", "normalized"),
        dt_units=model_config.get("dt_units", "orbits"),
        loss_weighting=training_config.get("loss_weighting", "area"),
    )


def denormalize(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_norm * mean.reshape((1, 1, 1, -1)) + std.reshape((1, 1, 1, -1))


def spectral_band_masks(ny: int, nx: int, args: argparse.Namespace) -> dict[str, np.ndarray]:
    kr = np.fft.fftfreq(ny) * ny
    kt = np.fft.rfftfreq(nx) * nx
    kr_norm = np.abs(kr) / max(1.0, ny / 2.0)
    kt_abs = np.abs(kt)
    rr = kr_norm[:, None]
    tt = kt_abs[None, :]
    low = (rr <= args.low_frac) & (tt <= args.theta_low)
    mid = (~low) & (rr <= args.mid_frac) & (tt <= args.theta_mid)
    high = ~(low | mid)
    return {"low": low, "mid": mid, "high": high}


def band_energy_fraction(error_raw: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    # error_raw shape: batch, radial, theta, channels
    err_ft = np.fft.rfft2(error_raw, axes=(1, 2))
    energy = np.abs(err_ft) ** 2
    out: dict[str, dict[str, float]] = {}
    total_all = float(np.sum(energy))
    out["all"] = {
        f"{band}_frac": float(np.sum(energy[:, mask, :]) / max(total_all, 1.0e-30))
        for band, mask in masks.items()
    }
    for channel_idx in range(error_raw.shape[-1]):
        ch_energy = energy[..., channel_idx]
        total = float(np.sum(ch_energy))
        out[str(channel_idx)] = {
            f"{band}_frac": float(np.sum(ch_energy[:, mask]) / max(total, 1.0e-30))
            for band, mask in masks.items()
        }
    return out


def mass_and_flux_metrics(pred_raw: np.ndarray, target_raw: np.ndarray, channels: list[str], r: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "log_sigma" not in channels:
        return metrics
    log_idx = channels.index("log_sigma")
    pred_sigma = np.power(10.0, np.clip(pred_raw[..., log_idx], -12.0, 12.0))
    target_sigma = np.power(10.0, np.clip(target_raw[..., log_idx], -12.0, 12.0))
    weights = r.reshape((1, -1, 1))
    pred_mass = np.sum(pred_sigma * weights, axis=(1, 2))
    target_mass = np.sum(target_sigma * weights, axis=(1, 2))
    metrics["mass_rel_error_pct"] = float(100.0 * np.mean(np.abs(pred_mass - target_mass) / np.maximum(np.abs(target_mass), 1.0e-12)))
    metrics["mass_signed_drift_pct"] = float(100.0 * np.mean((pred_mass - target_mass) / np.maximum(np.abs(target_mass), 1.0e-12)))
    if "delta_v_r" in channels:
        vr_idx = channels.index("delta_v_r")
        pred_flux = np.sum(pred_sigma * pred_raw[..., vr_idx] * weights, axis=(1, 2))
        target_flux = np.sum(target_sigma * target_raw[..., vr_idx] * weights, axis=(1, 2))
        denom = np.maximum(np.mean(np.abs(target_flux)), 1.0e-12)
        metrics["radial_flux_proxy_abs_error"] = float(np.mean(np.abs(pred_flux - target_flux)))
        metrics["radial_flux_proxy_rel_error_pct"] = float(100.0 * np.mean(np.abs(pred_flux - target_flux)) / denom)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("format") != "fargo_operator_v3_checkpoint":
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')}")

    channels = args.channels or list(payload.get("channels", ["log_sigma", "delta_v_r", "delta_v_theta"]))
    ckpt_cfg = checkpoint_args(payload)
    ds = V3.FargoMemmapDataset(args.dataset, channels, ckpt_cfg.time_input_units, ckpt_cfg.dt_units)
    coords = V3.coordinate_grid(ds.r, ds.theta)
    model_name = payload.get("model") or payload.get("model_config", {}).get("architecture")
    model = V3.make_model(model_name, coords, ckpt_cfg, ds.n_channels)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload["params"])
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    loss_weights = V3.spatial_loss_weights(ds.r, args.loss_weighting)
    masks = spectral_band_masks(ds.ny, ds.nx, args)
    rng = np.random.default_rng(args.seed)
    if args.case_split == "all":
        case_ids = np.arange(ds.n_cases, dtype=np.int32)
    elif args.case_split == "train":
        case_ids = ds.train_cases
    elif args.case_split == "validation":
        case_ids = ds.validation_cases
    else:
        case_ids = ds.test_cases
    if case_ids.size == 0:
        raise ValueError(f"Requested case split {args.case_split!r} is empty")
    valid_starts = np.arange(0, ds.n_frames - args.horizon, dtype=np.int32)

    accum: dict[int, dict[str, float]] = {}
    count: dict[int, int] = {}
    channel_names = list(ds.channels)

    for _ in range(args.batches):
        cases = rng.choice(case_ids, size=args.batch_size, replace=True).astype(np.int32)
        start_ids = rng.choice(valid_starts, size=args.batch_size, replace=True).astype(np.int32)
        prev_ids = np.maximum(start_ids - 1, 0)
        previous = (ds.read_state(cases, prev_ids) - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        current = (ds.read_state(cases, start_ids) - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
        for step in range(1, args.horizon + 1):
            spans = np.ones(args.batch_size, dtype=np.int32)
            t, dt = ds.read_time(cases, start_ids + step - 1, spans)
            batch = {
                "x_prev": jnp.asarray(previous),
                "x": jnp.asarray(current),
                "mu": jnp.asarray(ds.read_params(cases)),
                "t": jnp.asarray(t),
                "dt": jnp.asarray(dt),
            }
            pred = np.asarray(model.apply(params, batch))
            target = (ds.read_state(cases, start_ids + step) - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))
            rmse, rmse_ch, num, den, num_ch, den_ch = V3.evaluate_predictions(pred, target, mean, std, channel_names, loss_weights)
            pred_raw = denormalize(pred, mean, std)
            target_raw = denormalize(target, mean, std)
            error_raw = pred_raw - target_raw
            band_fracs = band_energy_fraction(error_raw, masks)
            row = {
                "step": float(step),
                "duration_orbits": float(np.asarray(ds.times[0, step]) / ORBIT_PERIOD),
                "rmse": float(rmse),
                "rel_l2_pct": float(100.0 * math.sqrt(num / max(den, args.rel_l2_floor**2))),
            }
            for i, channel in enumerate(channel_names):
                row[f"{channel}_rmse"] = float(rmse_ch[channel])
                row[f"{channel}_rel_l2_pct"] = float(100.0 * math.sqrt(num_ch[i] / max(den_ch[i], args.rel_l2_floor**2)))
                for band in ("low", "mid", "high"):
                    row[f"{channel}_{band}_freq_error_frac"] = band_fracs[str(i)][f"{band}_frac"]
            for band in ("low", "mid", "high"):
                row[f"all_{band}_freq_error_frac"] = band_fracs["all"][f"{band}_frac"]
            row.update(mass_and_flux_metrics(pred_raw, target_raw, channel_names, np.asarray(ds.r)))
            if step not in accum:
                accum[step] = {key: 0.0 for key in row}
                count[step] = 0
            for key, value in row.items():
                accum[step][key] = accum[step].get(key, 0.0) + float(value)
            count[step] += 1
            previous = current
            current = pred

    rows = []
    for step in sorted(accum):
        rows.append({key: value / count[step] for key, value in accum[step].items()})
    summary = {
        "setup": {
            "dataset": str(args.dataset),
            "checkpoint": str(args.checkpoint),
            "model": model_name,
            "channels": channel_names,
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "batches": args.batches,
            "case_split": args.case_split,
            "evaluated_cases": int(case_ids.size),
            "frequency_bands": {
                "low": {"radial_norm_max": args.low_frac, "theta_mode_max": args.theta_low},
                "mid": {"radial_norm_max": args.mid_frac, "theta_mode_max": args.theta_mid},
                "high": "remaining modes",
            },
        },
        "curve": rows,
    }
    json_path = args.output_dir / "rollout_spectral_diagnostics.json"
    csv_path = args.output_dir / "rollout_spectral_diagnostics.csv"
    json_path.write_text(json.dumps(V3.to_jsonable(summary), indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved spectral diagnostics to {json_path}")
    print(f"Saved spectral diagnostics CSV to {csv_path}")


if __name__ == "__main__":
    main()

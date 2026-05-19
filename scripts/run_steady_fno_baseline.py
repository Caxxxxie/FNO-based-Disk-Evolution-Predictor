#!/usr/bin/env python3
"""Run an FNO sanity check on the original steady PPDONet mapping.

The target is the bundled pretrained PPDONet `single_log_sigma` model. This
answers a narrow question before time-dependent work: can an FNO-style grid
operator learn the baseline steady map mu -> log_sigma on the same disk grid?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from ppdonet_steady import (
    load_single_log_sigma_source,
    predict_log_sigma,
    raw_to_normalized,
    sample_raw_parameters,
)
from fno import coordinate_channels, make_fno_regressor
from training import prediction_speed_ms, rmse, train_regressor


jax.config.update("jax_platforms", "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--num-train", type=int, default=64)
    parser.add_argument("--num-test", type=int, default=32)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--modes-r", type=int, default=8)
    parser.add_argument("--modes-theta", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--speed-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "steady_fno_baseline")
    return parser.parse_args()


def make_batch(fields: np.ndarray, mu: np.ndarray, mean: float, std: float) -> dict[str, jnp.ndarray]:
    n, ny, nx = fields.shape
    zeros = np.zeros((n, ny, nx, 1), dtype=np.float32)
    y = ((fields - mean) / std)[..., None].astype(np.float32)
    return {
        "x": jnp.asarray(zeros),
        "mu": jnp.asarray(mu.astype(np.float32)),
        "y": jnp.asarray(y),
    }


def unnormalized_rmse(model, params, batch: dict[str, jnp.ndarray], std: float) -> float:
    pred = model.apply(params, batch)
    return float(jnp.sqrt(jnp.mean((pred - batch["y"]) ** 2)) * std)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = load_single_log_sigma_source()
    raw_train = sample_raw_parameters(source, args.num_train, args.seed)
    raw_test = sample_raw_parameters(source, args.num_test, args.seed + 1)
    y_train = predict_log_sigma(source, raw_train, args.ny, args.nx)
    y_test = predict_log_sigma(source, raw_test, args.ny, args.nx)
    mu_train = raw_to_normalized(raw_train, source)
    mu_test = raw_to_normalized(raw_test, source)

    mean = float(y_train.mean())
    std = float(y_train.std() + 1.0e-6)
    train = make_batch(y_train, mu_train, mean, std)
    test = make_batch(y_test, mu_test, mean, std)

    coords = coordinate_channels(args.ny, args.nx)
    model = make_fno_regressor(coords, args.width, args.modes_r, args.modes_theta, args.depth)
    params, _, last_loss = train_regressor(
        model,
        train,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    train_rmse_norm = rmse(model, params, train)
    test_rmse_norm = rmse(model, params, test)
    summary = {
        "setup": {
            "target": "pretrained PPDONet single_log_sigma predictions",
            "ny": args.ny,
            "nx": args.nx,
            "num_train": args.num_train,
            "num_test": args.num_test,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "normalization_mean": mean,
            "normalization_std": std,
            "parameter_names": source.parameter_names,
            "raw_u_min": source.raw_u_min.tolist(),
            "raw_u_max": source.raw_u_max.tolist(),
        },
        "fno": {
            "last_train_loss": last_loss,
            "train_rmse_normalized": train_rmse_norm,
            "test_rmse_normalized": test_rmse_norm,
            "train_rmse_log_sigma": unnormalized_rmse(model, params, train, std),
            "test_rmse_log_sigma": unnormalized_rmse(model, params, test, std),
            "speed_ms_per_batch": prediction_speed_ms(model, params, test, args.speed_repeats),
        },
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate saved v3 operator checkpoints at multiple rollout horizons."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fargo_operator.checkpoints import load_fargo_checkpoint
from src.fargo_operator.constants import to_jsonable
from src.fargo_operator.data import FargoMemmapDataset, coordinate_grid, spatial_loss_weights
from src.fargo_operator.evaluation import evaluate_rollout
from src.fargo_operator.models import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, default=None, help="Override checkpoint dataset path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--loss-weighting", choices=["uniform", "area"], default=None)
    parser.add_argument("--physics-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jax-platform", choices=["default", "cpu", "gpu", "cuda"], default="default")
    parser.add_argument("--rel-l2-floor", type=float, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return load_fargo_checkpoint(path)


def namespace_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> argparse.Namespace:
    model_config = checkpoint["model_config"]
    training_config = checkpoint.get("training_config", {})
    dataset = args.dataset if args.dataset is not None else Path(checkpoint["dataset"])
    return argparse.Namespace(
        dataset=dataset,
        channels=list(checkpoint["channels"]),
        batch_size=args.batch_size or int(training_config.get("batch_size", 8)),
        eval_batches=args.eval_batches,
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        modes_r=int(model_config["modes_r"]),
        modes_theta=int(model_config["modes_theta"]),
        radial_kernels=list(model_config.get("radial_kernels", [3, 5, 5])),
        radial_dilations=list(model_config.get("radial_dilations", [1, 2, 4])),
        radial_padding=model_config.get("radial_padding", "edge"),
        radial_global_mixing=model_config.get("radial_global_mixing", "none"),
        radial_global_weight=float(model_config.get("radial_global_weight", 1.0)),
        unet_levels=int(model_config.get("unet_levels", 3)),
        convlstm_steps=int(model_config.get("convlstm_steps", 2)),
        time_input_units=model_config.get("time_input_units", "normalized"),
        dt_units=model_config.get("dt_units", "orbits"),
        loss_weighting=args.loss_weighting or training_config.get("loss_weighting", "area"),
        rel_l2_floor=float(args.rel_l2_floor or training_config.get("rel_l2_floor", 1.0e-6)),
        physics_metrics=bool(args.physics_metrics),
        planet_radius=1.0,
        gap_r_min=0.6,
        gap_r_max=1.4,
        ring_r_min=0.6,
        ring_r_max=2.2,
        spiral_r_min=0.5,
        spiral_r_max=2.0,
        sigma_ref=1.0,
        sigma_ref_slope=0.5,
        gap_detection_fraction=0.9,
        ring_detection_factor=1.05,
        diagnostic_smoothing=5,
        spiral_modes=[1, 2, 3],
        seed=int(args.seed if args.seed is not None else training_config.get("seed", 7)),
    )


def evaluate_checkpoint(path: Path, args: argparse.Namespace) -> tuple[str, dict]:
    checkpoint = load_checkpoint(path)
    eval_args = namespace_from_checkpoint(checkpoint, args)
    ds = FargoMemmapDataset(
        eval_args.dataset,
        eval_args.channels,
        eval_args.time_input_units,
        eval_args.dt_units,
    )
    coords = coordinate_grid(ds.r, ds.theta)
    model_name = str(checkpoint["model"])
    model = make_model(model_name, coords, eval_args, len(eval_args.channels))
    params = checkpoint["params"]
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    loss_weights = spatial_loss_weights(ds.r, eval_args.loss_weighting)
    case_ids = ds.test_cases if ds.test_cases.size else ds.train_cases
    horizon_results = {}
    for horizon in args.horizons:
        if horizon >= ds.n_frames:
            print(f"Skipping horizon {horizon}: dataset only has {ds.n_frames} frames")
            continue
        start = time.perf_counter()
        metrics = evaluate_rollout(
            model,
            params,
            ds,
            case_ids,
            eval_args,
            mean,
            std,
            loss_weights,
            int(horizon),
        )
        horizon_results[str(int(horizon))] = {
            "metrics": to_jsonable(metrics),
            "wall_time_sec": time.perf_counter() - start,
        }
        rel = horizon_results[str(int(horizon))]["metrics"]["rel_l2_pct"]
        print(f"{model_name} rollout@{horizon}: relL2={rel:.4f}%")
    return model_name, {
        "checkpoint": str(path),
        "dataset": str(eval_args.dataset),
        "batch_size": eval_args.batch_size,
        "eval_batches": eval_args.eval_batches,
        "horizons": horizon_results,
    }


def main() -> None:
    args = parse_args()
    if args.jax_platform != "default":
        import jax

        platform = "cuda" if args.jax_platform == "gpu" else args.jax_platform
        jax.config.update("jax_platforms", platform)
        args.jax_platform = platform
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": {
            "checkpoints": [str(path) for path in args.checkpoint],
            "dataset_override": str(args.dataset) if args.dataset else None,
            "horizons": args.horizons,
            "eval_batches": args.eval_batches,
            "batch_size_override": args.batch_size,
            "loss_weighting_override": args.loss_weighting,
            "physics_metrics": args.physics_metrics,
            "jax_platform": args.jax_platform,
        },
        "results": {},
    }
    for checkpoint_path in args.checkpoint:
        model_name, result = evaluate_checkpoint(checkpoint_path, args)
        key = model_name
        if key in payload["results"]:
            key = checkpoint_path.stem
        payload["results"][key] = result
    output_path = args.output_dir / "rollout_metrics.json"
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved rollout metrics to {output_path}")


if __name__ == "__main__":
    main()

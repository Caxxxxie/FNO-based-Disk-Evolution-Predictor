#!/usr/bin/env python3
"""Benchmark FNO operators on the large transient FARGO dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "fargo_transient_10orbits_128f")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "fargo_operator_benchmark_v3")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["log_sigma", "delta_v_r", "delta_v_theta"],
        help="State channels to train. Defaults to perturbation velocities, not full velocities.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fno"],
        choices=["fno", "fno_flow"],
        help="fno is the main v3 baseline; fno_flow is the multi-span/semigroup extension.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes-r", type=int, default=12, help="Legacy 2D-FNO option; v2 uses nonperiodic radial local conv.")
    parser.add_argument("--modes-theta", type=int, default=24)
    parser.add_argument("--radial-kernel-size", type=int, default=None, help="Compatibility alias for a single radial kernel.")
    parser.add_argument("--radial-kernels", type=int, nargs="+", default=[3, 5, 5])
    parser.add_argument("--radial-dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--radial-padding", choices=["zero", "edge", "reflect"], default="edge")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--speed-repeats", type=int, default=10)
    parser.add_argument("--normalization-samples", type=int, default=512)
    parser.add_argument("--temporal-bins", type=int, default=8)
    parser.add_argument("--temporal-train-frac", type=float, default=0.70)
    parser.add_argument("--temporal-val-frac", type=float, default=0.15)
    parser.add_argument("--fno-spans", type=int, nargs="+", default=[1])
    parser.add_argument("--fno-flow-spans", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--consistency-spans", type=int, nargs=2, default=[1, 1], metavar=("SPAN_A", "SPAN_B"))
    parser.add_argument("--rollout-train-weight", type=float, default=0.0)
    parser.add_argument("--rollout-train-horizon", type=int, default=1)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Autoregressive horizons reported in metrics.json.",
    )
    parser.add_argument("--time-input-units", choices=["normalized", "orbits", "code"], default="normalized")
    parser.add_argument("--dt-units", choices=["normalized", "orbits", "code"], default="orbits")
    parser.add_argument("--loss-weighting", choices=["uniform", "area"], default="area")
    parser.add_argument("--rel-l2-floor", type=float, default=1.0e-6)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--jax-platform", choices=["default", "cpu", "gpu"], default="default")
    parser.add_argument("--save-loss-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.jax_platform != "default":
        import jax

        jax.config.update("jax_platforms", args.jax_platform)
    if args.temporal_bins <= 0:
        raise ValueError("--temporal-bins must be positive")
    if not (0.0 < args.temporal_train_frac < 1.0):
        raise ValueError("--temporal-train-frac must be in (0, 1)")
    if not (0.0 <= args.temporal_val_frac < 1.0):
        raise ValueError("--temporal-val-frac must be in [0, 1)")
    if args.temporal_train_frac + args.temporal_val_frac >= 1.0:
        raise ValueError("temporal train + validation fractions must be < 1")
    if any(span <= 0 for span in args.fno_spans + args.fno_flow_spans + args.consistency_spans):
        raise ValueError("all spans must be positive")
    if args.rollout_horizon <= 0 or any(horizon <= 0 for horizon in args.rollout_horizons):
        raise ValueError("rollout horizons must be positive")
    if args.rollout_train_weight < 0.0:
        raise ValueError("--rollout-train-weight must be nonnegative")
    if args.rollout_train_horizon <= 0:
        raise ValueError("--rollout-train-horizon must be positive")
    if args.rollout_horizon not in args.rollout_horizons:
        args.rollout_horizons = sorted(set(args.rollout_horizons + [args.rollout_horizon]))
    if args.radial_kernel_size is not None:
        args.radial_kernels = [args.radial_kernel_size]
        args.radial_dilations = [1]
    if len(args.radial_kernels) != len(args.radial_dilations):
        raise ValueError("--radial-kernels and --radial-dilations must have the same length")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in args.radial_kernels):
        raise ValueError("all radial kernels must be positive odd integers")
    if any(dilation <= 0 for dilation in args.radial_dilations):
        raise ValueError("all radial dilations must be positive")
    if args.rel_l2_floor <= 0.0:
        raise ValueError("--rel-l2-floor must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    from fargo_benchmark import run_benchmark

    run_benchmark(args)


if __name__ == "__main__":
    main()

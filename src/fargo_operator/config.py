"""Command-line configuration for v3.1 FARGO operator experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from .constants import ANALYTIC_BASELINE_NAMES, MODEL_CHOICES, ROOT, TRAINABLE_MODEL_NAMES


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
    parser.add_argument("--models", nargs="+", default=["fno"], choices=MODEL_CHOICES)
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
    parser.add_argument(
        "--radial-global-mixing",
        choices=["none", "reflect_fft"],
        default="none",
        help="Optional radial global mixing branch for geometry-aware FNO layers.",
    )
    parser.add_argument(
        "--radial-global-weight",
        type=float,
        default=1.0,
        help="Relative weight for the radial global mixing branch.",
    )
    parser.add_argument("--unet-levels", type=int, default=3)
    parser.add_argument("--convlstm-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument(
        "--rollout-train-weight",
        type=float,
        default=0.0,
        help="Optional autoregressive rollout loss weight during training.",
    )
    parser.add_argument(
        "--rollout-train-horizon",
        type=int,
        default=1,
        help="Number of one-step autoregressive predictions used by rollout training loss.",
    )
    parser.add_argument("--time-input-units", choices=["normalized", "orbits", "code"], default="normalized")
    parser.add_argument("--dt-units", choices=["normalized", "orbits", "code"], default="orbits")
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
    parser.add_argument("--jax-platform", choices=["default", "cpu", "gpu", "cuda"], default="default")
    parser.add_argument("--save-loss-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if "all" in args.models:
        args.models = list(ANALYTIC_BASELINE_NAMES) + list(TRAINABLE_MODEL_NAMES)
    if args.jax_platform != "default":
        platform = "cuda" if args.jax_platform == "gpu" else args.jax_platform
        jax.config.update("jax_platforms", platform)
        args.jax_platform = platform
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
    if args.rollout_train_weight < 0.0:
        raise ValueError("--rollout-train-weight must be nonnegative")
    if args.rollout_train_horizon <= 0:
        raise ValueError("--rollout-train-horizon must be positive")
    if args.radial_kernel_size is not None:
        args.radial_kernels = [args.radial_kernel_size]
        args.radial_dilations = [1]
    if len(args.radial_kernels) != len(args.radial_dilations):
        raise ValueError("--radial-kernels and --radial-dilations must have the same length")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in args.radial_kernels):
        raise ValueError("all radial kernels must be positive odd integers")
    if any(dilation <= 0 for dilation in args.radial_dilations):
        raise ValueError("all radial dilations must be positive")
    if args.radial_global_weight < 0.0:
        raise ValueError("--radial-global-weight must be nonnegative")
    if args.unet_levels <= 0:
        raise ValueError("--unet-levels must be positive")
    if args.convlstm_steps <= 0:
        raise ValueError("--convlstm-steps must be positive")
    if args.rel_l2_floor <= 0.0:
        raise ValueError("--rel-l2-floor must be positive")
    if not (args.gap_r_min < args.gap_r_max):
        raise ValueError("--gap-r-min must be smaller than --gap-r-max")
    if not (args.ring_r_min < args.ring_r_max):
        raise ValueError("--ring-r-min must be smaller than --ring-r-max")
    if not (args.spiral_r_min < args.spiral_r_max):
        raise ValueError("--spiral-r-min must be smaller than --spiral-r-max")
    if args.sigma_ref <= 0.0:
        raise ValueError("--sigma-ref must be positive")
    if args.diagnostic_smoothing <= 0:
        raise ValueError("--diagnostic-smoothing must be positive")
    if any(mode <= 0 for mode in args.spiral_modes):
        raise ValueError("--spiral-modes must be positive")
    return args

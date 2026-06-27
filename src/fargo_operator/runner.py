"""End-to-end v3.1 benchmark orchestration."""

from __future__ import annotations

import json

import numpy as np

from .config import parse_args
from .constants import ModelResult, to_jsonable
from .data import (
    FargoMemmapDataset,
    coordinate_grid,
    estimate_normalization,
    make_batch,
    spatial_loss_weights,
    split_temporal_pairs,
)
from .evaluation import (
    evaluate_linear_extrapolation,
    evaluate_linear_extrapolation_rollout,
    evaluate_model,
    evaluate_persistence,
    evaluate_persistence_rollout,
    evaluate_rollout,
    evaluate_semigroup,
)
from .models import make_model
from .outputs import save_loss_plot, save_model_checkpoint, speed_ms_per_batch, summarize_splits
from .training import train_model


def run_linear_extrapolation(ds, pair_splits, args, mean, std, loss_weights) -> ModelResult:
    rollout = evaluate_linear_extrapolation_rollout(
        ds,
        ds.test_cases if ds.test_cases.size else ds.train_cases,
        args,
        mean,
        std,
        loss_weights,
        args.rollout_horizon,
    )
    return ModelResult(
        train=evaluate_linear_extrapolation(ds, ds.train_cases, pair_splits["train"], args, mean, std, loss_weights),
        validation=evaluate_linear_extrapolation(
            ds,
            ds.validation_cases if ds.validation_cases.size else ds.train_cases,
            pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_time=evaluate_linear_extrapolation(ds, ds.train_cases, pair_splits["test"], args, mean, std, loss_weights),
        heldout_parameter=evaluate_linear_extrapolation(
            ds,
            ds.test_cases if ds.test_cases.size else ds.validation_cases,
            pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_parameter_time=evaluate_linear_extrapolation(
            ds,
            ds.test_cases if ds.test_cases.size else ds.validation_cases,
            pair_splits["test"],
            args,
            mean,
            std,
            loss_weights,
        ),
        rollout=rollout,
        speed_ms_per_batch=None,
    )


def run_model(
    name,
    ds,
    coords,
    train_cases,
    val_cases,
    test_cases,
    pair_splits,
    spans,
    args,
    mean,
    std,
    loss_weights,
    seed_offset,
):
    model = make_model(name, coords, args, ds.n_channels)
    use_consistency = name == "fno_flow" and args.consistency_weight > 0.0
    params, history = train_model(
        name,
        model,
        ds,
        train_cases,
        val_cases if val_cases.size else train_cases,
        pair_splits["train"],
        pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
        args,
        mean,
        std,
        loss_weights,
        seed_offset,
        use_consistency,
    )
    if args.save_checkpoints:
        checkpoint_path = save_model_checkpoint(args.output_dir, name, params, ds, args, mean, std, pair_splits)
        print(f"Saved {name} checkpoint to {checkpoint_path}")
    speed_batch = make_batch(ds, np.random.default_rng(args.seed + 333), test_cases if test_cases.size else train_cases, pair_splits["test"], args.batch_size, mean, std)
    rollout = evaluate_rollout(
        model,
        params,
        ds,
        test_cases if test_cases.size else train_cases,
        args,
        mean,
        std,
        loss_weights,
        args.rollout_horizon,
    )
    sg = sg_ch = None
    if name == "fno_flow":
        sg, sg_ch = evaluate_semigroup(
            model,
            params,
            ds,
            test_cases if test_cases.size else train_cases,
            args,
            mean,
            std,
            loss_weights,
        )
    best = next((row for row in reversed(history) if "best_step" in row), {})
    result = ModelResult(
        train=evaluate_model(model, params, ds, train_cases, pair_splits["train"], args, mean, std, loss_weights),
        validation=evaluate_model(
            model,
            params,
            ds,
            val_cases if val_cases.size else train_cases,
            pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_time=evaluate_model(model, params, ds, train_cases, pair_splits["test"], args, mean, std, loss_weights),
        heldout_parameter=evaluate_model(
            model,
            params,
            ds,
            test_cases if test_cases.size else val_cases,
            pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_parameter_time=evaluate_model(
            model,
            params,
            ds,
            test_cases if test_cases.size else val_cases,
            pair_splits["test"],
            args,
            mean,
            std,
            loss_weights,
        ),
        rollout=rollout,
        speed_ms_per_batch=speed_ms_per_batch(model, params, speed_batch, args.speed_repeats),
        trained_steps=max([row.get("step", 0) for row in history]),
        best_step=best.get("best_step"),
        best_validation_rmse=best.get("best_validation_rmse"),
        semigroup_rmse=sg,
        semigroup_rmse_by_channel=sg_ch,
    )
    return result, history


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = FargoMemmapDataset(args.dataset, args.channels, args.time_input_units, args.dt_units)
    coords = coordinate_grid(ds.r, ds.theta)
    loss_weights = spatial_loss_weights(ds.r, args.loss_weighting)
    print(f"Loaded {args.dataset}: cases={ds.n_cases}, frames={ds.n_frames}, grid={ds.ny}x{ds.nx}, channels={args.channels}")
    print(f"Time input units: {args.time_input_units}; residual dt units: {args.dt_units}")
    print(f"Loss weighting: {args.loss_weighting}")

    all_spans = sorted(set(args.fno_spans + args.fno_flow_spans))
    all_pair_splits = split_temporal_pairs(
        ds.t_norm,
        all_spans,
        args.temporal_bins,
        args.temporal_train_frac,
        args.temporal_val_frac,
        args.seed,
    )
    fno_pair_splits = {
        split: rows[np.isin(rows[:, 1], args.fno_spans)] for split, rows in all_pair_splits.items()
    }
    fno_flow_pair_splits = {
        split: rows[np.isin(rows[:, 1], args.fno_flow_spans)] for split, rows in all_pair_splits.items()
    }
    norm_rng = np.random.default_rng(args.seed + 111)
    mean, std = estimate_normalization(ds, norm_rng, fno_flow_pair_splits["train"], args.normalization_samples)
    print("Normalization mean:", dict(zip(args.channels, mean.tolist())))
    print("Normalization std:", dict(zip(args.channels, std.tolist())))

    persistence_rollout = evaluate_persistence_rollout(
        ds,
        ds.test_cases if ds.test_cases.size else ds.train_cases,
        args,
        mean,
        std,
        loss_weights,
        args.rollout_horizon,
    )
    results = {
        "persistence_rollout": to_jsonable(persistence_rollout),
        "persistence_fno_spans": to_jsonable(
            ModelResult(
                train=evaluate_persistence(ds, ds.train_cases, fno_pair_splits["train"], args, mean, std, loss_weights),
                validation=evaluate_persistence(
                    ds,
                    ds.validation_cases if ds.validation_cases.size else ds.train_cases,
                    fno_pair_splits["validation"],
                    args,
                    mean,
                    std,
                    loss_weights,
                ),
                heldout_time=evaluate_persistence(ds, ds.train_cases, fno_pair_splits["test"], args, mean, std, loss_weights),
                heldout_parameter=evaluate_persistence(ds, ds.test_cases, fno_pair_splits["train"], args, mean, std, loss_weights),
                heldout_parameter_time=evaluate_persistence(
                    ds, ds.test_cases, fno_pair_splits["test"], args, mean, std, loss_weights
                ),
                rollout=persistence_rollout,
                speed_ms_per_batch=None,
            )
        ),
        "persistence_fno_flow_spans": to_jsonable(
            ModelResult(
                train=evaluate_persistence(ds, ds.train_cases, fno_flow_pair_splits["train"], args, mean, std, loss_weights),
                validation=evaluate_persistence(
                    ds,
                    ds.validation_cases if ds.validation_cases.size else ds.train_cases,
                    fno_flow_pair_splits["validation"],
                    args,
                    mean,
                    std,
                    loss_weights,
                ),
                heldout_time=evaluate_persistence(
                    ds, ds.train_cases, fno_flow_pair_splits["test"], args, mean, std, loss_weights
                ),
                heldout_parameter=evaluate_persistence(
                    ds, ds.test_cases, fno_flow_pair_splits["train"], args, mean, std, loss_weights
                ),
                heldout_parameter_time=evaluate_persistence(
                    ds, ds.test_cases, fno_flow_pair_splits["test"], args, mean, std, loss_weights
                ),
                rollout=persistence_rollout,
                speed_ms_per_batch=None,
            )
        ),
    }
    history_by_model = {}
    if "linear_extrapolation" in args.models:
        results["linear_extrapolation"] = to_jsonable(
            run_linear_extrapolation(ds, fno_pair_splits, args, mean, std, loss_weights)
        )
    single_step_trainable = [
        ("plain_fno", 30),
        ("fno_reflect2d", 32),
        ("unet", 40),
        ("periodic_unet", 45),
        ("convlstm", 50),
        ("fno", 10),
        ("fno_radial", 12),
    ]
    for model_name, seed_offset in single_step_trainable:
        if model_name not in args.models:
            continue
        result, history = run_model(
            model_name,
            ds,
            coords,
            ds.train_cases,
            ds.validation_cases,
            ds.test_cases,
            fno_pair_splits,
            args.fno_spans,
            args,
            mean,
            std,
            loss_weights,
            seed_offset,
        )
        results[model_name] = to_jsonable(result)
        history_by_model[model_name] = history
    if "fno_flow" in args.models:
        result, history = run_model(
            "fno_flow",
            ds,
            coords,
            ds.train_cases,
            ds.validation_cases,
            ds.test_cases,
            fno_flow_pair_splits,
            args.fno_flow_spans,
            args,
            mean,
            std,
            loss_weights,
            20,
        )
        results["fno_flow"] = to_jsonable(result)
        history_by_model["fno_flow"] = history

    summary = {
        "setup": {
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "channels": args.channels,
            "models": args.models,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "radial_kernels": args.radial_kernels,
            "radial_dilations": args.radial_dilations,
            "radial_padding": args.radial_padding,
            "radial_global_mixing": args.radial_global_mixing,
            "radial_global_weight": args.radial_global_weight,
            "unet_levels": args.unet_levels,
            "convlstm_steps": args.convlstm_steps,
            "model_family": {
                "geometry_aware_fno": "theta Fourier spectral conv + multiscale nonperiodic radial local conv",
                "fno_radial": "geometry-aware FNO with theta Fourier, radial local conv, and reflect-FFT radial global mixing",
                "plain_fno": "rectangular 2D spectral conv with positive and negative radial low modes",
                "fno_reflect2d": "2D spectral conv on an even radial reflection extension and periodic theta",
                "unet": "convolutional encoder-decoder residual time stepper",
                "periodic_unet": "U-Net with theta circular padding and configured radial padding",
                "convlstm": "two-frame convolutional LSTM residual time stepper",
                "linear_extrapolation": "finite-difference extrapolation from the previous two frames",
            },
            "time_input_units": args.time_input_units,
            "dt_units": args.dt_units,
            "loss_weighting": args.loss_weighting,
            "rel_l2_floor": args.rel_l2_floor,
            "physics_metrics": {
                "enabled": args.physics_metrics,
                "planet_radius": args.planet_radius,
                "gap_r_range": [args.gap_r_min, args.gap_r_max],
                "ring_r_range": [args.ring_r_min, args.ring_r_max],
                "spiral_r_range": [args.spiral_r_min, args.spiral_r_max],
                "sigma_ref": args.sigma_ref,
                "sigma_ref_slope": args.sigma_ref_slope,
                "sigma_ref_profile": "sigma_ref * (r / planet_radius) ** (-sigma_ref_slope)",
                "gap_detection_fraction": args.gap_detection_fraction,
                "ring_detection_factor": args.ring_detection_factor,
                "diagnostic_smoothing": args.diagnostic_smoothing,
                "spiral_modes": args.spiral_modes,
            },
            "fno_spans": args.fno_spans,
            "fno_flow_spans": args.fno_flow_spans,
            "rollout_horizon": args.rollout_horizon,
            "rollout_train_weight": args.rollout_train_weight,
            "rollout_train_horizon": args.rollout_train_horizon,
            "save_checkpoints": args.save_checkpoints,
            "temporal_bins": args.temporal_bins,
            "normalization_mean": dict(zip(args.channels, mean.tolist())),
            "normalization_std": dict(zip(args.channels, std.tolist())),
            "case_counts": {
                "train": int(ds.train_cases.size),
                "validation": int(ds.validation_cases.size),
                "test": int(ds.test_cases.size),
            },
            "temporal_pair_splits_fno": summarize_splits(fno_pair_splits),
            "temporal_pair_splits_fno_flow": summarize_splits(fno_flow_pair_splits),
            "dataset_meta": ds.meta,
        },
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "loss_history.json").write_text(json.dumps(to_jsonable(history_by_model), indent=2))
    if args.save_loss_plots:
        save_loss_plot(history_by_model, args.output_dir)
    print(f"Saved benchmark outputs to {args.output_dir}")

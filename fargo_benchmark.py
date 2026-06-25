"""Top-level orchestration for the main FARGO operator benchmark."""

from __future__ import annotations

import json

import numpy as np

from fargo_data import (
    FargoMemmapDataset,
    coordinate_grid,
    estimate_normalization,
    spatial_loss_weights,
    split_temporal_pairs,
    validate_temporal_pair_splits,
)
from fargo_experiment import evaluate_operator, evaluate_persistence_suite, methods_from_args, select_pair_splits
from fargo_outputs import (
    save_loss_plot,
    save_model_checkpoint,
    summarize_splits,
    to_jsonable,
)
from fargo_model import FargoFNOConfig, make_fargo_fno
from fargo_training import TrainingConfig, train_operator_model


def make_operator_model(args, coords, output_channels: int):
    return make_fargo_fno(coords, FargoFNOConfig.from_args(args, output_channels))


def run_model(
    method,
    ds,
    coords,
    pair_splits,
    args,
    mean,
    std,
    loss_weights,
):
    model = make_operator_model(args, coords, ds.n_channels)
    train_config = TrainingConfig.from_args(args)
    training_result = train_operator_model(
        method.name,
        model,
        ds,
        ds.train_cases,
        ds.validation_cases if ds.validation_cases.size else ds.train_cases,
        pair_splits["train"],
        pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
        train_config,
        mean,
        std,
        loss_weights,
        method.seed_offset,
        method.consistency_weight,
    )
    params = training_result.params
    history = training_result.history
    if args.save_checkpoints:
        checkpoint_path = save_model_checkpoint(args.output_dir, method.name, params, ds, args, mean, std, pair_splits)
        print(f"Saved {method.name} checkpoint to {checkpoint_path}")
    result = evaluate_operator(model, params, ds, pair_splits, args, mean, std, loss_weights, method)
    result.trained_steps = training_result.trained_steps
    result.best_step = training_result.best_step
    result.best_validation_rmse = training_result.best_validation_rmse
    return result, history


def run_benchmark(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = FargoMemmapDataset(args.dataset, args.channels, args.time_input_units, args.dt_units)
    invalid_horizons = [horizon for horizon in args.rollout_horizons if horizon >= ds.n_frames]
    if args.rollout_horizon >= ds.n_frames or args.rollout_train_horizon >= ds.n_frames or invalid_horizons:
        raise ValueError(
            f"rollout horizons must be smaller than dataset frame count ({ds.n_frames}); "
            f"got rollout_horizon={args.rollout_horizon}, "
            f"rollout_train_horizon={args.rollout_train_horizon}, rollout_horizons={args.rollout_horizons}"
        )
    coords = coordinate_grid(ds.r, ds.theta)
    loss_weights = spatial_loss_weights(ds.r, args.loss_weighting)
    print(f"Loaded {args.dataset}: cases={ds.n_cases}, frames={ds.n_frames}, grid={ds.ny}x{ds.nx}, channels={args.channels}")
    print(f"Time input units: {args.time_input_units}; residual dt units: {args.dt_units}")
    print(f"Loss weighting: {args.loss_weighting}")

    methods = methods_from_args(args)
    if not methods:
        raise ValueError("At least one trainable model must be selected")
    all_spans = sorted({span for method in methods for span in method.spans})
    invalid_spans = [span for span in all_spans if span >= ds.n_frames]
    if invalid_spans:
        raise ValueError(f"Training spans must be smaller than frame count ({ds.n_frames}); got {invalid_spans}")
    all_pair_splits = split_temporal_pairs(
        ds.t_norm,
        all_spans,
        args.temporal_bins,
        args.temporal_train_frac,
        args.temporal_val_frac,
        args.seed,
    )
    pair_splits_by_method = {method.name: select_pair_splits(all_pair_splits, method.spans) for method in methods}
    for method in methods:
        validate_temporal_pair_splits(pair_splits_by_method[method.name], method.name)
        print(
            f"Method {method.name}: spans={list(method.spans)} "
            f"consistency_weight={method.consistency_weight}"
        )
    primary_method = next((method for method in methods if method.name == "fno"), methods[0])
    primary_pair_splits = pair_splits_by_method[primary_method.name]
    norm_rng = np.random.default_rng(args.seed + 111)
    mean, std = estimate_normalization(ds, norm_rng, primary_pair_splits["train"], args.normalization_samples)
    print("Normalization mean:", dict(zip(args.channels, mean.tolist())))
    print("Normalization std:", dict(zip(args.channels, std.tolist())))

    results = {}
    for method in methods:
        pair_splits = pair_splits_by_method[method.name]
        results[f"persistence_{method.name}"] = to_jsonable(
            evaluate_persistence_suite(ds, pair_splits, args, mean, std, loss_weights)
        )
    history_by_model = {}
    for method in methods:
        result, history = run_model(
            method,
            ds,
            coords,
            pair_splits_by_method[method.name],
            args,
            mean,
            std,
            loss_weights,
        )
        results[method.name] = to_jsonable(result)
        history_by_model[method.name] = history

    summary = {
        "setup": {
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "channels": args.channels,
            "models": args.models,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "width": args.width,
            "depth": args.depth,
            "modes_r": args.modes_r,
            "modes_theta": args.modes_theta,
            "radial_kernels": args.radial_kernels,
            "radial_dilations": args.radial_dilations,
            "radial_padding": args.radial_padding,
            "operator_geometry": "theta Fourier spectral conv + multiscale nonperiodic radial local conv",
            "time_input_units": args.time_input_units,
            "dt_units": args.dt_units,
            "loss_weighting": args.loss_weighting,
            "rel_l2_floor": args.rel_l2_floor,
            "fno_spans": args.fno_spans,
            "fno_flow_spans": args.fno_flow_spans,
            "rollout_horizon": args.rollout_horizon,
            "rollout_horizons": args.rollout_horizons,
            "rollout_train_weight": args.rollout_train_weight,
            "rollout_train_horizon": args.rollout_train_horizon,
            "save_checkpoints": args.save_checkpoints,
            "temporal_bins": args.temporal_bins,
            "normalization_mean": dict(zip(args.channels, mean.tolist())),
            "normalization_std": dict(zip(args.channels, std.tolist())),
            "normalization_reference_model": primary_method.name,
            "case_counts": {
                "train": int(ds.train_cases.size),
                "validation": int(ds.validation_cases.size),
                "test": int(ds.test_cases.size),
            },
            "methods": {
                method.name: {
                    "spans": list(method.spans),
                    "consistency_weight": method.consistency_weight,
                    "uses_semigroup_loss": method.uses_semigroup_loss,
                    "temporal_pair_splits": summarize_splits(pair_splits_by_method[method.name]),
                }
                for method in methods
            },
            "dataset_meta": ds.meta,
        },
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "loss_history.json").write_text(json.dumps(to_jsonable(history_by_model), indent=2))
    if args.save_loss_plots:
        save_loss_plot(history_by_model, args.output_dir)
    print(f"Saved benchmark outputs to {args.output_dir}")

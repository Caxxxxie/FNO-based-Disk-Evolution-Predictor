"""Top-level orchestration for the main FARGO operator benchmark."""

from __future__ import annotations

import json

import numpy as np

from fargo_data import (
    FargoMemmapDataset,
    coordinate_grid,
    estimate_normalization,
    make_batch,
    spatial_loss_weights,
    split_temporal_pairs,
)
from fargo_metrics import (
    evaluate_model_by_span,
    evaluate_model,
    evaluate_persistence,
    evaluate_persistence_by_span,
    evaluate_persistence_rollout,
    evaluate_persistence_rollouts,
    evaluate_rollout,
    evaluate_rollouts,
    evaluate_semigroup,
)
from fargo_outputs import (
    ModelResult,
    save_loss_plot,
    save_model_checkpoint,
    speed_ms_per_batch,
    summarize_splits,
    to_jsonable,
)
from fargo_model import FargoFNOConfig, make_fargo_fno
from fargo_training import TrainingConfig, train_operator_model


def make_operator_model(args, coords, output_channels: int):
    return make_fargo_fno(coords, FargoFNOConfig.from_args(args, output_channels))


def run_model(
    name,
    ds,
    coords,
    train_cases,
    val_cases,
    test_cases,
    pair_splits,
    args,
    mean,
    std,
    loss_weights,
    seed_offset,
):
    model = make_operator_model(args, coords, ds.n_channels)
    train_config = TrainingConfig.from_args(args)
    use_consistency = name == "fno_flow" and args.consistency_weight > 0.0
    training_result = train_operator_model(
        name,
        model,
        ds,
        train_cases,
        val_cases if val_cases.size else train_cases,
        pair_splits["train"],
        pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
        train_config,
        mean,
        std,
        loss_weights,
        seed_offset,
        use_consistency,
    )
    params = training_result.params
    history = training_result.history
    if args.save_checkpoints:
        checkpoint_path = save_model_checkpoint(args.output_dir, name, params, ds, args, mean, std, pair_splits)
        print(f"Saved {name} checkpoint to {checkpoint_path}")
    speed_batch = make_batch(
        ds,
        np.random.default_rng(args.seed + 333),
        test_cases if test_cases.size else train_cases,
        pair_splits["test"],
        args.batch_size,
        mean,
        std,
    )
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
    rollout_by_horizon = evaluate_rollouts(
        model,
        params,
        ds,
        test_cases if test_cases.size else train_cases,
        args,
        mean,
        std,
        loss_weights,
        args.rollout_horizons,
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
        heldout_parameter_time_by_span=evaluate_model_by_span(
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
        rollout_by_horizon=rollout_by_horizon,
        speed_ms_per_batch=speed_ms_per_batch(model, params, speed_batch, args.speed_repeats),
        trained_steps=training_result.trained_steps,
        best_step=training_result.best_step,
        best_validation_rmse=training_result.best_validation_rmse,
        semigroup_rmse=sg,
        semigroup_rmse_by_channel=sg_ch,
    )
    return result, history


def run_benchmark(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = FargoMemmapDataset(args.dataset, args.channels, args.time_input_units, args.dt_units)
    invalid_horizons = [horizon for horizon in args.rollout_horizons if horizon >= ds.n_frames]
    if args.rollout_horizon >= ds.n_frames or invalid_horizons:
        raise ValueError(
            f"rollout horizons must be smaller than dataset frame count ({ds.n_frames}); "
            f"got rollout_horizon={args.rollout_horizon}, rollout_horizons={args.rollout_horizons}"
        )
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
    persistence_rollout_by_horizon = evaluate_persistence_rollouts(
        ds,
        ds.test_cases if ds.test_cases.size else ds.train_cases,
        args,
        mean,
        std,
        loss_weights,
        args.rollout_horizons,
    )
    results = {
        "persistence_rollout": to_jsonable(persistence_rollout),
        "persistence_rollout_by_horizon": to_jsonable(persistence_rollout_by_horizon),
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
                heldout_parameter_time_by_span=evaluate_persistence_by_span(
                    ds, ds.test_cases, fno_pair_splits["test"], args, mean, std, loss_weights
                ),
                rollout_by_horizon=persistence_rollout_by_horizon,
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
                heldout_parameter_time_by_span=evaluate_persistence_by_span(
                    ds, ds.test_cases, fno_flow_pair_splits["test"], args, mean, std, loss_weights
                ),
                rollout_by_horizon=persistence_rollout_by_horizon,
                speed_ms_per_batch=None,
            )
        ),
    }
    history_by_model = {}
    if "fno" in args.models:
        result, history = run_model(
            "fno",
            ds,
            coords,
            ds.train_cases,
            ds.validation_cases,
            ds.test_cases,
            fno_pair_splits,
            args,
            mean,
            std,
            loss_weights,
            10,
        )
        results["fno"] = to_jsonable(result)
        history_by_model["fno"] = history
    if "fno_flow" in args.models:
        result, history = run_model(
            "fno_flow",
            ds,
            coords,
            ds.train_cases,
            ds.validation_cases,
            ds.test_cases,
            fno_flow_pair_splits,
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

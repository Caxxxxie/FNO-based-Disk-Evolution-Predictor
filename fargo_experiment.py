"""Experiment specifications for FARGO transient operator learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fargo_data import FargoMemmapDataset, make_batch
from fargo_metrics import (
    evaluate_model,
    evaluate_model_by_span,
    evaluate_persistence,
    evaluate_persistence_by_span,
    evaluate_persistence_rollout,
    evaluate_persistence_rollouts,
    evaluate_rollout,
    evaluate_rollouts,
    evaluate_semigroup,
)
from fargo_outputs import ModelResult, speed_ms_per_batch


@dataclass(frozen=True)
class OperatorMethod:
    """Training recipe for one time-conditioned operator variant."""

    name: str
    spans: tuple[int, ...]
    consistency_weight: float = 0.0
    seed_offset: int = 0

    @property
    def uses_semigroup_loss(self) -> bool:
        return self.consistency_weight > 0.0


def methods_from_args(args) -> list[OperatorMethod]:
    """Translate CLI model names into explicit training recipes."""
    specs = []
    if "fno" in args.models:
        specs.append(OperatorMethod("fno", tuple(args.fno_spans), consistency_weight=0.0, seed_offset=10))
    if "fno_flow" in args.models:
        specs.append(
            OperatorMethod(
                "fno_flow",
                tuple(args.fno_flow_spans),
                consistency_weight=args.consistency_weight,
                seed_offset=20,
            )
        )
    return specs


def select_pair_splits(pair_splits: dict[str, np.ndarray], spans: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Filter split rows to the spans used by one method."""
    return {split: rows[np.isin(rows[:, 1], spans)] for split, rows in pair_splits.items()}


def evaluate_operator(
    model,
    params,
    ds: FargoMemmapDataset,
    pair_splits: dict[str, np.ndarray],
    args,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
    method: OperatorMethod | None = None,
) -> ModelResult:
    """Run the standard train/validation/held-out/rollout evaluations."""
    train_cases = ds.train_cases
    validation_cases = ds.validation_cases if ds.validation_cases.size else ds.train_cases
    test_cases = ds.test_cases if ds.test_cases.size else validation_cases
    speed_batch = make_batch(
        ds,
        np.random.default_rng(args.seed + 333),
        test_cases,
        pair_splits["test"],
        args.batch_size,
        mean,
        std,
    )
    semigroup_rmse = semigroup_by_channel = None
    if method is not None and method.uses_semigroup_loss:
        semigroup_rmse, semigroup_by_channel = evaluate_semigroup(
            model, params, ds, test_cases, args, mean, std, loss_weights
        )
    return ModelResult(
        train=evaluate_model(model, params, ds, train_cases, pair_splits["train"], args, mean, std, loss_weights),
        validation=evaluate_model(
            model,
            params,
            ds,
            validation_cases,
            pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_time=evaluate_model(model, params, ds, train_cases, pair_splits["test"], args, mean, std, loss_weights),
        heldout_parameter=evaluate_model(model, params, ds, test_cases, pair_splits["train"], args, mean, std, loss_weights),
        heldout_parameter_time=evaluate_model(
            model, params, ds, test_cases, pair_splits["test"], args, mean, std, loss_weights
        ),
        heldout_parameter_time_by_span=evaluate_model_by_span(
            model, params, ds, test_cases, pair_splits["test"], args, mean, std, loss_weights
        ),
        rollout=evaluate_rollout(model, params, ds, test_cases, args, mean, std, loss_weights, args.rollout_horizon),
        rollout_by_horizon=evaluate_rollouts(
            model, params, ds, test_cases, args, mean, std, loss_weights, args.rollout_horizons
        ),
        speed_ms_per_batch=speed_ms_per_batch(model, params, speed_batch, args.speed_repeats),
        semigroup_rmse=semigroup_rmse,
        semigroup_rmse_by_channel=semigroup_by_channel,
    )


def evaluate_persistence_suite(
    ds: FargoMemmapDataset,
    pair_splits: dict[str, np.ndarray],
    args,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
) -> ModelResult:
    """Evaluate the no-change baseline on the same splits as one method."""
    train_cases = ds.train_cases
    validation_cases = ds.validation_cases if ds.validation_cases.size else ds.train_cases
    test_cases = ds.test_cases if ds.test_cases.size else validation_cases
    return ModelResult(
        train=evaluate_persistence(ds, train_cases, pair_splits["train"], args, mean, std, loss_weights),
        validation=evaluate_persistence(
            ds,
            validation_cases,
            pair_splits["validation"] if pair_splits["validation"].shape[0] else pair_splits["train"],
            args,
            mean,
            std,
            loss_weights,
        ),
        heldout_time=evaluate_persistence(ds, train_cases, pair_splits["test"], args, mean, std, loss_weights),
        heldout_parameter=evaluate_persistence(ds, test_cases, pair_splits["train"], args, mean, std, loss_weights),
        heldout_parameter_time=evaluate_persistence(ds, test_cases, pair_splits["test"], args, mean, std, loss_weights),
        heldout_parameter_time_by_span=evaluate_persistence_by_span(
            ds, test_cases, pair_splits["test"], args, mean, std, loss_weights
        ),
        rollout=evaluate_persistence_rollout(ds, test_cases, args, mean, std, loss_weights, args.rollout_horizon),
        rollout_by_horizon=evaluate_persistence_rollouts(
            ds, test_cases, args, mean, std, loss_weights, args.rollout_horizons
        ),
        speed_ms_per_batch=None,
    )

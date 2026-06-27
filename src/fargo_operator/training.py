"""Training loop for FARGO operator models."""

from __future__ import annotations

import argparse
import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import FargoMemmapDataset, make_batch, make_consistency_batch, make_rollout_batch
from .evaluation import evaluate_model
from .models import batch_mse, set_time_span


def train_model(
    name: str,
    model,
    ds: FargoMemmapDataset,
    train_cases: np.ndarray,
    val_cases: np.ndarray,
    train_pairs: np.ndarray,
    val_pairs: np.ndarray,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
    seed_offset: int,
    use_consistency: bool,
) -> tuple[hk.Params, list[dict]]:
    rng = np.random.default_rng(args.seed + seed_offset)
    key = jax.random.PRNGKey(args.seed + seed_offset)
    init_batch = make_batch(ds, rng, train_cases, train_pairs, min(args.batch_size, 2), mean, std)
    params = model.init(key, init_batch)
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.lr))
    opt_state = opt.init(params)
    span_a, span_b = args.consistency_spans
    loss_weights_jax = jnp.asarray(loss_weights)

    @jax.jit
    def supervised_loss(params, batch):
        pred = model.apply(params, batch)
        return batch_mse(pred, batch["y"], loss_weights_jax)

    @jax.jit
    def total_loss(params, batch, consistency_batch, rollout_batch):
        pred = model.apply(params, batch)
        sup = batch_mse(pred, batch["y"], loss_weights_jax)
        consistency = jnp.asarray(0.0, dtype=sup.dtype)
        if use_consistency:
            direct = model.apply(params, set_time_span(consistency_batch, consistency_batch["t_ab"], consistency_batch["dt_ab"]))
            first = model.apply(params, consistency_batch)
            second_batch = dict(consistency_batch)
            second_batch["x"] = first
            second_batch["t"] = consistency_batch["t_b"]
            second_batch["dt"] = consistency_batch["dt_b"]
            second = model.apply(params, second_batch)
            consistency = batch_mse(direct, second, loss_weights_jax) + batch_mse(
                direct, consistency_batch["y"], loss_weights_jax
            )
        rollout = jnp.asarray(0.0, dtype=sup.dtype)
        if args.rollout_train_weight > 0.0:
            current = rollout_batch["x0"]
            previous = rollout_batch["x_prev"]
            losses = []
            for i in range(args.rollout_train_horizon):
                step_batch = {
                    "x_prev": previous,
                    "x": current,
                    "mu": rollout_batch["mu"],
                    "t": rollout_batch["t"][:, i, :],
                    "dt": rollout_batch["dt"][:, i, :],
                }
                previous = current
                current = model.apply(params, step_batch)
                losses.append(batch_mse(current, rollout_batch["targets"][:, i, ...], loss_weights_jax))
            rollout = sum(losses) / len(losses)
        return sup + args.consistency_weight * consistency + args.rollout_train_weight * rollout, (sup, consistency, rollout)

    @jax.jit
    def train_step(params, opt_state, batch, consistency_batch, rollout_batch):
        (loss, (sup, consistency, rollout)), grads = jax.value_and_grad(total_loss, has_aux=True)(
            params, batch, consistency_batch, rollout_batch
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, sup, consistency, rollout

    history = []
    best_params = params
    best_val = math.inf
    best_step = 0
    stale = 0
    empty_consistency = make_consistency_batch(ds, rng, train_cases, span_a, span_b, min(args.batch_size, 2), mean, std)
    empty_rollout = make_rollout_batch(
        ds,
        rng,
        train_cases,
        args.rollout_train_horizon,
        min(args.batch_size, 2),
        mean,
        std,
    )
    for step in range(1, args.steps + 1):
        batch = make_batch(ds, rng, train_cases, train_pairs, args.batch_size, mean, std)
        if use_consistency:
            consistency_batch = make_consistency_batch(ds, rng, train_cases, span_a, span_b, args.batch_size, mean, std)
        else:
            consistency_batch = empty_consistency
        if args.rollout_train_weight > 0.0:
            rollout_batch = make_rollout_batch(
                ds,
                rng,
                train_cases,
                args.rollout_train_horizon,
                args.batch_size,
                mean,
                std,
            )
        else:
            rollout_batch = empty_rollout
        params, opt_state, loss, sup, consistency, rollout = train_step(params, opt_state, batch, consistency_batch, rollout_batch)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_eval = evaluate_model(
                model,
                params,
                ds,
                train_cases,
                train_pairs,
                args,
                mean,
                std,
                loss_weights,
                max_batches=4,
                include_physics=False,
            )
            val_eval = evaluate_model(
                model,
                params,
                ds,
                val_cases,
                val_pairs,
                args,
                mean,
                std,
                loss_weights,
                max_batches=max(4, args.eval_batches // 2),
                include_physics=False,
            )
            val_rmse = val_eval.rmse
            history.append(
                {
                    "step": step,
                    "batch_rmse": float(jnp.sqrt(loss)),
                    "supervised_rmse": float(jnp.sqrt(sup)),
                    "consistency_rmse": float(jnp.sqrt(consistency)) if use_consistency else 0.0,
                    "rollout_rmse": float(jnp.sqrt(rollout)) if args.rollout_train_weight > 0.0 else 0.0,
                    "train_rmse": train_eval.rmse,
                    "validation_rmse": val_eval.rmse,
                }
            )
            print(f"{name} step {step}: train={train_eval.rmse:.5f} val={val_rmse:.5f}")
            if val_rmse < best_val - args.early_stop_min_delta:
                best_val = val_rmse
                best_params = params
                best_step = step
                stale = 0
            else:
                stale += 1
            if args.early_stop_patience and stale >= args.early_stop_patience:
                print(f"{name} early stopped at step {step}")
                break
    history.append({"best_step": best_step, "best_validation_rmse": best_val})
    return best_params, history

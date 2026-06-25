"""Training utilities for the main FARGO operator benchmark."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fargo_data import FargoMemmapDataset, make_batch, make_consistency_batch, set_time_span
from fargo_metrics import evaluate_model


def batch_mse(pred, target, weights):
    return jnp.mean(((pred - target) ** 2) * weights)


def train_operator_model(
    name: str,
    model,
    ds: FargoMemmapDataset,
    train_cases: np.ndarray,
    val_cases: np.ndarray,
    train_pairs: np.ndarray,
    val_pairs: np.ndarray,
    args,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
    seed_offset: int,
    use_consistency: bool,
) -> tuple[object, list[dict]]:
    rng = np.random.default_rng(args.seed + seed_offset)
    key = jax.random.PRNGKey(args.seed + seed_offset)
    init_batch = make_batch(ds, rng, train_cases, train_pairs, min(args.batch_size, 2), mean, std)
    params = model.init(key, init_batch)
    opt = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adam(args.lr))
    opt_state = opt.init(params)
    span_a, span_b = args.consistency_spans
    loss_weights_jax = jnp.asarray(loss_weights)

    @jax.jit
    def total_loss(params, batch, consistency_batch):
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
        return sup + args.consistency_weight * consistency, (sup, consistency)

    @jax.jit
    def train_step(params, opt_state, batch, consistency_batch):
        (loss, (sup, consistency)), grads = jax.value_and_grad(total_loss, has_aux=True)(params, batch, consistency_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, sup, consistency

    history = []
    best_params = params
    best_val = math.inf
    best_step = 0
    stale = 0
    empty_consistency = make_consistency_batch(ds, rng, train_cases, span_a, span_b, min(args.batch_size, 2), mean, std)
    for step in range(1, args.steps + 1):
        batch = make_batch(ds, rng, train_cases, train_pairs, args.batch_size, mean, std)
        if use_consistency:
            consistency_batch = make_consistency_batch(ds, rng, train_cases, span_a, span_b, args.batch_size, mean, std)
        else:
            consistency_batch = empty_consistency
        params, opt_state, loss, sup, consistency = train_step(params, opt_state, batch, consistency_batch)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_eval = evaluate_model(
                model, params, ds, train_cases, train_pairs, args, mean, std, loss_weights, max_batches=4
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
            )
            val_rmse = val_eval.rmse
            history.append(
                {
                    "step": step,
                    "batch_rmse": float(jnp.sqrt(loss)),
                    "supervised_rmse": float(jnp.sqrt(sup)),
                    "consistency_rmse": float(jnp.sqrt(consistency)) if use_consistency else 0.0,
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

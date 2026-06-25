"""Training utilities for the main FARGO operator benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fargo_data import FargoMemmapDataset, make_batch, make_consistency_batch, set_time_span
from fargo_metrics import evaluate_model


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    lr: float
    grad_clip_norm: float
    warmup_steps: int
    min_lr_ratio: float
    seed: int
    eval_every: int
    eval_batches: int
    rel_l2_floor: float
    consistency_weight: float
    consistency_spans: tuple[int, int]
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1.0e-4

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        return cls(
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip_norm=args.grad_clip_norm,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            seed=args.seed,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            rel_l2_floor=args.rel_l2_floor,
            consistency_weight=args.consistency_weight,
            consistency_spans=tuple(args.consistency_spans),
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
        )

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if self.rel_l2_floor <= 0.0:
            raise ValueError("rel_l2_floor must be positive")
        if self.consistency_weight < 0.0:
            raise ValueError("consistency_weight must be nonnegative")
        if len(self.consistency_spans) != 2 or any(span <= 0 for span in self.consistency_spans):
            raise ValueError("consistency_spans must contain two positive spans")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be nonnegative")
        if self.early_stop_min_delta < 0.0:
            raise ValueError("early_stop_min_delta must be nonnegative")


@dataclass
class TrainingResult:
    params: object
    history: list[dict]
    best_step: int
    best_validation_rmse: float
    trained_steps: int


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
    config: TrainingConfig,
    mean: np.ndarray,
    std: np.ndarray,
    loss_weights: np.ndarray,
    seed_offset: int,
    consistency_weight: float,
) -> TrainingResult:
    config.validate()
    rng = np.random.default_rng(config.seed + seed_offset)
    key = jax.random.PRNGKey(config.seed + seed_offset)
    init_batch = make_batch(ds, rng, train_cases, train_pairs, min(config.batch_size, 2), mean, std)
    params = model.init(key, init_batch)
    warmup_steps = min(config.warmup_steps, max(0, config.steps - 1))
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.lr,
            warmup_steps=warmup_steps,
            decay_steps=config.steps,
            end_value=config.lr * config.min_lr_ratio,
        )
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=config.lr,
            decay_steps=config.steps,
            alpha=config.min_lr_ratio,
        )
    opt = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), optax.adam(schedule))
    opt_state = opt.init(params)
    span_a, span_b = config.consistency_spans
    loss_weights_jax = jnp.asarray(loss_weights)

    @jax.jit
    def total_loss(params, batch, consistency_batch):
        pred = model.apply(params, batch)
        sup = batch_mse(pred, batch["y"], loss_weights_jax)
        consistency = jnp.asarray(0.0, dtype=sup.dtype)
        if consistency_weight > 0.0:
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
        return sup + consistency_weight * consistency, (sup, consistency)

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
    empty_consistency = make_consistency_batch(ds, rng, train_cases, span_a, span_b, min(config.batch_size, 2), mean, std)
    for step in range(1, config.steps + 1):
        batch = make_batch(ds, rng, train_cases, train_pairs, config.batch_size, mean, std)
        if consistency_weight > 0.0:
            consistency_batch = make_consistency_batch(ds, rng, train_cases, span_a, span_b, config.batch_size, mean, std)
        else:
            consistency_batch = empty_consistency
        params, opt_state, loss, sup, consistency = train_step(params, opt_state, batch, consistency_batch)
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            train_eval = evaluate_model(
                model, params, ds, train_cases, train_pairs, config, mean, std, loss_weights, max_batches=4
            )
            val_eval = evaluate_model(
                model,
                params,
                ds,
                val_cases,
                val_pairs,
                config,
                mean,
                std,
                loss_weights,
                max_batches=max(4, config.eval_batches // 2),
            )
            val_rmse = val_eval.rmse
            history.append(
                {
                    "step": step,
                    "batch_rmse": float(jnp.sqrt(loss)),
                    "supervised_rmse": float(jnp.sqrt(sup)),
                    "consistency_rmse": float(jnp.sqrt(consistency)) if consistency_weight > 0.0 else 0.0,
                    "train_rmse": train_eval.rmse,
                    "validation_rmse": val_eval.rmse,
                    "learning_rate": float(schedule(step)),
                }
            )
            print(f"{name} step {step}: train={train_eval.rmse:.5f} val={val_rmse:.5f}")
            if val_rmse < best_val - config.early_stop_min_delta:
                best_val = val_rmse
                best_params = params
                best_step = step
                stale = 0
            else:
                stale += 1
            if config.early_stop_patience and stale >= config.early_stop_patience:
                print(f"{name} early stopped at step {step}")
                break
    history.append({"best_step": best_step, "best_validation_rmse": best_val})
    trained_steps = max(row.get("step", 0) for row in history)
    return TrainingResult(
        params=best_params,
        history=history,
        best_step=best_step,
        best_validation_rmse=best_val,
        trained_steps=trained_steps,
    )

"""Small JAX/Haiku training helpers shared by experiment runners."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax


def train_regressor(model, train: dict[str, jnp.ndarray], *, steps: int, batch_size: int, lr: float, seed: int):
    key = jax.random.PRNGKey(seed)
    params = model.init(key, {k: v[:1] for k, v in train.items()})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch):
        pred = model.apply(params, batch)
        return jnp.mean((pred - batch["y"]) ** 2)

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = train["x"].shape[0]
    last_loss = 0.0
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, n)
        batch = {k: v[idx] for k, v in train.items()}
        params, opt_state, last_loss = train_step(params, opt_state, batch)

    return params, loss_fn, float(last_loss)


def rmse(model, params, batch: dict[str, jnp.ndarray]) -> float:
    pred = model.apply(params, batch)
    return float(jnp.sqrt(jnp.mean((pred - batch["y"]) ** 2)))


def prediction_speed_ms(model, params, batch: dict[str, jnp.ndarray], repeats: int) -> float:
    apply_fn = jax.jit(lambda p, b: model.apply(p, b))
    apply_fn(params, batch).block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        y = apply_fn(params, batch)
    y.block_until_ready()
    return (time.perf_counter() - start) * 1000.0 / repeats


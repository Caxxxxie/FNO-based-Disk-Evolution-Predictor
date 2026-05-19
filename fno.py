"""Small FNO-style grid operators used by the project experiments."""

from __future__ import annotations

import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def coordinate_channels(ny: int, nx: int) -> np.ndarray:
    """Return normalized polar-grid coordinate channels."""
    r = np.linspace(-1.0, 1.0, ny, dtype=np.float32)[:, None]
    theta = np.linspace(-np.pi, np.pi, nx, endpoint=False, dtype=np.float32)[None, :]
    return np.stack(
        [
            np.broadcast_to(r, (ny, nx)),
            np.broadcast_to(np.sin(theta), (ny, nx)),
            np.broadcast_to(np.cos(theta), (ny, nx)),
        ],
        axis=-1,
    ).astype(np.float32)


class SpectralConv2D(hk.Module):
    """2D spectral convolution over radial and azimuthal grid axes."""

    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        mr = min(self.modes_r, ny)
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real = hk.get_parameter(
            "weight_real",
            shape=(mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        imag = hk.get_parameter(
            "weight_imag",
            shape=(mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        weight = real + 1j * imag
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low = jnp.einsum("bhwi,hwio->bhwo", x_ft[:, :mr, :mt, :], weight)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low)
        return jnp.fft.irfft2(out_ft, s=(ny, nx), axes=(1, 2))


def grid_condition_inputs(batch: dict[str, jnp.ndarray], coords: np.ndarray) -> jnp.ndarray:
    """Build per-cell FNO inputs from grid state, coordinates, and parameters."""
    x = batch["x"]
    b, ny, nx, _ = x.shape
    coord = jnp.broadcast_to(jnp.asarray(coords)[None, :, :, :], (b, ny, nx, coords.shape[-1]))
    mu = jnp.broadcast_to(batch["mu"][:, None, None, :], (b, ny, nx, batch["mu"].shape[-1]))
    return jnp.concatenate([x, coord, mu], axis=-1)


def make_fno_regressor(
    coords: np.ndarray,
    width: int,
    modes_r: int,
    modes_theta: int,
    depth: int,
):
    """Create a grid-to-grid FNO regressor for steady fields."""

    def forward(batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
        h = hk.Linear(width)(grid_condition_inputs(batch, coords))
        for i in range(depth):
            spectral = SpectralConv2D(width, modes_r, modes_theta, name=f"spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        return hk.nets.MLP([width, 1], activation=jax.nn.gelu)(h)

    return hk.without_apply_rng(hk.transform(forward))


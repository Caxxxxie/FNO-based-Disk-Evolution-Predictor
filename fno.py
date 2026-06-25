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


def pad_radial(x, pad: int, mode: str):
    if pad == 0:
        return x
    pad_width = ((0, 0), (pad, pad), (0, 0), (0, 0))
    if mode == "zero":
        return jnp.pad(x, pad_width)
    if mode == "edge":
        return jnp.pad(x, pad_width, mode="edge")
    if mode == "reflect":
        return jnp.pad(x, pad_width, mode="reflect")
    raise ValueError(f"Unknown radial padding mode: {mode}")


class ThetaSpectralRadialConv(hk.Module):
    """Periodic theta Fourier mixing plus multiscale nonperiodic radial mixing."""

    def __init__(
        self,
        out_channels: int,
        modes_theta: int,
        radial_kernels: list[int],
        radial_dilations: list[int],
        radial_padding: str,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_theta = modes_theta
        self.radial_kernels = tuple(radial_kernels)
        self.radial_dilations = tuple(radial_dilations)
        self.radial_padding = radial_padding

    def __call__(self, x):
        _, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft(x, axis=2)
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real = hk.get_parameter(
            "theta_weight_real", (mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale)
        )
        imag = hk.get_parameter(
            "theta_weight_imag", (mt, in_channels, self.out_channels), init=hk.initializers.RandomNormal(scale)
        )
        weight = real + 1j * imag
        out_ft = jnp.zeros((x.shape[0], ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low_theta = jnp.einsum("byki,kio->byko", x_ft[:, :, :mt, :], weight)
        out_ft = out_ft.at[:, :, :mt, :].set(low_theta)
        theta_mixed = jnp.fft.irfft(out_ft, n=nx, axis=2)
        radial_terms = []
        for idx, (kernel, dilation) in enumerate(zip(self.radial_kernels, self.radial_dilations)):
            pad = dilation * (kernel - 1) // 2
            x_padded = pad_radial(x, pad, self.radial_padding)
            radial_terms.append(
                hk.Conv2D(
                    self.out_channels,
                    kernel_shape=(kernel, 1),
                    rate=(dilation, 1),
                    padding="VALID",
                    with_bias=False,
                    name=f"radial_local_{idx}",
                )(x_padded)
            )
        radial_mixed = sum(radial_terms) / math.sqrt(len(radial_terms))
        return theta_mixed + radial_mixed


def time_conditioned_grid_inputs(batch: dict[str, jnp.ndarray], coords: np.ndarray) -> jnp.ndarray:
    """Build grid inputs from state, disk coordinates, parameters, and time span."""
    x = batch["x"]
    b, ny, nx, _ = x.shape
    coord = jnp.broadcast_to(jnp.asarray(coords)[None, :, :, :], (b, ny, nx, 3))
    cond = jnp.concatenate([batch["mu"], batch["t"], batch["dt"]], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (b, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coord, cond], axis=-1)


def make_time_conditioned_fno(
    coords,
    width,
    modes_theta,
    depth,
    output_channels: int,
    residual: bool = True,
    radial_kernels: list[int] | None = None,
    radial_dilations: list[int] | None = None,
    radial_padding: str = "edge",
):
    """Create the main disk operator used by the large FARGO benchmark."""
    radial_kernels = [3, 5, 5] if radial_kernels is None else radial_kernels
    radial_dilations = [1, 2, 4] if radial_dilations is None else radial_dilations

    def forward(batch):
        h = hk.Linear(width)(time_conditioned_grid_inputs(batch, coords))
        for i in range(depth):
            spectral = ThetaSpectralRadialConv(
                width,
                modes_theta,
                radial_kernels,
                radial_dilations,
                radial_padding,
                name=f"theta_spectral_radial_{i}",
            )(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([width, output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))

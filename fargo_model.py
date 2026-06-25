"""Core FARGO disk-operator model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FargoFNOConfig:
    """Architecture for the time-conditioned disk FNO."""

    width: int = 48
    depth: int = 4
    modes_theta: int = 24
    output_channels: int = 3
    residual: bool = True
    radial_kernels: tuple[int, ...] = field(default_factory=lambda: (3, 5, 5))
    radial_dilations: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4))
    radial_padding: str = "edge"

    @classmethod
    def from_args(cls, args, output_channels: int) -> "FargoFNOConfig":
        return cls(
            width=args.width,
            depth=args.depth,
            modes_theta=args.modes_theta,
            output_channels=output_channels,
            radial_kernels=tuple(args.radial_kernels),
            radial_dilations=tuple(args.radial_dilations),
            radial_padding=args.radial_padding,
        )

    def validate(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.modes_theta <= 0:
            raise ValueError("modes_theta must be positive")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if len(self.radial_kernels) != len(self.radial_dilations):
            raise ValueError("radial_kernels and radial_dilations must have the same length")
        if not self.radial_kernels:
            raise ValueError("at least one radial kernel is required")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.radial_kernels):
            raise ValueError("all radial kernels must be positive odd integers")
        if any(dilation <= 0 for dilation in self.radial_dilations):
            raise ValueError("all radial dilations must be positive")
        if self.radial_padding not in {"zero", "edge", "reflect"}:
            raise ValueError("radial_padding must be one of: zero, edge, reflect")


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
        radial_kernels: tuple[int, ...],
        radial_dilations: tuple[int, ...],
        radial_padding: str,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_theta = modes_theta
        self.radial_kernels = radial_kernels
        self.radial_dilations = radial_dilations
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


def make_fargo_fno(coords: np.ndarray, config: FargoFNOConfig):
    """Create the main time-conditioned disk FNO operator."""
    config.validate()

    def forward(batch):
        h = hk.Linear(config.width)(time_conditioned_grid_inputs(batch, coords))
        for i in range(config.depth):
            spectral = ThetaSpectralRadialConv(
                config.width,
                config.modes_theta,
                config.radial_kernels,
                config.radial_dilations,
                config.radial_padding,
                name=f"theta_spectral_radial_{i}",
            )(h)
            pointwise = hk.Linear(config.width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([config.width, config.output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if config.residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_time_conditioned_fno(
    coords: np.ndarray,
    width: int,
    modes_theta: int,
    depth: int,
    output_channels: int,
    residual: bool = True,
    radial_kernels: list[int] | tuple[int, ...] | None = None,
    radial_dilations: list[int] | tuple[int, ...] | None = None,
    radial_padding: str = "edge",
):
    """Compatibility wrapper around :func:`make_fargo_fno`."""
    config = FargoFNOConfig(
        width=width,
        depth=depth,
        modes_theta=modes_theta,
        output_channels=output_channels,
        residual=residual,
        radial_kernels=tuple((3, 5, 5) if radial_kernels is None else radial_kernels),
        radial_dilations=tuple((1, 2, 4) if radial_dilations is None else radial_dilations),
        radial_padding=radial_padding,
    )
    return make_fargo_fno(coords, config)


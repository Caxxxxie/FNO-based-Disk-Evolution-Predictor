"""Model architectures for FARGO operator experiments."""

from __future__ import annotations

import argparse
import math

import haiku as hk
import jax
import jax.numpy as jnp


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
        modes_r: int = 0,
        radial_global_mixing: str = "none",
        radial_global_weight: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_theta = modes_theta
        self.radial_kernels = tuple(radial_kernels)
        self.radial_dilations = tuple(radial_dilations)
        self.radial_padding = radial_padding
        self.modes_r = modes_r
        self.radial_global_mixing = radial_global_mixing
        self.radial_global_weight = radial_global_weight

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
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
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
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
        result = theta_mixed + radial_mixed
        if self.radial_global_mixing == "reflect_fft" and self.radial_global_weight > 0.0:
            radial_global = RadialReflectSpectralConv(
                self.out_channels,
                self.modes_r,
                name="radial_reflect_global",
            )(x)
            result = result + self.radial_global_weight * radial_global
        return result


class RadialReflectSpectralConv(hk.Module):
    """Nonperiodic radial global mixing via even reflection and low Fourier modes."""

    def __init__(self, out_channels: int, modes_r: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
        if ny <= 1:
            return hk.Linear(self.out_channels, with_bias=False, name="degenerate_linear")(x)
        reflected = jnp.concatenate([x, x[:, -2:0:-1, :, :]], axis=1)
        n_reflect = reflected.shape[1]
        x_ft = jnp.fft.rfft(reflected, axis=1)
        mr = min(self.modes_r, n_reflect // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        real = hk.get_parameter(
            "weight_real",
            (mr, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        imag = hk.get_parameter(
            "weight_imag",
            (mr, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        weight = real + 1j * imag
        out_ft = jnp.zeros((batch, n_reflect // 2 + 1, nx, self.out_channels), dtype=jnp.complex64)
        low = jnp.einsum("bkxi,kio->bkxo", x_ft[:, :mr, :, :], weight)
        out_ft = out_ft.at[:, :mr, :, :].set(low)
        mixed = jnp.fft.irfft(out_ft, n=n_reflect, axis=1)
        return mixed[:, :ny, :, :]


def grid_inputs(batch, coords):
    x = batch["x"]
    b, ny, nx, _ = x.shape
    coord = jnp.broadcast_to(jnp.asarray(coords)[None, :, :, :], (b, ny, nx, 3))
    cond = jnp.concatenate([batch["mu"], batch["t"], batch["dt"]], axis=-1)
    cond = jnp.broadcast_to(cond[:, None, None, :], (b, ny, nx, cond.shape[-1]))
    return jnp.concatenate([x, coord, cond], axis=-1)


def make_fno(
    coords,
    width,
    modes_r,
    modes_theta,
    depth,
    output_channels: int,
    residual: bool = True,
    radial_kernels: list[int] | None = None,
    radial_dilations: list[int] | None = None,
    radial_padding: str = "edge",
    radial_global_mixing: str = "none",
    radial_global_weight: float = 1.0,
):
    radial_kernels = [3, 5, 5] if radial_kernels is None else radial_kernels
    radial_dilations = [1, 2, 4] if radial_dilations is None else radial_dilations

    def forward(batch):
        h = hk.Linear(width)(grid_inputs(batch, coords))
        for i in range(depth):
            spectral = ThetaSpectralRadialConv(
                width,
                modes_theta,
                radial_kernels,
                radial_dilations,
                radial_padding,
                modes_r=modes_r,
                radial_global_mixing=radial_global_mixing,
                radial_global_weight=radial_global_weight,
                name=f"theta_spectral_radial_{i}",
            )(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([width, output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


class PlainSpectralConv2D(hk.Module):
    """Rectangular-grid FNO layer without polar boundary handling."""

    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))
        mr = min(self.modes_r, max(1, ny // 2))
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        pos_real = hk.get_parameter(
            "weight_pos_real",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        pos_imag = hk.get_parameter(
            "weight_pos_imag",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        neg_real = hk.get_parameter(
            "weight_neg_real",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        neg_imag = hk.get_parameter(
            "weight_neg_imag",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        pos_weight = pos_real + 1j * pos_imag
        neg_weight = neg_real + 1j * neg_imag
        out_ft = jnp.zeros((batch, ny, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low_pos = jnp.einsum("byxi,yxio->byxo", x_ft[:, :mr, :mt, :], pos_weight)
        low_neg = jnp.einsum("byxi,yxio->byxo", x_ft[:, -mr:, :mt, :], neg_weight)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low_pos)
        out_ft = out_ft.at[:, -mr:, :mt, :].set(low_neg)
        return jnp.fft.irfft2(out_ft, s=(ny, nx), axes=(1, 2))


class ReflectSpectralConv2D(hk.Module):
    """2D spectral conv with even radial reflection and periodic theta."""

    def __init__(self, out_channels: int, modes_r: int, modes_theta: int, name: str | None = None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.modes_r = modes_r
        self.modes_theta = modes_theta

    def __call__(self, x):
        batch, ny, nx, in_channels = x.shape
        if ny > 2:
            reflected = jnp.concatenate([x, x[:, -2:0:-1, :, :]], axis=1)
        else:
            reflected = x
        n_reflect = reflected.shape[1]
        x_ft = jnp.fft.rfft2(reflected, axes=(1, 2))
        mr = min(self.modes_r, max(1, n_reflect // 2))
        mt = min(self.modes_theta, nx // 2 + 1)
        scale = 1.0 / math.sqrt(in_channels * self.out_channels)
        pos_real = hk.get_parameter(
            "weight_pos_real",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        pos_imag = hk.get_parameter(
            "weight_pos_imag",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        neg_real = hk.get_parameter(
            "weight_neg_real",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        neg_imag = hk.get_parameter(
            "weight_neg_imag",
            (mr, mt, in_channels, self.out_channels),
            init=hk.initializers.RandomNormal(scale),
        )
        pos_weight = pos_real + 1j * pos_imag
        neg_weight = neg_real + 1j * neg_imag
        out_ft = jnp.zeros((batch, n_reflect, nx // 2 + 1, self.out_channels), dtype=jnp.complex64)
        low_pos = jnp.einsum("byxi,yxio->byxo", x_ft[:, :mr, :mt, :], pos_weight)
        low_neg = jnp.einsum("byxi,yxio->byxo", x_ft[:, -mr:, :mt, :], neg_weight)
        out_ft = out_ft.at[:, :mr, :mt, :].set(low_pos)
        out_ft = out_ft.at[:, -mr:, :mt, :].set(low_neg)
        mixed = jnp.fft.irfft2(out_ft, s=(n_reflect, nx), axes=(1, 2))
        return mixed[:, :ny, :, :]


def make_plain_fno(
    coords,
    width,
    modes_r,
    modes_theta,
    depth,
    output_channels: int,
    residual: bool = True,
):
    def forward(batch):
        h = hk.Linear(width)(grid_inputs(batch, coords))
        for i in range(depth):
            spectral = PlainSpectralConv2D(width, modes_r, modes_theta, name=f"plain_spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([width, output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_reflect_fno(
    coords,
    width,
    modes_r,
    modes_theta,
    depth,
    output_channels: int,
    residual: bool = True,
):
    def forward(batch):
        h = hk.Linear(width)(grid_inputs(batch, coords))
        for i in range(depth):
            spectral = ReflectSpectralConv2D(width, modes_r, modes_theta, name=f"reflect_spectral_{i}")(h)
            pointwise = hk.Linear(width, name=f"pointwise_{i}")(h)
            h = jax.nn.gelu(spectral + pointwise)
        y = hk.nets.MLP([width, output_channels], activation=jax.nn.gelu)(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def conv_block(x, channels: int, name: str):
    h = hk.Conv2D(channels, kernel_shape=3, padding="SAME", name=f"{name}_conv0")(x)
    h = jax.nn.gelu(h)
    h = hk.Conv2D(channels, kernel_shape=3, padding="SAME", name=f"{name}_conv1")(h)
    return jax.nn.gelu(h)


def polar_periodic_pad(x, radial_pad: int, theta_pad: int, radial_padding: str):
    if radial_pad:
        x = pad_radial(x, radial_pad, radial_padding)
    if theta_pad:
        x = jnp.concatenate([x[:, :, -theta_pad:, :], x, x[:, :, :theta_pad, :]], axis=2)
    return x


def polar_conv2d(
    x,
    channels: int,
    kernel_shape,
    name: str,
    radial_padding: str,
    stride=1,
    with_bias: bool = True,
):
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape, kernel_shape)
    kh, kw = kernel_shape
    h = polar_periodic_pad(x, kh // 2, kw // 2, radial_padding)
    return hk.Conv2D(
        channels,
        kernel_shape=kernel_shape,
        stride=stride,
        padding="VALID",
        with_bias=with_bias,
        name=name,
    )(h)


def periodic_conv_block(x, channels: int, name: str, radial_padding: str):
    h = polar_conv2d(x, channels, 3, f"{name}_conv0", radial_padding)
    h = jax.nn.gelu(h)
    h = polar_conv2d(h, channels, 3, f"{name}_conv1", radial_padding)
    return jax.nn.gelu(h)


def make_unet_stepper(
    coords,
    width,
    levels,
    output_channels: int,
    residual: bool = True,
):
    def forward(batch):
        h = conv_block(grid_inputs(batch, coords), width, "input")
        skips = []
        for level in range(levels):
            skips.append(h)
            channels = width * (2 ** min(level + 1, 3))
            h = hk.Conv2D(channels, kernel_shape=3, stride=2, padding="SAME", name=f"down_{level}")(h)
            h = conv_block(jax.nn.gelu(h), channels, f"down_block_{level}")
        h = conv_block(h, h.shape[-1], "bottleneck")
        for level, skip in reversed(list(enumerate(skips))):
            h = jax.image.resize(h, skip.shape, method="nearest")
            h = jnp.concatenate([h, skip], axis=-1)
            h = conv_block(h, skip.shape[-1], f"up_block_{level}")
        y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_periodic_unet_stepper(
    coords,
    width,
    levels,
    output_channels: int,
    radial_padding: str,
    residual: bool = True,
):
    def forward(batch):
        h = periodic_conv_block(grid_inputs(batch, coords), width, "input", radial_padding)
        skips = []
        for level in range(levels):
            skips.append(h)
            channels = width * (2 ** min(level + 1, 3))
            h = polar_conv2d(h, channels, 3, f"down_{level}", radial_padding, stride=2)
            h = periodic_conv_block(jax.nn.gelu(h), channels, f"down_block_{level}", radial_padding)
        h = periodic_conv_block(h, h.shape[-1], "bottleneck", radial_padding)
        for level, skip in reversed(list(enumerate(skips))):
            h = jax.image.resize(h, skip.shape, method="nearest")
            h = jnp.concatenate([h, skip], axis=-1)
            h = periodic_conv_block(h, skip.shape[-1], f"up_block_{level}", radial_padding)
        y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


class ConvLSTMCell(hk.Module):
    def __init__(self, hidden_channels: int, name: str | None = None):
        super().__init__(name=name)
        self.hidden_channels = hidden_channels

    def __call__(self, x, h, c):
        gates = hk.Conv2D(
            4 * self.hidden_channels,
            kernel_shape=3,
            padding="SAME",
            name="gates",
        )(jnp.concatenate([x, h], axis=-1))
        i, f, o, g = jnp.split(gates, 4, axis=-1)
        c = jax.nn.sigmoid(f + 1.0) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, c


def make_convlstm_stepper(
    coords,
    width,
    depth,
    recurrent_steps,
    output_channels: int,
    residual: bool = True,
):
    def forward(batch):
        x_prev = batch["x_prev"] if "x_prev" in batch else batch["x"]
        frames = [x_prev, batch["x"]]
        frames.extend([batch["x"]] * max(0, recurrent_steps - len(frames)))
        b, ny, nx, _ = batch["x"].shape
        h = jnp.zeros((b, ny, nx, width), dtype=batch["x"].dtype)
        c = jnp.zeros_like(h)
        cell = ConvLSTMCell(width, name="convlstm_cell")
        for frame in frames:
            h, c = cell(grid_inputs(dict(batch, x=frame), coords), h, c)
        for i in range(max(0, depth - 1)):
            h = hk.Conv2D(width, kernel_shape=3, padding="SAME", name=f"post_conv_{i}")(h)
            h = jax.nn.gelu(h)
        y = hk.Conv2D(output_channels, kernel_shape=1, padding="SAME", name="output")(h)
        return batch["x"] + batch["dt"][:, None, None, :] * y if residual else y

    return hk.without_apply_rng(hk.transform(forward))


def make_model(name: str, coords, args: argparse.Namespace, output_channels: int):
    if name in {"fno", "fno_radial", "fno_flow"}:
        radial_global_mixing = "reflect_fft" if name == "fno_radial" else args.radial_global_mixing
        return make_fno(
            coords,
            args.width,
            args.modes_r,
            args.modes_theta,
            args.depth,
            output_channels,
            radial_kernels=args.radial_kernels,
            radial_dilations=args.radial_dilations,
            radial_padding=args.radial_padding,
            radial_global_mixing=radial_global_mixing,
            radial_global_weight=args.radial_global_weight,
        )
    if name == "plain_fno":
        return make_plain_fno(coords, args.width, args.modes_r, args.modes_theta, args.depth, output_channels)
    if name == "fno_reflect2d":
        return make_reflect_fno(coords, args.width, args.modes_r, args.modes_theta, args.depth, output_channels)
    if name == "unet":
        return make_unet_stepper(coords, args.width, args.unet_levels, output_channels)
    if name == "periodic_unet":
        return make_periodic_unet_stepper(
            coords,
            args.width,
            args.unet_levels,
            output_channels,
            args.radial_padding,
        )
    if name == "convlstm":
        return make_convlstm_stepper(coords, args.width, args.depth, args.convlstm_steps, output_channels)
    raise ValueError(f"Unknown trainable model: {name}")


def batch_mse(pred, target, weights):
    return jnp.mean(((pred - target) ** 2) * weights)


def set_time_span(batch, t, dt):
    out = dict(batch)
    out["t"] = t
    out["dt"] = dt
    return out

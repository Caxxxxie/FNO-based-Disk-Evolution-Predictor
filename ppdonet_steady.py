"""Load steady targets from the bundled pretrained PPDONet model."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent
PPDONET_ROOT = ROOT / "ppdonet"
sys.path.insert(0, PPDONET_ROOT.as_posix())

import onet_disk2D.grids  # noqa: E402
import onet_disk2D.run  # noqa: E402


@dataclass
class PPDONetSteadySource:
    job: object
    job_args: dict
    parameter_names: list[str]
    raw_u_min: np.ndarray
    raw_u_max: np.ndarray
    transformed_u_min: np.ndarray
    transformed_u_max: np.ndarray


def load_single_log_sigma_source() -> PPDONetSteadySource:
    run_dir = PPDONET_ROOT / "trained_network" / "single_log_sigma"
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        args_file="args.yml",
        arg_groups_file="arg_groups.yml",
        fargo_setup_file="fargo_setups.yml",
    )
    job = onet_disk2D.run.JOB(job_args)
    job.load_model(run_dir)
    transformed_u_min = np.asarray(job_args["u_min"], dtype=np.float32)
    transformed_u_max = np.asarray(job_args["u_max"], dtype=np.float32)
    raw_min = transformed_to_raw(transformed_u_min[None, :], job_args)[0]
    raw_max = transformed_to_raw(transformed_u_max[None, :], job_args)[0]
    return PPDONetSteadySource(
        job=job,
        job_args=job_args,
        parameter_names=sorted(job_args["parameter"]),
        raw_u_min=raw_min,
        raw_u_max=raw_max,
        transformed_u_min=transformed_u_min,
        transformed_u_max=transformed_u_max,
    )


def transformed_to_raw(values: np.ndarray, job_args: dict) -> np.ndarray:
    raw = values.astype(np.float32).copy()
    for i, transform in enumerate(job_args["u_transform"]):
        if transform == "log10":
            raw[:, i] = 10.0 ** raw[:, i]
    return raw


def raw_to_normalized(raw_values: np.ndarray, source: PPDONetSteadySource) -> np.ndarray:
    values = raw_values.astype(np.float32).copy()
    for i, transform in enumerate(source.job_args["u_transform"]):
        if transform == "log10":
            values[:, i] = np.log10(values[:, i])
    center = 0.5 * (source.transformed_u_min + source.transformed_u_max)
    width = source.transformed_u_max - source.transformed_u_min
    return (2.0 * (values - center) / width).astype(np.float32)


def sample_raw_parameters(source: PPDONetSteadySource, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    transformed = rng.uniform(source.transformed_u_min, source.transformed_u_max, size=(n, len(source.parameter_names)))
    return transformed_to_raw(transformed.astype(np.float32), source.job_args)


def fargo_sigma_coords(source: PPDONetSteadySource, ny: int, nx: int) -> np.ndarray:
    grids = onet_disk2D.grids.Grids(
        ymin=float(source.job.fargo_setups["ymin"]),
        ymax=float(source.job.fargo_setups["ymax"]),
        xmin=-np.pi,
        xmax=np.pi,
        ny=ny,
        nx=nx,
    )
    return np.asarray(grids.coords_fargo_all["sigma"].reshape((-1, 2)), dtype=np.float32)


def predict_log_sigma(source: PPDONetSteadySource, raw_params: np.ndarray, ny: int, nx: int) -> np.ndarray:
    coords = jnp.asarray(fargo_sigma_coords(source, ny, nx), dtype=jnp.float32)
    u = jnp.asarray(raw_params.astype(np.float32), dtype=jnp.float32)
    predict_fn = jax.jit(
        lambda u_batch: source.job.s_pred_fn(
            source.job.model.params,
            source.job.state,
            {"u_net": u_batch, "y_net": coords},
        )
    )
    pred = predict_fn(u).block_until_ready()
    return np.asarray(pred, dtype=np.float32).reshape((raw_params.shape[0], ny, nx))

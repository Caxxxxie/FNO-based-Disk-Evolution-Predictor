"""Shared constants and result containers for FARGO operator experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORBIT_PERIOD = 2.0 * math.pi
TRAINABLE_MODEL_NAMES = (
    "fno",
    "fno_radial",
    "fno_reflect2d",
    "fno_flow",
    "plain_fno",
    "unet",
    "periodic_unet",
    "convlstm",
)
ANALYTIC_BASELINE_NAMES = ("linear_extrapolation",)
MODEL_CHOICES = ("all",) + TRAINABLE_MODEL_NAMES + ANALYTIC_BASELINE_NAMES


@dataclass
class EvalMetrics:
    rmse: float
    rmse_by_channel: dict[str, float]
    rel_l2_pct: float
    rel_l2_pct_by_channel: dict[str, float]
    batches: int
    samples: int
    physics: dict[str, float] | None = None


@dataclass
class ModelResult:
    train: EvalMetrics
    validation: EvalMetrics
    heldout_time: EvalMetrics
    heldout_parameter: EvalMetrics
    heldout_parameter_time: EvalMetrics
    rollout: EvalMetrics | None
    speed_ms_per_batch: float | None
    trained_steps: int | None = None
    best_step: int | None = None
    best_validation_rmse: float | None = None
    semigroup_rmse: float | None = None
    semigroup_rmse_by_channel: dict[str, float] | None = None

def to_jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

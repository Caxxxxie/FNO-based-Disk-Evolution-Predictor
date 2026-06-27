#!/usr/bin/env python3
"""Compatibility CLI for the modular v3.1 FARGO operator benchmark.

The active code is split under ``src/fargo_operator/`` by responsibility:
configuration, data, models, training, evaluation, outputs, and runner
orchestration.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fargo_operator.config import parse_args
from src.fargo_operator.constants import (
    ANALYTIC_BASELINE_NAMES,
    MODEL_CHOICES,
    ORBIT_PERIOD,
    TRAINABLE_MODEL_NAMES,
    EvalMetrics,
    ModelResult,
    to_jsonable,
)
from src.fargo_operator.data import (
    FargoMemmapDataset,
    coordinate_grid,
    estimate_normalization,
    make_batch,
    make_consistency_batch,
    make_rollout_batch,
    sample_pairs,
    spatial_loss_weights,
    split_temporal_pairs,
)
from src.fargo_operator.evaluation import (
    accumulate_physics_metrics,
    compute_physics_metrics,
    evaluate_linear_extrapolation,
    evaluate_linear_extrapolation_rollout,
    evaluate_model,
    evaluate_persistence,
    evaluate_persistence_rollout,
    evaluate_predictions,
    evaluate_rollout,
    evaluate_semigroup,
    finalize_eval_metrics,
    linear_extrapolate_raw,
)
from src.fargo_operator.models import (
    ConvLSTMCell,
    PlainSpectralConv2D,
    RadialReflectSpectralConv,
    ReflectSpectralConv2D,
    ThetaSpectralRadialConv,
    batch_mse,
    grid_inputs,
    make_convlstm_stepper,
    make_fno,
    make_model,
    make_periodic_unet_stepper,
    make_plain_fno,
    make_reflect_fno,
    make_unet_stepper,
    set_time_span,
)
from src.fargo_operator.outputs import (
    save_loss_plot,
    save_model_checkpoint,
    speed_ms_per_batch,
    summarize_splits,
)
from src.fargo_operator.runner import main, run_linear_extrapolation, run_model
from src.fargo_operator.training import train_model


if __name__ == "__main__":
    main()

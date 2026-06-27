"""Checkpoint loading helpers for FARGO operator experiments.

The server checkpoints were saved with NumPy 2.x, whose pickle module paths use
``numpy._core``.  Some local environments still run NumPy 1.x, where those
objects live under ``numpy.core``.  The custom unpickler below keeps the
checkpoints readable in both environments.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT = "fargo_operator_v3_checkpoint"


class NumpyCompatUnpickler(pickle.Unpickler):
    """Map NumPy 2.x private pickle module names to NumPy 1.x names."""

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }

    def find_class(self, module: str, name: str) -> Any:
        module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)


def load_pickle_compat(path: str | Path) -> Any:
    """Load a pickle file that may have been written with NumPy 1.x or 2.x."""

    with Path(path).open("rb") as f:
        return NumpyCompatUnpickler(f).load()


def load_fargo_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved FARGO operator checkpoint."""

    checkpoint = load_pickle_compat(path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} did not contain a checkpoint dictionary")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    required = {"model", "params", "channels", "mean", "std", "model_config"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"{path} is missing required checkpoint keys: {missing}")
    return checkpoint


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    """Return a compact JSON-friendly summary for one checkpoint."""

    checkpoint = load_fargo_checkpoint(path)
    model_config = dict(checkpoint.get("model_config", {}))
    training_config = dict(checkpoint.get("training_config", {}))
    return {
        "path": str(path),
        "model": checkpoint.get("model"),
        "dataset": checkpoint.get("dataset"),
        "channels": list(checkpoint.get("channels", [])),
        "model_config": {
            key: model_config.get(key)
            for key in (
                "architecture",
                "width",
                "depth",
                "modes_r",
                "modes_theta",
                "radial_padding",
                "radial_global_mixing",
                "radial_global_weight",
                "unet_levels",
                "convlstm_steps",
                "time_input_units",
                "dt_units",
            )
            if key in model_config
        },
        "training_config": {
            key: training_config.get(key)
            for key in (
                "steps",
                "batch_size",
                "learning_rate",
                "grad_clip_norm",
                "seed",
                "loss_weighting",
                "temporal_bins",
                "temporal_train_frac",
                "temporal_val_frac",
                "fno_spans",
                "fno_flow_spans",
            )
            if key in training_config
        },
        "case_splits": {
            split: int(len(values))
            for split, values in checkpoint.get("case_splits", {}).items()
        },
        "temporal_pair_splits": {
            split: int(getattr(values, "shape", (len(values),))[0])
            for split, values in checkpoint.get("temporal_pair_splits", {}).items()
        },
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "NumpyCompatUnpickler",
    "checkpoint_summary",
    "load_fargo_checkpoint",
    "load_pickle_compat",
]

#!/usr/bin/env python3
"""Inspect FARGO operator checkpoints and companion metric files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fargo_operator.checkpoints import checkpoint_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Checkpoint files or directories containing *_checkpoint.pkl files.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a markdown table.")
    return parser.parse_args()


def find_checkpoints(paths: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for path in paths:
        if path.is_dir():
            checkpoints.extend(sorted(path.glob("**/*_checkpoint.pkl")))
        elif path.is_file():
            checkpoints.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(checkpoints))


def read_metrics(checkpoint_path: Path) -> dict[str, float | None]:
    metrics_path = checkpoint_path.with_name("metrics.json")
    if not metrics_path.exists():
        return {"hpt_rmse": None, "rollout_rmse": None, "speed_ms_per_batch": None}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    model = checkpoint_path.parent.name
    result = data.get("results", {}).get(model, {})
    hpt = result.get("heldout_parameter_time", {})
    rollout = result.get("rollout", {})
    return {
        "hpt_rmse": hpt.get("rmse"),
        "rollout_rmse": rollout.get("rmse"),
        "speed_ms_per_batch": result.get("speed_ms_per_batch"),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for checkpoint in find_checkpoints(args.paths):
        summary = checkpoint_summary(checkpoint)
        row = {
            "path": str(checkpoint),
            "model": summary["model"],
            "dataset": summary["dataset"],
            "channels": summary["channels"],
            "model_config": summary["model_config"],
            **read_metrics(checkpoint),
        }
        rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print("| Model | Checkpoint | Dataset | HPT RMSE | Rollout RMSE |")
    print("|---|---|---|---:|---:|")
    for row in rows:
        hpt = "" if row["hpt_rmse"] is None else f"{row['hpt_rmse']:.6f}"
        rollout = "" if row["rollout_rmse"] is None else f"{row['rollout_rmse']:.6f}"
        print(f"| {row['model']} | `{row['path']}` | `{row['dataset']}` | {hpt} | {rollout} |")


if __name__ == "__main__":
    main()

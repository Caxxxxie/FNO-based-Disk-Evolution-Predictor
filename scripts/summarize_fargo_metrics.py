#!/usr/bin/env python3
"""Print a compact comparison table from a v3 FARGO metrics.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="Path to metrics.json from benchmark_fargo_operators_v3.py")
    parser.add_argument("--horizon", default=None, help="Rollout horizon key to report. Defaults to the largest available.")
    return parser.parse_args()


def metric_value(result: dict, split: str, key: str = "rel_l2_pct") -> float | None:
    value = result.get(split)
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    return None if raw is None else float(raw)


def rollout_value(result: dict, horizon: str, key: str = "rel_l2_pct") -> float | None:
    by_horizon = result.get("rollout_by_horizon")
    if not isinstance(by_horizon, dict) or horizon not in by_horizon:
        return None
    raw = by_horizon[horizon].get(key)
    return None if raw is None else float(raw)


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:9.3f}"


def choose_horizon(results: dict, requested: str | None) -> str:
    if requested is not None:
        return str(requested)
    horizons = set()
    for result in results.values():
        by_horizon = result.get("rollout_by_horizon")
        if isinstance(by_horizon, dict):
            horizons.update(by_horizon.keys())
    if not horizons:
        return ""
    return str(max(int(horizon) for horizon in horizons))


def main() -> None:
    args = parse_args()
    payload = json.loads(args.metrics.read_text())
    results = payload["results"]
    horizon = choose_horizon(results, args.horizon)
    rows = []
    for name in sorted(results):
        result = results[name]
        rows.append(
            (
                name,
                metric_value(result, "validation"),
                metric_value(result, "heldout_parameter_time"),
                rollout_value(result, horizon) if horizon else None,
                result.get("semigroup_rmse"),
            )
        )
    print(f"metrics: {args.metrics}")
    print(f"rollout horizon: {horizon or 'n/a'}")
    print("model                         val relL2%   heldout relL2%   rollout relL2%   semigroup RMSE")
    print("-" * 91)
    for name, validation, heldout, rollout, semigroup in rows:
        semigroup_text = "n/a" if semigroup is None else f"{float(semigroup):13.5f}"
        print(
            f"{name:<28} {format_value(validation)}   {format_value(heldout)}   "
            f"{format_value(rollout)}   {semigroup_text}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print compact comparison tables from v3 FARGO metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="+", help="Path(s) to metrics.json from benchmark_fargo_operators_v3.py")
    parser.add_argument("--horizon", default=None, help="Rollout horizon key to report. Defaults to the largest available.")
    parser.add_argument("--channels", action="store_true", help="Also print heldout and rollout relative L2 by channel.")
    return parser.parse_args()


def metric_value(result: dict, split: str, key: str = "rel_l2_pct") -> float | None:
    value = result.get(split)
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    return None if raw is None else float(raw)


def rollout_value(result: dict, horizon: str, key: str = "rel_l2_pct") -> float | None:
    by_horizon = result.get("rollout_by_horizon") or result.get("rollouts")
    if not isinstance(by_horizon, dict) or horizon not in by_horizon:
        return None
    raw = by_horizon[horizon].get(key)
    return None if raw is None else float(raw)


def channel_values(result: dict, split: str) -> dict[str, float]:
    value = result.get(split)
    if not isinstance(value, dict):
        return {}
    by_channel = value.get("rel_l2_pct_by_channel")
    if not isinstance(by_channel, dict):
        return {}
    return {str(channel): float(raw) for channel, raw in by_channel.items()}


def rollout_channel_values(result: dict, horizon: str) -> dict[str, float]:
    by_horizon = result.get("rollout_by_horizon") or result.get("rollouts")
    if not isinstance(by_horizon, dict) or horizon not in by_horizon:
        return {}
    by_channel = by_horizon[horizon].get("rel_l2_pct_by_channel")
    if not isinstance(by_channel, dict):
        return {}
    return {str(channel): float(raw) for channel, raw in by_channel.items()}


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:9.3f}"


def choose_horizon(results: dict, requested: str | None) -> str:
    if requested is not None:
        return str(requested)
    horizons = set()
    for result in results.values():
        by_horizon = result.get("rollout_by_horizon") or result.get("rollouts")
        if isinstance(by_horizon, dict):
            horizons.update(by_horizon.keys())
    if not horizons:
        return ""
    return str(max(int(horizon) for horizon in horizons))


def run_label(metrics_path: Path, payload: dict) -> str:
    output_dir = payload.get("setup", {}).get("output_dir")
    if output_dir:
        return Path(output_dir).name
    return metrics_path.parent.name


def collect_rows(metrics_path: Path, requested_horizon: str | None):
    payload = json.loads(metrics_path.read_text())
    results = payload["results"]
    horizon = choose_horizon(results, requested_horizon)
    label = run_label(metrics_path, payload)
    rows = []
    for name in sorted(results):
        result = results[name]
        rows.append(
            (
                label,
                name,
                metric_value(result, "validation"),
                metric_value(result, "heldout_parameter_time"),
                rollout_value(result, horizon) if horizon else None,
                result.get("semigroup_rmse"),
            )
        )
    return horizon, rows


def print_channel_rows(rows, metrics_paths, requested_horizon):
    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text())
        results = payload["results"]
        horizon = choose_horizon(results, requested_horizon)
        label = run_label(metrics_path, payload)
        print(f"\nchannel relL2%: {label} (rollout horizon {horizon or 'n/a'})")
        print("model                         split      log_sigma   delta_v_r   delta_v_theta")
        print("-" * 82)
        for name in sorted(results):
            result = results[name]
            for split_name, values in [
                ("heldout", channel_values(result, "heldout_parameter_time")),
                ("rollout", rollout_channel_values(result, horizon) if horizon else {}),
            ]:
                if not values:
                    continue
                print(
                    f"{name:<28} {split_name:<8} "
                    f"{format_value(values.get('log_sigma'))}   "
                    f"{format_value(values.get('delta_v_r'))}   "
                    f"{format_value(values.get('delta_v_theta'))}"
                )


def main() -> None:
    args = parse_args()
    rows = []
    horizons = []
    for metrics_path in args.metrics:
        horizon, path_rows = collect_rows(metrics_path, args.horizon)
        horizons.append(horizon)
        rows.extend(path_rows)
    horizon_label = ", ".join(sorted(set(h or "n/a" for h in horizons)))
    print("metrics:")
    for metrics_path in args.metrics:
        print(f"  {metrics_path}")
    print(f"rollout horizon(s): {horizon_label}")
    print("run                                model                         val relL2%   heldout relL2%   rollout relL2%   semigroup RMSE")
    print("-" * 126)
    for label, name, validation, heldout, rollout, semigroup in rows:
        semigroup_text = "n/a" if semigroup is None else f"{float(semigroup):13.5f}"
        print(
            f"{label:<34} {name:<28} {format_value(validation)}   {format_value(heldout)}   "
            f"{format_value(rollout)}   {semigroup_text}"
        )
    if args.channels:
        print_channel_rows(rows, args.metrics, args.horizon)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a tiny FARGO3D time-series dataset for local operator sanity checks.

The script uses the vendored `fargo3d/` source tree, builds the CPU sequential
`fargo_nu` setup if needed, runs a small parameter sweep, and packs gas surface
density snapshots into an NPZ file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FARGO_ROOT = ROOT / "fargo3d"


@dataclass(frozen=True)
class FargoCase:
    alpha: float
    aspect_ratio: float
    planet_mass: float


# Keep the default smoke sweep inside the bundled PPDONet/paper parameter
# domain. In particular, ASPECTRATIO should not go below 0.05, otherwise the
# steady PPDONet baseline is queried outside its training support.
ASPECT_RATIO_MIN = 0.05
ASPECT_RATIO_MAX = 0.10


SMOKE_CASES = [
    FargoCase(alpha=5.0e-4, aspect_ratio=0.050, planet_mass=5.0e-4),
    FargoCase(alpha=1.0e-3, aspect_ratio=0.060, planet_mass=1.0e-3),
    FargoCase(alpha=2.0e-3, aspect_ratio=0.075, planet_mass=1.5e-3),
    FargoCase(alpha=7.0e-4, aspect_ratio=0.090, planet_mass=7.0e-4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="smoke_fargo_nu")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.314159265359)
    parser.add_argument("--ninterm", type=int, default=4)
    parser.add_argument("--ntot", type=int, default=24)
    parser.add_argument("--ymin", type=float, default=0.4)
    parser.add_argument("--ymax", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-random-cases", type=int, default=0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_fargo(skip_build: bool) -> None:
    exe = FARGO_ROOT / "fargo3d"
    if skip_build and exe.exists():
        return
    run(["make", "SETUP=fargo_nu", "PARALLEL=0", "GPU=0"], cwd=FARGO_ROOT)


def cases_from_args(args: argparse.Namespace) -> list[FargoCase]:
    cases = list(SMOKE_CASES)
    if args.num_random_cases > 0:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.num_random_cases):
            cases.append(
                FargoCase(
                    alpha=float(10.0 ** rng.uniform(-3.5, -2.5)),
                    aspect_ratio=float(rng.uniform(ASPECT_RATIO_MIN, ASPECT_RATIO_MAX)),
                    planet_mass=float(10.0 ** rng.uniform(-3.5, -2.7)),
                )
            )
    return cases


def write_planet_config(path: Path, planet_mass: float) -> None:
    path.write_text(
        "\n".join(
            [
                "###########################################################",
                "# Tiny local FARGO3D planet config for operator sanity checks",
                "###########################################################",
                "",
                "# Planet Name   Distance   Mass   Accretion   Feels Disk   Feels Others",
                f"Planet          1.0        {planet_mass:.10e}   0.0          NO           NO",
                "",
            ]
        )
    )


def write_par_file(path: Path, output_dir: Path, planet_cfg: Path, case: FargoCase, args: argparse.Namespace) -> None:
    path.write_text(
        f"""Setup               fargo_nu

### Disk parameters

AspectRatio         {case.aspect_ratio:.10e}
Sigma0              1.0
Alpha               {case.alpha:.10e}
SigmaSlope          0.5
FlaringIndex        0.0
DampingZone         1.15
TauDamp             0.3

### Planet parameters

PlanetConfig        {planet_cfg.as_posix()}
PlanetMass          {case.planet_mass:.10e}
ThicknessSmoothing  0.6
IndirectTerm        Yes

### Mesh parameters

Nx                  {args.nx}
Ny                  {args.ny}
Xmin               -3.14159265358979323844
Xmax                3.14159265358979323844
Ymin                {args.ymin}
Ymax                {args.ymax}

Spacing             N
XMa                 0.1
XMb                 0.2
XMc                 5.0
YMa                 0.1
YMb                 0.2
YMc                 5.0
YMy0                1.0

Frame               G
OmegaFrame          1.0005

### Output control parameters

DT                  {args.dt:.12e}
Ninterm             {args.ninterm}
Ntot                {args.ntot}

OutputDir           {output_dir.as_posix()}
"""
    )


def active_cell_centers(edges: np.ndarray, expected: int | None = None) -> np.ndarray:
    # FARGO3D writes ghost edges in y. The active radial cells are the interior
    # centers, so with Ny active zones we expect Ny + 1 interior edges after
    # removing the first three and last three ghost edges.
    if expected is not None and edges.size >= expected + 7:
        edges = edges[3:-3]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if expected is not None and centers.size != expected:
        raise ValueError(f"Expected {expected} cell centers, got {centers.size}")
    return centers.astype(np.float32)


def read_case(output_dir: Path, nx: int, ny: int, dt: float, ninterm: int) -> dict[str, np.ndarray]:
    frames = sorted(
        int(p.stem.replace("gasdens", ""))
        for p in output_dir.glob("gasdens*.dat")
        if p.stem.replace("gasdens", "").isdigit()
    )
    if not frames:
        raise FileNotFoundError(f"No gasdens*.dat files found in {output_dir}")

    sigma = []
    vx = []
    vy = []
    for frame in frames:
        sigma.append(np.fromfile(output_dir / f"gasdens{frame}.dat", dtype=np.float64).reshape(ny, nx))
        vx.append(np.fromfile(output_dir / f"gasvx{frame}.dat", dtype=np.float64).reshape(ny, nx))
        vy.append(np.fromfile(output_dir / f"gasvy{frame}.dat", dtype=np.float64).reshape(ny, nx))

    x_edges = np.loadtxt(output_dir / "domain_x.dat")
    y_edges = np.loadtxt(output_dir / "domain_y.dat")
    theta = (0.5 * (x_edges[:-1] + x_edges[1:])).astype(np.float32)
    r = active_cell_centers(y_edges, expected=ny)
    times = (np.asarray(frames, dtype=np.float32) * dt * ninterm).astype(np.float32)

    return {
        "sigma": np.asarray(sigma, dtype=np.float32),
        "log_sigma": np.log10(np.maximum(np.asarray(sigma, dtype=np.float32), 1.0e-30)),
        "v_theta": np.asarray(vx, dtype=np.float32),
        "v_r": np.asarray(vy, dtype=np.float32),
        "r": r,
        "theta": theta,
        "times": times,
        "frames": np.asarray(frames, dtype=np.int32),
    }


def main() -> None:
    args = parse_args()
    build_fargo(args.skip_build)

    cases = cases_from_args(args)
    dataset_dir = args.output_dir / args.dataset_name
    raw_root = dataset_dir / "raw"
    par_root = dataset_dir / "par"
    planet_root = dataset_dir / "planets"
    for path in [raw_root, par_root, planet_root]:
        path.mkdir(parents=True, exist_ok=True)

    packed_cases = []
    for i, case in enumerate(cases):
        case_name = f"case_{i:03d}"
        output_dir = raw_root / case_name
        planet_cfg = planet_root / f"{case_name}.cfg"
        par_file = par_root / f"{case_name}.par"
        write_planet_config(planet_cfg, case.planet_mass)
        write_par_file(par_file, output_dir, planet_cfg, case, args)
        if output_dir.exists() and not args.skip_run:
            shutil.rmtree(output_dir)
        if not args.skip_run:
            run([str(FARGO_ROOT / "fargo3d"), str(par_file)], cwd=FARGO_ROOT)
        data = read_case(output_dir, args.nx, args.ny, args.dt, args.ninterm)
        packed_cases.append(data)

    sigma = np.stack([case["sigma"] for case in packed_cases], axis=0)
    log_sigma = np.stack([case["log_sigma"] for case in packed_cases], axis=0)
    v_theta = np.stack([case["v_theta"] for case in packed_cases], axis=0)
    v_r = np.stack([case["v_r"] for case in packed_cases], axis=0)
    params = np.asarray([[c.alpha, c.aspect_ratio, c.planet_mass] for c in cases], dtype=np.float32)
    r = packed_cases[0]["r"]
    theta = packed_cases[0]["theta"]
    times = packed_cases[0]["times"]

    meta = {
        "dataset_name": args.dataset_name,
        "setup": "fargo_nu",
        "nx": args.nx,
        "ny": args.ny,
        "dt": args.dt,
        "ninterm": args.ninterm,
        "ntot": args.ntot,
        "parameter_columns": ["alpha", "aspect_ratio", "planet_mass"],
        "cases": [asdict(c) for c in cases],
        "note": "Small local FARGO3D sanity dataset, not a converged production simulation.",
    }

    dataset_dir.mkdir(parents=True, exist_ok=True)
    npz_path = dataset_dir / "dataset.npz"
    np.savez_compressed(
        npz_path,
        sigma=sigma,
        log_sigma=log_sigma,
        v_theta=v_theta,
        v_r=v_r,
        params=params,
        r=r,
        theta=theta,
        times=times,
        meta=json.dumps(meta, indent=2),
    )
    (dataset_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    if not args.keep_raw:
        shutil.rmtree(raw_root)
    print(f"Saved {npz_path}")
    print(f"shape log_sigma: {log_sigma.shape} = cases,time,ny,nx")


if __name__ == "__main__":
    main()

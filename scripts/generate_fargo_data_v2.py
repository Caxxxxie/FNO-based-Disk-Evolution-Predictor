#!/usr/bin/env python3
"""Generate the Benchmark v1 FARGO3D time-dependent disk dataset.

This is the larger counterpart to ``generate_fargo_dataset_v1.py``.  It keeps the
same FARGO3D packing workflow, but uses the PPDONet parameter domain and Sobol
sampling in normalized parameter space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
FARGO_ROOT = ROOT / "fargo3d"

PARAMETER_COLUMNS = ["alpha", "aspect_ratio", "planet_mass"]
PARAMETER_NAMES_PPDONET = ["ALPHA", "ASPECTRATIO", "PLANETMASS"]
U_MIN = np.asarray([-3.52, 0.05, -4.3], dtype=np.float64)
U_MAX = np.asarray([-1.0, 0.10, -2.7], dtype=np.float64)
U_TRANSFORM = ["log10", "", "log10"]


@dataclass(frozen=True)
class FargoCase:
    alpha: float
    aspect_ratio: float
    planet_mass: float
    split: str
    u0: float
    u1: float
    u2: float


def fargo_executable() -> Path:
    exe = FARGO_ROOT / "fargo3d"
    exe_win = FARGO_ROOT / "fargo3d.exe"
    return exe_win if exe_win.exists() else exe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="fargo_benchmark_v1")
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--train-cases", type=int, default=168)
    parser.add_argument("--val-cases", type=int, default=36)
    parser.add_argument("--test-cases", type=int, default=36)
    parser.add_argument("--sampler", choices=["sobol", "lhs"], default="sobol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.314159265359)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional safety cap for debugging; fail if Ntot exceeds this value.",
    )
    parser.add_argument("--tqs-factor", type=float, default=0.314)
    parser.add_argument(
        "--time-scale",
        choices=["viscous", "fixed"],
        default="viscous",
        help="Use 0.314 viscous times per case, or a fixed total time for debugging.",
    )
    parser.add_argument("--fixed-total-time", type=float, default=2.0 * math.pi)
    parser.add_argument("--ymin", type=float, default=0.4)
    parser.add_argument("--ymax", type=float, default=2.5)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma-slope", type=float, default=0.5)
    parser.add_argument("--flaring-index", type=float, default=0.0)
    parser.add_argument("--omega-frame", type=float, default=1.0005)
    parser.add_argument("--thickness-smoothing", type=float, default=0.6)
    parser.add_argument("--damping-zone", type=float, default=1.15)
    parser.add_argument("--tau-damp", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--pack-npz", action="store_true", help="Also write dataset.npz. Expensive for benchmark size.")
    parser.add_argument("--cpu", action="store_true", help="Build/run CPU FARGO3D instead of GPU.")
    parser.add_argument("--parallel", action="store_true", help="Build FARGO3D with PARALLEL=1.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and par files without running FARGO3D.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cases != args.train_cases + args.val_cases + args.test_cases:
        raise ValueError("--cases must equal train + validation + test cases")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.max_steps is not None and args.max_steps < args.frames - 1:
        raise ValueError("--max-steps must be at least frames - 1")


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    buildtools = ROOT / ".buildtools" / "Library"
    if buildtools.exists():
        env["PATH"] = os.pathsep.join(
            [
                str(buildtools / "usr" / "bin"),
                str(buildtools / "x86_64-w64-mingw32" / "bin"),
                str(buildtools / "bin"),
                env.get("PATH", ""),
            ]
        )
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def build_fargo(args: argparse.Namespace) -> None:
    exe = fargo_executable()
    if args.skip_build:
        if exe.exists():
            return
        raise FileNotFoundError(
            f"--skip-build was set, but the FARGO3D executable does not exist: {exe}. "
            "Build FARGO3D first or rerun without --skip-build."
        )
    gpu = "0" if args.cpu else "1"
    parallel = "1" if args.parallel else "0"
    run(["make", "SETUP=fargo_nu", f"PARALLEL={parallel}", f"GPU={gpu}"], cwd=FARGO_ROOT)


def normalized_to_physical(u_norm: np.ndarray) -> np.ndarray:
    transformed = 0.5 * (u_norm + 1.0) * (U_MAX - U_MIN) + U_MIN
    physical = transformed.copy()
    physical[:, 0] = 10.0 ** transformed[:, 0]
    physical[:, 2] = 10.0 ** transformed[:, 2]
    return physical


def sample_normalized_parameters(args: argparse.Namespace) -> np.ndarray:
    if args.sampler == "sobol":
        exponent = math.ceil(math.log2(args.cases))
        sampler = qmc.Sobol(d=3, scramble=True, seed=args.seed)
        unit = sampler.random_base2(exponent)[: args.cases]
    else:
        sampler = qmc.LatinHypercube(d=3, seed=args.seed)
        unit = sampler.random(args.cases)
    return (2.0 * unit - 1.0).astype(np.float64)


def make_cases(args: argparse.Namespace) -> list[FargoCase]:
    u_norm = sample_normalized_parameters(args)
    physical = normalized_to_physical(u_norm)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(args.cases)
    split_by_original = np.empty(args.cases, dtype=object)
    split_by_original[order[: args.train_cases]] = "train"
    split_by_original[order[args.train_cases : args.train_cases + args.val_cases]] = "validation"
    split_by_original[order[args.train_cases + args.val_cases :]] = "test"
    cases = []
    for i in range(args.cases):
        cases.append(
            FargoCase(
                alpha=float(physical[i, 0]),
                aspect_ratio=float(physical[i, 1]),
                planet_mass=float(physical[i, 2]),
                split=str(split_by_original[i]),
                u0=float(u_norm[i, 0]),
                u1=float(u_norm[i, 1]),
                u2=float(u_norm[i, 2]),
            )
        )
    return cases


def viscous_timescale(case: FargoCase) -> float:
    # With G = Mstar = rp = Omega_p = 1, nu = alpha * h^2, so t_nu = rp^2 / nu.
    return 1.0 / (case.alpha * case.aspect_ratio * case.aspect_ratio)


def total_time(case: FargoCase, args: argparse.Namespace) -> float:
    if args.time_scale == "fixed":
        return args.fixed_total_time
    return args.tqs_factor * viscous_timescale(case)


def write_planet_config(path: Path, planet_mass: float) -> None:
    path.write_text(
        "\n".join(
            [
                "###########################################################",
                "# Benchmark v1 FARGO3D planet config",
                "###########################################################",
                "",
                "# Planet Name   Distance   Mass   Accretion   Feels Disk   Feels Others",
                f"Planet          1.0        {planet_mass:.10e}   0.0          NO           NO",
                "",
            ]
        )
    )


def write_par_file(path: Path, output_dir: Path, planet_cfg: Path, case: FargoCase, args: argparse.Namespace) -> dict:
    output_ref = Path(os.path.relpath(output_dir, ROOT)).as_posix()
    planet_ref = Path(os.path.relpath(planet_cfg, ROOT)).as_posix()
    case_total_time = total_time(case, args)
    if args.time_scale == "fixed" and case_total_time < args.dt * (args.frames - 1):
        dt = case_total_time / (args.frames - 1)
        ninterm = 1
        ntot = args.frames - 1
        actual_total_time = case_total_time
    else:
        dt = args.dt
        ntot_raw = math.ceil(case_total_time / dt)
        ninterm = max(1, math.ceil(ntot_raw / (args.frames - 1)))
        ntot = ninterm * (args.frames - 1)
        actual_total_time = dt * ntot

    if args.max_steps is not None and ntot > args.max_steps:
        raise ValueError(
            f"Case requires Ntot={ntot} steps, exceeding --max-steps={args.max_steps}. "
            "Use a shorter time scale, larger dt, or raise the cap."
        )

    path.write_text(
        f"""Setup               fargo_nu

### Disk parameters

AspectRatio         {case.aspect_ratio:.10e}
Sigma0              {args.sigma0:.10e}
Alpha               {case.alpha:.10e}
SigmaSlope          {args.sigma_slope:.10e}
FlaringIndex        {args.flaring_index:.10e}
DampingZone         {args.damping_zone:.10e}
TauDamp             {args.tau_damp:.10e}

### Planet parameters

PlanetConfig        {planet_ref}
PlanetMass          {case.planet_mass:.10e}
ThicknessSmoothing  {args.thickness_smoothing:.10e}
IndirectTerm        Yes

### Mesh parameters

Nx                  {args.nx}
Ny                  {args.ny}
Xmin               -3.14159265358979323844
Xmax                3.14159265358979323844
Ymin                {args.ymin:.10e}
Ymax                {args.ymax:.10e}

Spacing             N
XMa                 0.1
XMb                 0.2
XMc                 5.0
YMa                 0.1
YMb                 0.2
YMc                 5.0
YMy0                1.0

Frame               G
OmegaFrame          {args.omega_frame:.10e}
FuncArchFile        fargo3d/std/func_arch.cfg

### Output control parameters

DT                  {dt:.12e}
Ninterm             {ninterm}
Ntot                {ntot}

OutputDir           {output_ref}
"""
    )
    return {
        "dt": dt,
        "ninterm": ninterm,
        "ntot": ntot,
        "target_total_time": case_total_time,
        "total_time": actual_total_time,
    }


def active_cell_centers(edges: np.ndarray, expected: int | None = None) -> np.ndarray:
    if expected is not None and edges.size >= expected + 7:
        edges = edges[3:-3]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if expected is not None and centers.size != expected:
        raise ValueError(f"Expected {expected} cell centers, got {centers.size}")
    return centers.astype(np.float32)


def read_field(path: Path, nx: int, ny: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    expected = nx * ny
    if values.size > expected:
        values = values[-expected:]
    elif values.size != expected:
        raise ValueError(f"Expected {expected} values in {path}, got {values.size}")
    return values.reshape(ny, nx)


def initial_velocity_background(r: np.ndarray, case: FargoCase, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    omega = np.sqrt(1.0 / (r * r * r))
    pressure_factor = 1.0 + case.aspect_ratio**2 * (r ** (2.0 * args.flaring_index)) * (
        2.0 * args.flaring_index - 1.0 - args.sigma_slope
    )
    if np.any(pressure_factor <= 0.0):
        raise ValueError("Initial azimuthal velocity background became imaginary")
    v_theta_bg = omega * r * np.sqrt(pressure_factor) - args.omega_frame * r
    v_r_bg = np.zeros_like(v_theta_bg)
    return v_r_bg.astype(np.float32), v_theta_bg.astype(np.float32)


def read_case(output_dir: Path, case: FargoCase, args: argparse.Namespace, timing: dict) -> dict[str, np.ndarray]:
    frames = sorted(
        int(p.stem.replace("gasdens", ""))
        for p in output_dir.glob("gasdens*.dat")
        if p.stem.replace("gasdens", "").isdigit()
    )
    if len(frames) != args.frames:
        raise ValueError(f"Expected {args.frames} frames in {output_dir}, got {len(frames)}")

    sigma = []
    vx = []
    vy = []
    for frame in frames:
        sigma.append(read_field(output_dir / f"gasdens{frame}.dat", args.nx, args.ny))
        vx.append(read_field(output_dir / f"gasvx{frame}.dat", args.nx, args.ny))
        vy.append(read_field(output_dir / f"gasvy{frame}.dat", args.nx, args.ny))

    x_edges = np.loadtxt(output_dir / "domain_x.dat")
    y_edges = np.loadtxt(output_dir / "domain_y.dat")
    theta = (0.5 * (x_edges[:-1] + x_edges[1:])).astype(np.float32)
    r = active_cell_centers(y_edges, expected=args.ny)
    times = (np.asarray(frames, dtype=np.float64) * timing["dt"] * timing["ninterm"]).astype(np.float32)
    t_norm = (times / timing["total_time"]).astype(np.float32)

    sigma = np.asarray(sigma, dtype=np.float32)
    v_theta = np.asarray(vx, dtype=np.float32)
    v_r = np.asarray(vy, dtype=np.float32)
    v_r_bg, v_theta_bg = initial_velocity_background(r, case, args)
    return {
        "log_sigma": np.log10(np.maximum(sigma, 1.0e-30)).astype(np.float32),
        "delta_v_r": (v_r - v_r_bg[None, :, None]).astype(np.float32),
        "delta_v_theta": (v_theta - v_theta_bg[None, :, None]).astype(np.float32),
        "v_r": v_r,
        "v_theta": v_theta,
        "v_r_background": v_r_bg,
        "v_theta_background": v_theta_bg,
        "r": r,
        "theta": theta,
        "times": times,
        "t_norm": t_norm,
        "frames": np.asarray(frames, dtype=np.int32),
    }


def create_memmaps(dataset_dir: Path, shape: tuple[int, int, int, int]) -> dict[str, np.memmap]:
    arrays = {}
    for name in ["log_sigma", "delta_v_r", "delta_v_theta", "v_r", "v_theta"]:
        path = dataset_dir / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    return arrays


def flush_memmaps(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.dry_run:
        build_fargo(args)

    cases = make_cases(args)
    dataset_dir = args.output_dir / args.dataset_name
    raw_root = dataset_dir / "raw"
    par_root = dataset_dir / "par"
    planet_root = dataset_dir / "planets"
    for path in [dataset_dir, raw_root, par_root, planet_root]:
        path.mkdir(parents=True, exist_ok=True)

    params = np.asarray([[c.alpha, c.aspect_ratio, c.planet_mass] for c in cases], dtype=np.float32)
    params_norm = np.asarray([[c.u0, c.u1, c.u2] for c in cases], dtype=np.float32)
    case_split = np.asarray([c.split for c in cases])
    np.save(dataset_dir / "params.npy", params)
    np.save(dataset_dir / "params_norm.npy", params_norm)
    np.save(dataset_dir / "case_split.npy", case_split)

    memmaps = None
    r = theta = frames = None
    times = np.empty((args.cases, args.frames), dtype=np.float32)
    t_norm = np.empty((args.cases, args.frames), dtype=np.float32)
    timing_by_case = []
    background_v_r = np.empty((args.cases, args.ny), dtype=np.float32)
    background_v_theta = np.empty((args.cases, args.ny), dtype=np.float32)

    for i, case in enumerate(cases):
        case_name = f"case_{i:03d}"
        output_dir = raw_root / case_name
        planet_cfg = planet_root / f"{case_name}.cfg"
        par_file = par_root / f"{case_name}.par"
        write_planet_config(planet_cfg, case.planet_mass)
        timing = write_par_file(par_file, output_dir, planet_cfg, case, args)
        timing_by_case.append(timing)

        if output_dir.exists() and not (args.skip_run or args.dry_run):
            shutil.rmtree(output_dir)
        if not (args.skip_run or args.dry_run):
            output_dir.mkdir(parents=True, exist_ok=True)
            par_ref = Path(os.path.relpath(par_file, ROOT)).as_posix()
            exe_ref = Path("fargo3d") / fargo_executable().name
            run([str(exe_ref), par_ref], cwd=ROOT)

        if args.dry_run:
            continue

        data = read_case(output_dir, case, args, timing)
        if memmaps is None:
            shape = (args.cases, args.frames, args.ny, args.nx)
            memmaps = create_memmaps(dataset_dir, shape)
            r = data["r"]
            theta = data["theta"]
            frames = data["frames"]
            np.save(dataset_dir / "r.npy", r)
            np.save(dataset_dir / "theta.npy", theta)
            np.save(dataset_dir / "frames.npy", frames)

        for name, array in memmaps.items():
            array[i] = data[name]
        times[i] = data["times"]
        t_norm[i] = data["t_norm"]
        background_v_r[i] = data["v_r_background"]
        background_v_theta[i] = data["v_theta_background"]

        if not args.keep_raw:
            shutil.rmtree(output_dir)
        print(f"Packed {case_name}: split={case.split}, total_time={timing['total_time']:.6e}")

    if memmaps is not None:
        flush_memmaps(memmaps)
        np.save(dataset_dir / "times.npy", times)
        np.save(dataset_dir / "t_norm.npy", t_norm)
        np.save(dataset_dir / "v_r_background.npy", background_v_r)
        np.save(dataset_dir / "v_theta_background.npy", background_v_theta)

    time_meta = {
        "scale": args.time_scale,
        "dt_requested": args.dt,
        "cadence": "uniform FARGO3D saved frames; nonuniform selection is deferred to training",
    }
    if args.time_scale == "fixed":
        time_meta.update(
            {
                "fixed_total_time": args.fixed_total_time,
                "definition": "Fixed physical integration time in FARGO code units; 2*pi corresponds to one orbit at r=1.",
            }
        )
    else:
        time_meta.update(
            {
                "tqs_factor": args.tqs_factor,
                "definition": "T_qs = 0.314 * t_nu, t_nu = 1 / (alpha * h^2) in code units",
            }
        )

    meta = {
        "dataset_name": args.dataset_name,
        "setup": "fargo_nu",
        "sampler": args.sampler,
        "seed": args.seed,
        "cases": args.cases,
        "split_counts": {
            "train": args.train_cases,
            "validation": args.val_cases,
            "test": args.test_cases,
        },
        "grid": {"ny": args.ny, "nx": args.nx, "ymin": args.ymin, "ymax": args.ymax},
        "frames": args.frames,
        "time": time_meta,
        "channels": {
            "state": ["log_sigma", "delta_v_r", "delta_v_theta"],
            "also_saved": ["v_r", "v_theta", "v_r_background", "v_theta_background"],
            "delta_definition": "velocity minus fargo_nu initial equilibrium background",
        },
        "parameter_columns": PARAMETER_COLUMNS,
        "ppdonet_parameter_names": PARAMETER_NAMES_PPDONET,
        "u_transform": U_TRANSFORM,
        "u_min": U_MIN.tolist(),
        "u_max": U_MAX.tolist(),
        "cases_detail": [asdict(case) for case in cases],
        "timing_by_case": timing_by_case,
        "storage": "npy directory",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if args.pack_npz and memmaps is not None:
        np.savez_compressed(
            dataset_dir / "dataset.npz",
            log_sigma=np.load(dataset_dir / "log_sigma.npy", mmap_mode="r"),
            delta_v_r=np.load(dataset_dir / "delta_v_r.npy", mmap_mode="r"),
            delta_v_theta=np.load(dataset_dir / "delta_v_theta.npy", mmap_mode="r"),
            v_r=np.load(dataset_dir / "v_r.npy", mmap_mode="r"),
            v_theta=np.load(dataset_dir / "v_theta.npy", mmap_mode="r"),
            params=params,
            params_norm=params_norm,
            case_split=case_split,
            r=np.load(dataset_dir / "r.npy"),
            theta=np.load(dataset_dir / "theta.npy"),
            times=times,
            t_norm=t_norm,
            meta=json.dumps(meta, indent=2),
        )

    print(f"Saved dataset scaffold/data under {dataset_dir}")


if __name__ == "__main__":
    main()

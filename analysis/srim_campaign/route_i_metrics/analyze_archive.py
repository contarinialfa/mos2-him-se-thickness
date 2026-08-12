"""Independent audit of the MoS2 30-keV SRIM route-(i) campaign.

This script is deliberately read-only with respect to the SRIM archive.  It
streams every TRANSMIT.txt record, constructs per-incident-ion observables,
pools only the two independent 100k seeds, compares the new Model-B driver
with the frozen Letter table, and reproduces the Letter's frozen-effective-
variance weighted-least-squares fit with both driver tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ARCHIVE = WORKSPACE / "mos2_30keV_route_i_srim_axis_resolution_v1"
DATA_CSV = WORKSPACE / "analysis" / "paper" / "flakes_bayesian_normalized_summary.csv"

E0_EV = 30_000.0
ANGLE_CUT_DEG = 20.0
COS20 = math.cos(math.radians(ANGLE_CUT_DEG))
SEC_CAP = 4.0
LAMBDA_B_FROZEN = 15.53
S_FROZEN = 0.7017
S_FIT_SD = 0.031
S_REPEAT_SD = 0.070

FROZEN_U = np.array(
    [0, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200,
     225, 250, 275, 300, 500, 560],
    dtype=float,
)
FROZEN_B = np.array(
    [1.0, 0.9994, 0.9942, 0.9867, 0.9807, 0.9684, 0.9557,
     0.9112, 0.8382, 0.7166, 0.5683, 0.4075, 0.2491, 0.1307,
     0.0597, 0.0196, 0.0055, 0.0, 0.0],
    dtype=float,
)

METRICS = (
    "transmission",
    "flux_gt20",
    "sqrt_flux_gt20",
    "sqrt_flux_all",
    "b4",
    "b_uncapped",
)


@dataclass
class RunStats:
    index: int
    name: str
    family: str
    seed: int
    thickness_nm: float
    n_incident: int
    status_path: str
    transmit_path: str
    n_transmitted: int = 0
    sums: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in METRICS})
    sums2: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in METRICS})
    energy_sum_eV: float = 0.0
    energy_sum2_eV2: float = 0.0
    angle_sum_deg: float = 0.0
    angle_sum2_deg2: float = 0.0
    cap_count: int = 0
    max_direction_norm_error: float = 0.0
    min_cosx: float = 1.0
    max_energy_eV: float = 0.0
    first10k_transmit_sha256: str = ""
    first10k_n_transmitted: int = 0
    first10k_sums: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in METRICS})


def mean_and_se(total: float, total2: float, n: int) -> tuple[float, float]:
    """Sample mean and standard error for n incident-ion values, including zeros."""
    if n <= 0:
        return math.nan, math.nan
    mean = total / n
    if n == 1:
        return mean, math.nan
    var = max((total2 - total * total / n) / (n - 1), 0.0)
    return mean, math.sqrt(var / n)


def conditional_mean_se(total: float, total2: float, n: int) -> tuple[float, float]:
    return mean_and_se(total, total2, n)


def record_values(energy_eV: float, cosx: float) -> dict[str, float]:
    sqrt_e = math.sqrt(max(energy_eV, 0.0) / E0_EV)
    sec = 1.0 / cosx
    gt20 = cosx < COS20
    return {
        "transmission": 1.0,
        "flux_gt20": 1.0 if gt20 else 0.0,
        "sqrt_flux_gt20": sqrt_e if gt20 else 0.0,
        "sqrt_flux_all": sqrt_e,
        "b4": sqrt_e * min(sec, SEC_CAP),
        "b_uncapped": sqrt_e * sec,
    }


def parse_transmit(run: RunStats) -> None:
    h10 = hashlib.sha256()
    with Path(run.transmit_path).open("r", encoding="latin-1", errors="strict") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.startswith("T"):
                continue
            parts = line[1:].split()
            if len(parts) != 9:
                raise ValueError(
                    f"Unexpected TRANSMIT record at {run.transmit_path}:{line_no}: {line!r}"
                )
            ion_printed = int(parts[0])
            ion_id = run.n_incident if ion_printed == 0 else ion_printed
            atom_z = int(parts[1])
            energy = float(parts[2])
            depth = float(parts[3])
            cosx, cosy, cosz = map(float, parts[6:9])

            if atom_z != 2:
                raise ValueError(f"Non-He transmitted atom Z={atom_z} in {run.name}")
            if not (0.0 < cosx <= 1.000001):
                raise ValueError(f"Non-forward transmitted record cosX={cosx} in {run.name}")
            if not (0.0 <= energy <= E0_EV * 1.000001):
                raise ValueError(f"Unphysical exit energy {energy} eV in {run.name}")
            target_A = 10.0 * run.thickness_nm
            if depth < target_A - 5.0:
                raise ValueError(
                    f"Exit depth {depth} A is materially before rear surface {target_A} A in {run.name}"
                )

            norm_error = abs(cosx * cosx + cosy * cosy + cosz * cosz - 1.0)
            run.max_direction_norm_error = max(run.max_direction_norm_error, norm_error)
            run.min_cosx = min(run.min_cosx, cosx)
            run.max_energy_eV = max(run.max_energy_eV, energy)
            angle = math.degrees(math.acos(min(max(cosx, -1.0), 1.0)))
            vals = record_values(energy, cosx)

            run.n_transmitted += 1
            run.energy_sum_eV += energy
            run.energy_sum2_eV2 += energy * energy
            run.angle_sum_deg += angle
            run.angle_sum2_deg2 += angle * angle
            if cosx < 1.0 / SEC_CAP:
                run.cap_count += 1
            for key, value in vals.items():
                run.sums[key] += value
                run.sums2[key] += value * value

            if ion_id <= 10_000:
                # Canonical tokens avoid newline/platform differences while retaining every value.
                h10.update((" ".join(parts) + "\n").encode("ascii"))
                run.first10k_n_transmitted += 1
                for key, value in vals.items():
                    run.first10k_sums[key] += value

    run.first10k_transmit_sha256 = h10.hexdigest()
    if run.sums["transmission"] != run.n_transmitted:
        raise AssertionError(f"Transmission count mismatch in {run.name}")


def discover_and_parse_runs() -> list[RunStats]:
    status_paths = sorted((ARCHIVE / "MoS2").rglob("status.json"))
    if len(status_paths) != 118:
        raise RuntimeError(f"Expected 118 status files, found {len(status_paths)}")
    runs: list[RunStats] = []
    for i, status_path in enumerate(status_paths, start=1):
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
        if status.get("state") != "completed" or status.get("trim_exit_code") != 0:
            raise RuntimeError(f"Incomplete/failed run: {status_path}")
        if status.get("warnings"):
            raise RuntimeError(f"Runner warning in {status_path}: {status['warnings']}")
        p = status["parameters"]
        transmit_path = status_path.parent / "artifacts" / "SRIM Outputs" / "TRANSMIT.txt"
        if not transmit_path.is_file():
            raise FileNotFoundError(transmit_path)
        run = RunStats(
            index=int(status["index"]),
            name=status["name"],
            family=p["campaign_family"],
            seed=int(p["random_seed"]),
            thickness_nm=float(p["layer_1_width_nm"]),
            n_incident=int(p["ion_count"]),
            status_path=str(status_path),
            transmit_path=str(transmit_path),
        )
        parse_transmit(run)
        runs.append(run)
        if i % 10 == 0 or i == len(status_paths):
            print(f"Parsed {i:3d}/{len(status_paths)} TRANSMIT files", flush=True)
    return sorted(runs, key=lambda r: r.index)


def run_to_row(run: RunStats) -> dict[str, object]:
    row: dict[str, object] = {
        "index": run.index,
        "name": run.name,
        "family": run.family,
        "seed": run.seed,
        "u_nm": run.thickness_nm,
        "n_incident": run.n_incident,
        "n_transmitted": run.n_transmitted,
    }
    for key in METRICS:
        mean, se = mean_and_se(run.sums[key], run.sums2[key], run.n_incident)
        row[key] = mean
        row[f"{key}_se"] = se
    e_mean, e_se = conditional_mean_se(
        run.energy_sum_eV / 1000.0,
        run.energy_sum2_eV2 / 1_000_000.0,
        run.n_transmitted,
    )
    a_mean, a_se = conditional_mean_se(
        run.angle_sum_deg, run.angle_sum2_deg2, run.n_transmitted
    )
    row.update(
        mean_exit_energy_keV=e_mean,
        mean_exit_energy_keV_se=e_se,
        mean_exit_angle_deg=a_mean,
        mean_exit_angle_deg_se=a_se,
        cap_count=run.cap_count,
        cap_fraction_transmitted=(run.cap_count / run.n_transmitted if run.n_transmitted else 0.0),
        cap_fraction_incident=run.cap_count / run.n_incident,
        max_direction_norm_error=run.max_direction_norm_error,
        min_cosx=run.min_cosx,
        max_energy_eV=run.max_energy_eV,
        first10k_n_transmitted=run.first10k_n_transmitted,
        first10k_transmit_sha256=run.first10k_transmit_sha256,
    )
    return row


def aggregate_runs(group: Iterable[RunStats], label: str, thickness: float) -> dict[str, object]:
    group = list(group)
    n = sum(r.n_incident for r in group)
    nt = sum(r.n_transmitted for r in group)
    row: dict[str, object] = {
        "label": label,
        "u_nm": thickness,
        "seeds": ";".join(str(r.seed) for r in group),
        "n_incident": n,
        "n_transmitted": nt,
    }
    for key in METRICS:
        total = sum(r.sums[key] for r in group)
        total2 = sum(r.sums2[key] for r in group)
        mean, se = mean_and_se(total, total2, n)
        row[key] = mean
        row[f"{key}_se"] = se
    es = sum(r.energy_sum_eV for r in group) / 1000.0
    es2 = sum(r.energy_sum2_eV2 for r in group) / 1_000_000.0
    ans = sum(r.angle_sum_deg for r in group)
    ans2 = sum(r.angle_sum2_deg2 for r in group)
    e_mean, e_se = conditional_mean_se(es, es2, nt)
    a_mean, a_se = conditional_mean_se(ans, ans2, nt)
    cap_count = sum(r.cap_count for r in group)
    row.update(
        mean_exit_energy_keV=e_mean,
        mean_exit_energy_keV_se=e_se,
        mean_exit_angle_deg=a_mean,
        mean_exit_angle_deg_se=a_se,
        cap_count=cap_count,
        cap_fraction_transmitted=(cap_count / nt if nt else 0.0),
        cap_fraction_incident=cap_count / n,
        model_b_full=row["b4"] * (1.0 - math.exp(-thickness / LAMBDA_B_FROZEN)),
        model_b_full_se=row["b4_se"] * (1.0 - math.exp(-thickness / LAMBDA_B_FROZEN)),
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def production_pooled(runs: list[RunStats]) -> list[dict[str, object]]:
    prod = [r for r in runs if "production_100k" in r.family]
    by_u: dict[float, list[RunStats]] = {}
    for run in prod:
        by_u.setdefault(run.thickness_nm, []).append(run)
    if len(by_u) != 41 or any(len(v) != 2 for v in by_u.values()):
        raise RuntimeError("Production grid is not 41 thicknesses x 2 seeds")
    return [aggregate_runs(by_u[u], "production_pooled", u) for u in sorted(by_u)]


def shared_prefix_audit(runs: list[RunStats]) -> dict[str, object]:
    recon = {r.thickness_nm: r for r in runs if "recon" in r.family}
    prod_seed = {
        r.thickness_nm: r
        for r in runs
        if "production_100k" in r.family and r.seed == 13579
    }
    overlap = sorted(set(recon) & set(prod_seed))
    exact = []
    for u in overlap:
        r10, r100 = recon[u], prod_seed[u]
        same_hash = r10.first10k_transmit_sha256 == r100.first10k_transmit_sha256
        same_n = r10.n_transmitted == r100.first10k_n_transmitted
        same_sums = all(
            math.isclose(r10.sums[k], r100.first10k_sums[k], rel_tol=0.0, abs_tol=1e-12)
            for k in METRICS
        )
        exact.append(same_hash and same_n and same_sums)
    return {
        "overlap_count": len(overlap),
        "all_transmitted_prefixes_exact": all(exact),
        "exact_count": sum(exact),
        "overlap_u_nm": overlap,
    }


def approximate_flat_interval(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    y = np.array([float(r[metric]) for r in rows])
    se = np.array([float(r[f"{metric}_se"]) for r in rows])
    x = np.array([float(r["u_nm"]) for r in rows])
    imax = int(np.argmax(y))
    z = (y[imax] - y) / np.sqrt(se[imax] ** 2 + se ** 2)
    mask = z <= 1.96
    return {
        "grid_peak_u_nm": float(x[imax]),
        "grid_peak_value": float(y[imax]),
        "grid_peak_se": float(se[imax]),
        "approx_95pct_indistinguishable_u_min_nm": float(x[mask].min()),
        "approx_95pct_indistinguishable_u_max_nm": float(x[mask].max()),
    }


def quadratic_peak(
    rows: list[dict[str, object]], metric: str, lo: float, hi: float, seed: int = 20260722
) -> dict[str, object]:
    selected = [r for r in rows if lo <= float(r["u_nm"]) <= hi]
    x = np.array([float(r["u_nm"]) for r in selected])
    y = np.array([float(r[metric]) for r in selected])
    se = np.array([float(r[f"{metric}_se"]) for r in selected])
    A = np.column_stack([x * x, x, np.ones_like(x)])
    W = np.diag(1.0 / (se * se))
    cov = np.linalg.inv(A.T @ W @ A)
    beta = cov @ (A.T @ W @ y)
    vertex = -beta[1] / (2.0 * beta[0])
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(beta, cov, size=200_000)
    good = (draws[:, 0] < 0.0)
    vertices = -draws[good, 1] / (2.0 * draws[good, 0])
    vertices = vertices[(vertices >= lo) & (vertices <= hi)]
    q16, q50, q84 = np.quantile(vertices, [0.16, 0.5, 0.84])
    return {
        "window_u_nm": [lo, hi],
        "n_points": len(x),
        "vertex_u_nm": float(vertex),
        "vertex_mc_median_u_nm": float(q50),
        "vertex_mc_16_u_nm": float(q16),
        "vertex_mc_84_u_nm": float(q84),
        "accepted_mc_fraction": float(len(vertices) / len(draws)),
        "quadratic_coefficients": beta.tolist(),
        "reduced_chi2_local": float(
            np.sum(((A @ beta - y) / se) ** 2) / max(len(x) - 3, 1)
        ),
    }


def seed_peak_summary(runs: list[RunStats], metric: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for seed in (13579, 24680):
        rows = [run_to_row(r) for r in runs if "production_100k" in r.family and r.seed == seed]
        imax = int(np.argmax([float(r[metric]) for r in rows]))
        out[str(seed)] = {
            "u_nm": float(rows[imax]["u_nm"]),
            "value": float(rows[imax][metric]),
        }
    return out


def build_new_b_table(
    runs: list[RunStats], pooled: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    points: dict[float, tuple[float, float, str, int]] = {0.0: (1.0, 0.0, "analytic", 0)}
    for row in pooled:
        points[float(row["u_nm"])] = (
            float(row["b4"]), float(row["b4_se"]), "production_pooled", int(row["n_incident"])
        )
    for run in runs:
        if "recon" not in run.family:
            continue
        if run.thickness_nm in points:
            continue
        mean, se = mean_and_se(run.sums["b4"], run.sums2["b4"], run.n_incident)
        points[run.thickness_nm] = (mean, se, "recon", run.n_incident)
    rows = [
        {"u_nm": u, "b4": values[0], "b4_se": values[1], "source": values[2], "n_incident": values[3]}
        for u, values in sorted(points.items())
    ]
    return (
        np.array([float(r["u_nm"]) for r in rows]),
        np.array([float(r["b4"]) for r in rows]),
        rows,
    )


def build_metric_table(
    runs: list[RunStats], pooled: list[dict[str, object]], metric: str, value_at_zero: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Build a continuous campaign table using production where available.

    The two independent 100k production seeds supply 40--320 nm.  The 10k
    reconnaissance campaign is used only outside that interval; overlap is
    intentionally not pooled because the 10k run is an exact random-stream
    prefix of seed 13579.
    """
    points: dict[float, tuple[float, float, str, int]] = {
        0.0: (value_at_zero, 0.0, "analytic", 0)
    }
    for row in pooled:
        points[float(row["u_nm"])] = (
            float(row[metric]),
            float(row[f"{metric}_se"]),
            "production_pooled",
            int(row["n_incident"]),
        )
    for run in runs:
        if "recon" not in run.family or run.thickness_nm in points:
            continue
        mean, se = mean_and_se(run.sums[metric], run.sums2[metric], run.n_incident)
        points[run.thickness_nm] = (mean, se, "recon", run.n_incident)
    rows = [
        {
            "u_nm": u,
            metric: values[0],
            f"{metric}_se": values[1],
            "source": values[2],
            "n_incident": values[3],
        }
        for u, values in sorted(points.items())
    ]
    return (
        np.array([float(r["u_nm"]) for r in rows]),
        np.array([float(r[metric]) for r in rows]),
        rows,
    )


def build_sqrt_le20_table(
    runs: list[RunStats], pooled: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Build Q_<=20 = sum sqrt(E/E0)*1[theta<=20]/N_incident."""
    points: dict[float, tuple[float, float, str, int]] = {
        0.0: (1.0, 0.0, "analytic", 0)
    }
    prod = [r for r in runs if "production_100k" in r.family]
    for u in sorted({r.thickness_nm for r in prod}):
        group = [r for r in prod if r.thickness_nm == u]
        n = sum(r.n_incident for r in group)
        total = sum(r.sums["sqrt_flux_all"] - r.sums["sqrt_flux_gt20"] for r in group)
        # The >20 and <=20 contributions are disjoint per ion, so squared sums subtract too.
        total2 = sum(r.sums2["sqrt_flux_all"] - r.sums2["sqrt_flux_gt20"] for r in group)
        mean, se = mean_and_se(total, total2, n)
        points[u] = (mean, se, "production_pooled", n)
    for run in runs:
        if "recon" not in run.family or run.thickness_nm in points:
            continue
        total = run.sums["sqrt_flux_all"] - run.sums["sqrt_flux_gt20"]
        total2 = run.sums2["sqrt_flux_all"] - run.sums2["sqrt_flux_gt20"]
        mean, se = mean_and_se(total, total2, run.n_incident)
        points[run.thickness_nm] = (mean, se, "recon", run.n_incident)
    rows = [
        {
            "u_nm": u,
            "sqrt_flux_le20": values[0],
            "sqrt_flux_le20_se": values[1],
            "source": values[2],
            "n_incident": values[3],
        }
        for u, values in sorted(points.items())
    ]
    return (
        np.array([float(r["u_nm"]) for r in rows]),
        np.array([float(r["sqrt_flux_le20"]) for r in rows]),
        rows,
    )


def load_fit_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, y, sy = [], [], []
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["thickness_nm"]))
            y.append(float(row["normalized_median"]))
            sy.append(
                0.5
                * (float(row["normalized_err_lower_68"]) + float(row["normalized_err_upper_68"]))
            )
    return np.array(t), np.array(y), np.array(sy)


def make_model(bu: np.ndarray, bb: np.ndarray):
    interp = PchipInterpolator(bu, bb, extrapolate=False)

    def bfunc(u):
        u = np.asarray(u, dtype=float)
        return np.nan_to_num(interp(np.clip(u, bu[0], bu[-1])), nan=0.0)

    def model(t, p):
        cf, cb, lb, s = np.asarray(p, dtype=float)
        u = np.asarray(t, dtype=float) / s
        front = cf * (1.0 - np.exp(-u / 2.0))
        back = cb * bfunc(u) * (1.0 - np.exp(-u / lb))
        return front + back

    return bfunc, model


def multistart_fit(model, t, y, sigma, rng_seed=12345):
    lower = np.array([0.0, 0.0, 1.0, 0.35])
    upper = np.array([1.0, 4.0, 100.0, 1.30])
    starts = [
        np.array([0.326, 1.09, 15.5, 0.702]),
        np.array([0.25, 1.0, 10.0, 0.65]),
        np.array([0.4, 1.5, 25.0, 0.8]),
    ]
    rng = np.random.default_rng(rng_seed)
    for _ in range(80):
        starts.append(
            np.array([
                rng.uniform(0.15, 0.55),
                rng.uniform(0.3, 2.5),
                math.exp(rng.uniform(math.log(2.0), math.log(60.0))),
                rng.uniform(0.45, 1.05),
            ])
        )
    best = None
    for p0 in starts:
        fit = least_squares(
            lambda p: (model(t, p) - y) / sigma,
            x0=np.clip(p0, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            method="trf",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=20_000,
        )
        chi2 = float(np.sum(fit.fun * fit.fun))
        if best is None or chi2 < best[0]:
            best = (chi2, fit)
    assert best is not None
    return best


def fit_with_frozen_effective_variance(label: str, bu: np.ndarray, bb: np.ndarray) -> dict[str, object]:
    bfunc, model = make_model(bu, bb)
    t, y, sy = load_fit_data()
    pre_chi2, pre = multistart_fit(model, t, y, sy, rng_seed=10)
    h = np.maximum(1e-3, 1e-4 * t)
    slope = (model(t + h, pre.x) - model(t - h, pre.x)) / (2.0 * h)
    sigma_eff = np.sqrt(sy * sy + (slope * 0.10 * t) ** 2)
    chi2, fit = multistart_fit(model, t, y, sigma_eff, rng_seed=20)
    dof = len(t) - len(fit.x)

    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * (chi2 / dof)
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)
        perr = np.full(4, np.nan)

    cf, cb, lb, s = fit.x
    product = lb * s
    grad = np.array([0.0, 0.0, s, lb])
    product_se = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
    tt = np.linspace(0.01, 230.0, 230_000)
    yy = model(tt, fit.x)
    peak_t = float(tt[int(np.argmax(yy))])
    peak_y = float(np.max(yy))
    udata = t / s
    gaussian_deviance = float(
        chi2 + np.sum(np.log(2.0 * math.pi * sigma_eff * sigma_eff))
    )

    return {
        "label": label,
        "procedure": "WLS; symmetrized 68% y errors; sigma_eff frozen from y-only preliminary slopes; sigma_t=0.10*t",
        "n_points": len(t),
        "parameters": {"C_f": cf, "C_b": cb, "lambda_b_raw_nm": lb, "s": s},
        "parameter_se_local_covariance": {
            "C_f": perr[0], "C_b": perr[1], "lambda_b_raw_nm": perr[2], "s": perr[3]
        },
        "parameter_covariance": cov.tolist(),
        "lambda_b_times_s_nm": product,
        "lambda_b_times_s_se_local_covariance_nm": product_se,
        "preliminary_y_only": {"parameters": pre.x.tolist(), "chi2": pre_chi2},
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "gaussian_deviance_with_frozen_sigma_eff": gaussian_deviance,
        "aic_with_frozen_sigma_eff": gaussian_deviance + 2.0 * len(fit.x),
        "bic_with_frozen_sigma_eff": gaussian_deviance + math.log(len(t)) * len(fit.x),
        "model_peak_t_nm": peak_t,
        "model_peak_intensity": peak_y,
        "max_abs_frozen_slope": float(np.max(np.abs(slope))),
        "sigma_eff": sigma_eff.tolist(),
        "u_data_min_nm": float(np.min(udata)),
        "u_data_max_nm": float(np.max(udata)),
        "b_at_fit_data": bfunc(udata).tolist(),
    }


def route_multistart_fit(model, t, y, sigma, rng_seed=54321):
    lower = np.array([0.0, 0.0, 0.35])
    upper = np.array([1.5, 10.0, 1.30])
    starts = [
        np.array([0.25, 2.5, 0.70]),
        np.array([0.33, 2.0, 0.60]),
        np.array([0.20, 3.0, 0.85]),
    ]
    rng = np.random.default_rng(rng_seed)
    for _ in range(80):
        starts.append(
            np.array([
                rng.uniform(0.05, 0.70),
                math.exp(rng.uniform(math.log(0.2), math.log(7.0))),
                rng.uniform(0.40, 1.10),
            ])
        )
    best = None
    for p0 in starts:
        fit = least_squares(
            lambda p: (model(t, p) - y) / sigma,
            x0=np.clip(p0, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            method="trf",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=20_000,
        )
        chi2 = float(np.sum(fit.fun * fit.fun))
        if best is None or chi2 < best[0]:
            best = (chi2, fit)
    assert best is not None
    return best


def fit_route_only_proxy(
    label: str, qu: np.ndarray, qq: np.ndarray, sigma_override: np.ndarray | None = None
) -> dict[str, object]:
    """Fit I(t)=C0+A*Q(t/s), with C0 and A constrained nonnegative."""
    interp = PchipInterpolator(qu, qq, extrapolate=False)

    def qfunc(u):
        u = np.asarray(u, dtype=float)
        return np.nan_to_num(interp(np.clip(u, qu[0], qu[-1])), nan=0.0)

    def model(t, p):
        c0, amp, s = np.asarray(p, dtype=float)
        return c0 + amp * qfunc(np.asarray(t, dtype=float) / s)

    t, y, sy = load_fit_data()
    pre_chi2, pre = route_multistart_fit(model, t, y, sy, rng_seed=30)
    h = np.maximum(1e-3, 1e-4 * t)
    slope = (model(t + h, pre.x) - model(t - h, pre.x)) / (2.0 * h)
    sigma_eff_own = np.sqrt(sy * sy + (slope * 0.10 * t) ** 2)
    sigma_eff = np.asarray(sigma_override, dtype=float) if sigma_override is not None else sigma_eff_own
    chi2, fit = route_multistart_fit(model, t, y, sigma_eff, rng_seed=40)
    n, k = len(t), len(fit.x)
    dof = n - k
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * (chi2 / dof)
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
        perr = np.full(k, np.nan)
    tt = np.linspace(0.01, 230.0, 230_000)
    yy = model(tt, fit.x)
    peak_t = float(tt[int(np.argmax(yy))])
    gaussian_deviance = float(
        chi2 + np.sum(np.log(2.0 * math.pi * sigma_eff * sigma_eff))
    )
    return {
        "label": label,
        "model": "I(t)=C0+A*Q(t/s), C0>=0, A>=0",
        "procedure": (
            "WLS with externally supplied common sigma_eff"
            if sigma_override is not None
            else "WLS; symmetrized 68% y errors; sigma_eff frozen from y-only preliminary slopes; sigma_t=0.10*t"
        ),
        "n_points": n,
        "parameters": {"C0": fit.x[0], "A": fit.x[1], "s": fit.x[2]},
        "parameter_se_local_covariance": {"C0": perr[0], "A": perr[1], "s": perr[2]},
        "parameter_covariance": cov.tolist(),
        "preliminary_y_only": {"parameters": pre.x.tolist(), "chi2": pre_chi2},
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "sigma_eff": sigma_eff.tolist(),
        "own_preliminary_slope_sigma_eff": sigma_eff_own.tolist(),
        "gaussian_deviance_with_frozen_sigma_eff": gaussian_deviance,
        "aic_with_frozen_sigma_eff": gaussian_deviance + 2.0 * k,
        "bic_with_frozen_sigma_eff": gaussian_deviance + math.log(n) * k,
        "model_peak_t_nm": peak_t,
        "model_peak_intensity": float(np.max(yy)),
        "u_data_min_nm": float(np.min(t / fit.x[2])),
        "u_data_max_nm": float(np.max(t / fit.x[2])),
        "q_at_fit_data": qfunc(t / fit.x[2]).tolist(),
    }


def fit_route_mixture_proxy(
    label: str,
    qi_u: np.ndarray,
    qi: np.ndarray,
    qii_u: np.ndarray,
    qii: np.ndarray,
    *,
    s_fixed: float | None = None,
    s_prior: tuple[float, float] | None = None,
    sigma_override: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit C0 + Ai*Q>20(t/s) + Aii*Q<=20(t/s), nonnegative amplitudes."""
    qii_interp = PchipInterpolator(qi_u, qi, extrapolate=False)
    qdir_interp = PchipInterpolator(qii_u, qii, extrapolate=False)

    def eval_curves(t, s):
        u = np.asarray(t, dtype=float) / s
        vi = np.nan_to_num(qii_interp(np.clip(u, qi_u[0], qi_u[-1])), nan=0.0)
        vii = np.nan_to_num(qdir_interp(np.clip(u, qii_u[0], qii_u[-1])), nan=0.0)
        return vi, vii

    def model(t, p):
        if s_fixed is None:
            c0, ai, aii, s = np.asarray(p, dtype=float)
        else:
            c0, ai, aii = np.asarray(p, dtype=float)
            s = s_fixed
        vi, vii = eval_curves(t, s)
        return c0 + ai * vi + aii * vii

    if s_fixed is None:
        lower = np.array([0.0, 0.0, 0.0, 0.35])
        upper = np.array([1.5, 10.0, 10.0, 1.30])
        base_starts = [
            np.array([0.33, 1.8, 0.75, 0.66]),
            np.array([0.25, 2.5, 0.5, 0.70]),
            np.array([0.40, 1.0, 1.0, 0.80]),
        ]
    else:
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.5, 10.0, 10.0])
        base_starts = [
            np.array([0.33, 1.8, 0.75]),
            np.array([0.25, 2.5, 0.5]),
            np.array([0.40, 1.0, 1.0]),
        ]

    t, y, sy = load_fit_data()

    def optimize(sigma, seed, include_prior):
        rng = np.random.default_rng(seed)
        starts = list(base_starts)
        for _ in range(100):
            common = [
                rng.uniform(0.05, 0.70),
                math.exp(rng.uniform(math.log(0.1), math.log(7.0))),
                math.exp(rng.uniform(math.log(0.1), math.log(4.0))),
            ]
            if s_fixed is None:
                common.append(rng.uniform(0.40, 1.10))
            starts.append(np.array(common))

        def residual(p):
            out = list((model(t, p) - y) / sigma)
            if include_prior and s_prior is not None and s_fixed is None:
                out.append((p[3] - s_prior[0]) / s_prior[1])
            return np.asarray(out)

        best = None
        for p0 in starts:
            fit = least_squares(
                residual,
                x0=np.clip(p0, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                method="trf",
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=20_000,
            )
            objective = float(np.sum(fit.fun * fit.fun))
            if best is None or objective < best[0]:
                best = (objective, fit)
        assert best is not None
        return best

    pre_objective, pre = optimize(sy, 50, include_prior=True)
    h = np.maximum(1e-3, 1e-4 * t)
    slope = (model(t + h, pre.x) - model(t - h, pre.x)) / (2.0 * h)
    sigma_eff_own = np.sqrt(sy * sy + (slope * 0.10 * t) ** 2)
    sigma_eff = np.asarray(sigma_override, dtype=float) if sigma_override is not None else sigma_eff_own
    objective, fit = optimize(sigma_eff, 60, include_prior=True)
    data_resid = (model(t, fit.x) - y) / sigma_eff
    data_chi2 = float(np.sum(data_resid * data_resid))
    prior_chi2 = 0.0
    if s_prior is not None and s_fixed is None:
        prior_chi2 = float(((fit.x[3] - s_prior[0]) / s_prior[1]) ** 2)
    n, k = len(t), len(fit.x)
    dof_data = n - k
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * (objective / max(len(fit.fun) - k, 1))
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
        perr = np.full(k, np.nan)
    if s_fixed is None:
        c0, ai, aii, s = fit.x
        pnames = ("C0", "A_i_gt20", "A_ii_le20", "s")
    else:
        c0, ai, aii = fit.x
        s = s_fixed
        pnames = ("C0", "A_i_gt20", "A_ii_le20")
    tt = np.linspace(0.01, 230.0, 230_000)
    yy = model(tt, fit.x)
    gaussian_deviance = float(
        data_chi2 + np.sum(np.log(2.0 * math.pi * sigma_eff * sigma_eff))
    )
    return {
        "label": label,
        "model": "I(t)=C0+Ai*Q_sqrtE,>20(t/s)+Aii*Q_sqrtE,<=20(t/s); amplitudes nonnegative",
        "s_fixed": s_fixed,
        "s_gaussian_constraint": list(s_prior) if s_prior is not None else None,
        "parameters": {name: float(value) for name, value in zip(pnames, fit.x)},
        "parameter_se_local_covariance": {name: float(value) for name, value in zip(pnames, perr)},
        "parameter_covariance": cov.tolist(),
        "preliminary_y_only_objective": pre_objective,
        "preliminary_y_only_parameters": pre.x.tolist(),
        "data_chi2": data_chi2,
        "prior_chi2": prior_chi2,
        "total_objective": objective,
        "data_dof": dof_data,
        "reduced_data_chi2": data_chi2 / dof_data,
        "gaussian_deviance_with_frozen_sigma_eff": gaussian_deviance,
        "aic_with_frozen_sigma_eff": gaussian_deviance + 2.0 * k,
        "bic_with_frozen_sigma_eff": gaussian_deviance + math.log(n) * k,
        "sigma_eff": sigma_eff.tolist(),
        "own_preliminary_slope_sigma_eff": sigma_eff_own.tolist(),
        "model_peak_t_nm": float(tt[int(np.argmax(yy))]),
        "model_peak_intensity": float(np.max(yy)),
        "effective_s": float(s),
        "amplitude_ratio_Ai_over_Aii": float(ai / aii) if aii > 0 else math.inf,
        "caveat": "Fixing or constraining s with a Model-B fit to the same intensity data is a sensitivity analysis, not independent evidence; an external calibration would be required for a valid prior.",
    }


def mapped_peak(u_result: dict[str, object]) -> dict[str, float]:
    u = float(u_result["vertex_mc_median_u_nm"])
    su_lo = u - float(u_result["vertex_mc_16_u_nm"])
    su_hi = float(u_result["vertex_mc_84_u_nm"]) - u
    su = 0.5 * (su_lo + su_hi)
    return {
        "u_nm": u,
        "u_mc_stat_sd_approx_nm": su,
        "t_at_s_frozen_nm": S_FROZEN * u,
        "t_stat_plus_s_fit_sd_nm": math.sqrt((S_FROZEN * su) ** 2 + (u * S_FIT_SD) ** 2),
        "additional_t_repeatability_sd_nm": u * S_REPEAT_SD,
    }


def main() -> None:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(ARCHIVE)
    runs = discover_and_parse_runs()
    run_rows = [run_to_row(run) for run in runs]
    write_csv(HERE / "per_run_metrics.csv", run_rows)

    pooled = production_pooled(runs)
    write_csv(HERE / "production_pooled_metrics.csv", pooled)

    new_u, new_b, new_b_rows = build_new_b_table(runs, pooled)
    write_csv(HERE / "new_b4_table.csv", new_b_rows)
    q20_u, q20, q20_rows = build_metric_table(
        runs, pooled, "sqrt_flux_gt20", value_at_zero=0.0
    )
    f20_u, f20, f20_rows = build_metric_table(
        runs, pooled, "flux_gt20", value_at_zero=0.0
    )
    qle20_u, qle20, qle20_rows = build_sqrt_le20_table(runs, pooled)
    b_uncap_u, b_uncap, b_uncap_rows = build_metric_table(
        runs, pooled, "b_uncapped", value_at_zero=1.0
    )
    write_csv(HERE / "route_i_sqrtE_gt20_table.csv", q20_rows)
    write_csv(HERE / "route_i_unweighted_gt20_table.csv", f20_rows)
    write_csv(HERE / "route_ii_sqrtE_le20_table.csv", qle20_rows)
    write_csv(HERE / "new_b_uncapped_table.csv", b_uncap_rows)
    new_interp = PchipInterpolator(new_u, new_b, extrapolate=False)
    comparison_rows = []
    for u, old in zip(FROZEN_U, FROZEN_B):
        new = float(new_interp(u))
        comparison_rows.append(
            {
                "u_nm": u,
                "frozen_b4": old,
                "new_b4": new,
                "new_minus_frozen": new - old,
                "relative_difference_percent": (100.0 * (new - old) / old if old != 0 else math.nan),
            }
        )
    write_csv(HERE / "frozen_b4_comparison.csv", comparison_rows)

    route_unweighted = approximate_flat_interval(pooled, "flux_gt20")
    route_weighted = approximate_flat_interval(pooled, "sqrt_flux_gt20")
    model_full = approximate_flat_interval(pooled, "model_b_full")
    quad_unweighted = quadratic_peak(pooled, "flux_gt20", 90.0, 140.0, seed=101)
    quad_weighted = quadratic_peak(pooled, "sqrt_flux_gt20", 60.0, 105.0, seed=102)
    quad_model = quadratic_peak(pooled, "model_b_full", 45.0, 75.0, seed=103)

    old_fit = fit_with_frozen_effective_variance("frozen_letter_b4_table", FROZEN_U, FROZEN_B)
    new_fit = fit_with_frozen_effective_variance("new_campaign_b4_table", new_u, new_b)
    uncapped_fit = fit_with_frozen_effective_variance(
        "new_campaign_b_uncapped_table", b_uncap_u, b_uncap
    )
    route_q20_fit = fit_route_only_proxy("route_i_sqrtE_gt20", q20_u, q20)
    route_f20_fit = fit_route_only_proxy("route_i_unweighted_gt20", f20_u, f20)
    mixture_free = fit_route_mixture_proxy(
        "route_i_plus_ii_free_s", q20_u, q20, qle20_u, qle20
    )
    mixture_fixed = fit_route_mixture_proxy(
        "route_i_plus_ii_s_fixed_to_new_Model_B",
        q20_u,
        q20,
        qle20_u,
        qle20,
        s_fixed=float(new_fit["parameters"]["s"]),
    )
    mixture_constrained = fit_route_mixture_proxy(
        "route_i_plus_ii_s_constrained_by_new_Model_B_local_covariance",
        q20_u,
        q20,
        qle20_u,
        qle20,
        s_prior=(
            float(new_fit["parameters"]["s"]),
            float(new_fit["parameter_se_local_covariance"]["s"]),
        ),
    )
    common_sigma = np.asarray(new_fit["sigma_eff"], dtype=float)
    route_q20_common = fit_route_only_proxy(
        "route_i_sqrtE_gt20_common_Model_B_sigma", q20_u, q20, sigma_override=common_sigma
    )
    mixture_free_common = fit_route_mixture_proxy(
        "route_i_plus_ii_free_s_common_Model_B_sigma",
        q20_u,
        q20,
        qle20_u,
        qle20,
        sigma_override=common_sigma,
    )
    mixture_fixed_common = fit_route_mixture_proxy(
        "route_i_plus_ii_fixed_s_common_Model_B_sigma",
        q20_u,
        q20,
        qle20_u,
        qle20,
        s_fixed=float(new_fit["parameters"]["s"]),
        sigma_override=common_sigma,
    )

    production_runs = [r for r in runs if "production_100k" in r.family]
    seed_diffs = []
    for u in sorted({r.thickness_nm for r in production_runs}):
        pair = sorted([r for r in production_runs if r.thickness_nm == u], key=lambda r: r.seed)
        a, b = run_to_row(pair[0]), run_to_row(pair[1])
        row = {"u_nm": u}
        for key in ("transmission", "flux_gt20", "sqrt_flux_gt20", "b4", "model_b_full"):
            if key == "model_b_full":
                factor = 1.0 - math.exp(-u / LAMBDA_B_FROZEN)
                av = float(a["b4"]) * factor
                bv = float(b["b4"]) * factor
                ase = float(a["b4_se"]) * factor
                bse = float(b["b4_se"]) * factor
            else:
                av, bv = float(a[key]), float(b[key])
                ase, bse = float(a[f"{key}_se"]), float(b[f"{key}_se"])
            row[f"{key}_seed13579"] = av
            row[f"{key}_seed24680"] = bv
            row[f"{key}_difference"] = bv - av
            row[f"{key}_z_independent"] = (bv - av) / math.sqrt(ase * ase + bse * bse)
        seed_diffs.append(row)
    write_csv(HERE / "production_seed_comparison.csv", seed_diffs)

    cap_total = sum(r.cap_count for r in runs)
    trans_total = sum(r.n_transmitted for r in runs)
    n_total = sum(r.n_incident for r in runs)
    new_prod_interp_at_grid = np.array([float(new_interp(float(r["u_nm"]))) for r in pooled])
    old_interp = PchipInterpolator(FROZEN_U, FROZEN_B, extrapolate=False)
    old_at_prod = old_interp(np.array([float(r["u_nm"]) for r in pooled]))
    relative = (new_prod_interp_at_grid - old_at_prod) / old_at_prod

    result = {
        "archive": str(ARCHIVE),
        "integrity": {
            "runs": len(runs),
            "incident_ions": n_total,
            "transmitted_records": trans_total,
            "all_status_complete_zero_exit_no_warnings": True,
            "max_direction_cosine_norm_error": max(r.max_direction_norm_error for r in runs),
            "max_exit_energy_eV": max(r.max_energy_eV for r in runs),
            "sec_cap_affected_records": cap_total,
            "sec_cap_fraction_transmitted": cap_total / trans_total,
            "sec_cap_fraction_incident": cap_total / n_total,
            "shared_prefix_audit": shared_prefix_audit(runs),
        },
        "definitions": {
            "transmission": "N_transmitted/N_incident",
            "flux_gt20": "sum[theta_exit>20 deg]/N_incident",
            "sqrt_flux_gt20": "sum[theta_exit>20 deg]*sqrt(E_exit/E0)/N_incident",
            "b4": "sum_transmitted sqrt(E_exit/E0)*min(sec(theta_exit),4)/N_incident",
            "model_b_full": "b4*(1-exp(-u/15.53 nm))",
        },
        "route_i_unweighted": {
            **route_unweighted,
            "local_quadratic": quad_unweighted,
            "individual_seed_grid_peaks": seed_peak_summary(runs, "flux_gt20"),
            "mapped_local_quadratic": mapped_peak(quad_unweighted),
        },
        "route_i_sqrtE_weighted": {
            **route_weighted,
            "local_quadratic": quad_weighted,
            "individual_seed_grid_peaks": seed_peak_summary(runs, "sqrt_flux_gt20"),
            "mapped_local_quadratic": mapped_peak(quad_weighted),
        },
        "model_b_full_frozen_parameters": {
            **model_full,
            "local_quadratic": quad_model,
            "individual_seed_grid_peaks": seed_peak_summary_full(runs),
            "mapped_local_quadratic": mapped_peak(quad_model),
        },
        "b4_comparison": {
            "max_abs_relative_difference_production_grid_40_320_percent": float(
                100.0 * np.max(np.abs(relative))
            ),
            "median_abs_relative_difference_production_grid_40_320_percent": float(
                100.0 * np.median(np.abs(relative))
            ),
            "max_abs_difference_production_grid_40_320": float(
                np.max(np.abs(new_prod_interp_at_grid - old_at_prod))
            ),
        },
        "fits": {
            "old_table_reproduction": old_fit,
            "new_campaign": new_fit,
            "new_campaign_uncapped_secant_sensitivity": uncapped_fit,
        },
        "route_i_only_proxy_fits": {
            "sqrtE_weighted_gt20": route_q20_fit,
            "unweighted_gt20": route_f20_fit,
            "sqrtE_weighted_gt20_common_Model_B_sigma": route_q20_common,
            "comparison_to_new_model_b": {
                "sqrtE_delta_chi2": route_q20_fit["chi2"] - new_fit["chi2"],
                "sqrtE_delta_aic_frozen_sigma_eff": route_q20_fit["aic_with_frozen_sigma_eff"] - new_fit["aic_with_frozen_sigma_eff"],
                "sqrtE_delta_bic_frozen_sigma_eff": route_q20_fit["bic_with_frozen_sigma_eff"] - new_fit["bic_with_frozen_sigma_eff"],
                "unweighted_delta_chi2": route_f20_fit["chi2"] - new_fit["chi2"],
                "unweighted_delta_aic_frozen_sigma_eff": route_f20_fit["aic_with_frozen_sigma_eff"] - new_fit["aic_with_frozen_sigma_eff"],
                "unweighted_delta_bic_frozen_sigma_eff": route_f20_fit["bic_with_frozen_sigma_eff"] - new_fit["bic_with_frozen_sigma_eff"],
                "caveat": "AIC/BIC use each model's own preliminary-slope-frozen sigma_eff; they are an approximate shape comparison because shared ROI normalization covariance is not available here.",
            },
        },
        "route_i_plus_ii_mixture_sensitivity": {
            "free_s": mixture_free,
            "s_fixed_to_new_Model_B": mixture_fixed,
            "s_gaussian_constrained_to_new_Model_B": mixture_constrained,
            "common_Model_B_sigma_eff": {
                "free_s": mixture_free_common,
                "s_fixed_to_new_Model_B": mixture_fixed_common,
                "route_i_only_sqrtE": route_q20_common,
                "free_s_delta_chi2_vs_Model_B": mixture_free_common["data_chi2"] - new_fit["chi2"],
                "fixed_s_delta_chi2_vs_Model_B": mixture_fixed_common["data_chi2"] - new_fit["chi2"],
                "route_i_only_delta_chi2_vs_Model_B": route_q20_common["chi2"] - new_fit["chi2"],
                "free_s_delta_AIC_vs_Model_B": (mixture_free_common["data_chi2"] - new_fit["chi2"]),
                "fixed_s_delta_AIC_vs_Model_B": (mixture_fixed_common["data_chi2"] - new_fit["chi2"] - 2.0),
                "route_i_only_delta_AIC_vs_Model_B": (route_q20_common["chi2"] - new_fit["chi2"] - 2.0),
            },
            "comparison_to_new_Model_B": {
                "free_s_delta_data_chi2": mixture_free["data_chi2"] - new_fit["chi2"],
                "fixed_s_delta_data_chi2": mixture_fixed["data_chi2"] - new_fit["chi2"],
                "constrained_s_delta_data_chi2": mixture_constrained["data_chi2"] - new_fit["chi2"],
                "free_s_delta_aic_frozen_sigma_eff": mixture_free["aic_with_frozen_sigma_eff"] - new_fit["aic_with_frozen_sigma_eff"],
                "fixed_s_delta_aic_frozen_sigma_eff": mixture_fixed["aic_with_frozen_sigma_eff"] - new_fit["aic_with_frozen_sigma_eff"],
                "constrained_s_delta_aic_frozen_sigma_eff": mixture_constrained["aic_with_frozen_sigma_eff"] - new_fit["aic_with_frozen_sigma_eff"],
            },
        },
        "fit_changes_new_minus_old": {
            "C_f": new_fit["parameters"]["C_f"] - old_fit["parameters"]["C_f"],
            "C_b": new_fit["parameters"]["C_b"] - old_fit["parameters"]["C_b"],
            "lambda_b_raw_nm": new_fit["parameters"]["lambda_b_raw_nm"] - old_fit["parameters"]["lambda_b_raw_nm"],
            "s": new_fit["parameters"]["s"] - old_fit["parameters"]["s"],
            "lambda_b_times_s_nm": new_fit["lambda_b_times_s_nm"] - old_fit["lambda_b_times_s_nm"],
            "model_peak_t_nm": new_fit["model_peak_t_nm"] - old_fit["model_peak_t_nm"],
            "chi2": new_fit["chi2"] - old_fit["chi2"],
        },
        "secant_cap_sensitivity_uncapped_minus_cap4": {
            "C_f": uncapped_fit["parameters"]["C_f"] - new_fit["parameters"]["C_f"],
            "C_b": uncapped_fit["parameters"]["C_b"] - new_fit["parameters"]["C_b"],
            "lambda_b_raw_nm": uncapped_fit["parameters"]["lambda_b_raw_nm"] - new_fit["parameters"]["lambda_b_raw_nm"],
            "s": uncapped_fit["parameters"]["s"] - new_fit["parameters"]["s"],
            "lambda_b_times_s_nm": uncapped_fit["lambda_b_times_s_nm"] - new_fit["lambda_b_times_s_nm"],
            "model_peak_t_nm": uncapped_fit["model_peak_t_nm"] - new_fit["model_peak_t_nm"],
            "chi2": uncapped_fit["chi2"] - new_fit["chi2"],
        },
        "fixed_landmark_mapping": {
            str(u): {
                "t_nm": S_FROZEN * u,
                "s_fit_component_sd_nm": S_FIT_SD * u,
                "additional_repeatability_component_sd_nm": S_REPEAT_SD * u,
            }
            for u in (80.0, 125.0, 255.0)
        },
    }
    (HERE / "analysis_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def seed_peak_summary_full(runs: list[RunStats]) -> dict[str, object]:
    out: dict[str, object] = {}
    for seed in (13579, 24680):
        candidate = [r for r in runs if "production_100k" in r.family and r.seed == seed]
        values = []
        for r in candidate:
            b4, _ = mean_and_se(r.sums["b4"], r.sums2["b4"], r.n_incident)
            values.append(b4 * (1.0 - math.exp(-r.thickness_nm / LAMBDA_B_FROZEN)))
        imax = int(np.argmax(values))
        out[str(seed)] = {"u_nm": candidate[imax].thickness_nm, "value": values[imax]}
    return out


if __name__ == "__main__":
    main()

"""Reanalyse the completed SRIM campaign at a 14.2 degree polar-angle split.

The original audit used 20 degrees as the operational split between the
large-angle external route and the near-axis route.  This script streams the
same immutable TRANSMIT.txt records and repeats the analysis at the measured
14.2 degree chamber-acceptance half-angle.

Outputs are deliberately confined to this analysis directory.  The SRIM
archive and manuscript files are never modified.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear

import analyze_archive as base


HERE = Path(__file__).resolve().parent
ANGLE_CUT_DEG = 14.2
COS_CUT = math.cos(math.radians(ANGLE_CUT_DEG))
E0_EV = 30_000.0
SEC_CAP = 4.0

METRICS = (
    "transmission",
    "flux_gt14p2",
    "sqrtE_all",
    "sqrtE_gt14p2",
    "sqrtE_sec_gt14p2",
    "sqrtE_sec_cap4_gt14p2",
    "sqrtE_csc_gt14p2",
)


def empty_metric_dict() -> dict[str, float]:
    return {key: 0.0 for key in METRICS}


@dataclass
class ThresholdRun:
    index: int
    name: str
    family: str
    seed: int
    thickness_nm: float
    n_incident: int
    transmit_path: Path
    n_transmitted: int = 0
    n_gt_cut: int = 0
    sums: dict[str, float] = field(default_factory=empty_metric_dict)
    sums2: dict[str, float] = field(default_factory=empty_metric_dict)
    exact_boundary_records: int = 0
    closest_angle_delta_deg: float = math.inf


def record_metrics(energy_eV: float, cos_theta: float) -> dict[str, float]:
    sqrt_e = math.sqrt(max(energy_eV, 0.0) / E0_EV)
    is_external = cos_theta < COS_CUT
    values = {
        "transmission": 1.0,
        "flux_gt14p2": 0.0,
        "sqrtE_all": sqrt_e,
        "sqrtE_gt14p2": 0.0,
        "sqrtE_sec_gt14p2": 0.0,
        "sqrtE_sec_cap4_gt14p2": 0.0,
        "sqrtE_csc_gt14p2": 0.0,
    }
    if not is_external:
        return values

    sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    sec_theta = 1.0 / cos_theta
    csc_theta = 1.0 / sin_theta
    values.update(
        flux_gt14p2=1.0,
        sqrtE_gt14p2=sqrt_e,
        sqrtE_sec_gt14p2=sqrt_e * sec_theta,
        sqrtE_sec_cap4_gt14p2=sqrt_e * min(sec_theta, SEC_CAP),
        sqrtE_csc_gt14p2=sqrt_e * csc_theta,
    )
    return values


def parse_transmit(run: ThresholdRun) -> None:
    with run.transmit_path.open("r", encoding="latin-1", errors="strict") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.startswith("T"):
                continue
            parts = line[1:].split()
            if len(parts) != 9:
                raise ValueError(
                    f"Unexpected TRANSMIT record at {run.transmit_path}:{line_no}"
                )
            atom_z = int(parts[1])
            energy_eV = float(parts[2])
            cos_theta = float(parts[6])
            if atom_z != 2:
                raise ValueError(f"Non-He transmitted atom Z={atom_z} in {run.name}")
            if not (0.0 < cos_theta <= 1.000001):
                raise ValueError(f"Invalid forward direction cosine {cos_theta} in {run.name}")
            if not (0.0 <= energy_eV <= E0_EV * 1.000001):
                raise ValueError(f"Invalid exit energy {energy_eV} eV in {run.name}")

            angle_deg = math.degrees(math.acos(min(max(cos_theta, -1.0), 1.0)))
            delta = abs(angle_deg - ANGLE_CUT_DEG)
            run.closest_angle_delta_deg = min(run.closest_angle_delta_deg, delta)
            if delta <= 1e-12:
                run.exact_boundary_records += 1

            values = record_metrics(energy_eV, cos_theta)
            run.n_transmitted += 1
            if values["flux_gt14p2"]:
                run.n_gt_cut += 1
            for key, value in values.items():
                run.sums[key] += value
                run.sums2[key] += value * value

    if run.n_transmitted != int(run.sums["transmission"]):
        raise AssertionError(f"Transmission-count mismatch in {run.name}")
    if run.n_gt_cut != int(run.sums["flux_gt14p2"]):
        raise AssertionError(f"Threshold-count mismatch in {run.name}")


def discover_and_parse() -> list[ThresholdRun]:
    status_paths = sorted((base.ARCHIVE / "MoS2").rglob("status.json"))
    if len(status_paths) != 118:
        raise RuntimeError(f"Expected 118 status files; found {len(status_paths)}")

    runs: list[ThresholdRun] = []
    for number, status_path in enumerate(status_paths, start=1):
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
        if status.get("state") != "completed" or status.get("trim_exit_code") != 0:
            raise RuntimeError(f"Incomplete or failed run: {status_path}")
        if status.get("warnings"):
            raise RuntimeError(f"Runner warning in {status_path}: {status['warnings']}")
        params = status["parameters"]
        transmit_path = (
            status_path.parent / "artifacts" / "SRIM Outputs" / "TRANSMIT.txt"
        )
        if not transmit_path.is_file():
            raise FileNotFoundError(transmit_path)
        run = ThresholdRun(
            index=int(status["index"]),
            name=status["name"],
            family=params["campaign_family"],
            seed=int(params["random_seed"]),
            thickness_nm=float(params["layer_1_width_nm"]),
            n_incident=int(params["ion_count"]),
            transmit_path=transmit_path,
        )
        parse_transmit(run)
        runs.append(run)
        if number % 10 == 0 or number == len(status_paths):
            print(f"Parsed {number:3d}/{len(status_paths)} TRANSMIT files", flush=True)
    return sorted(runs, key=lambda item: item.index)


def totals_to_curve_row(
    group: list[ThresholdRun], source: str, thickness_nm: float
) -> dict[str, object]:
    n_incident = sum(run.n_incident for run in group)
    row: dict[str, object] = {
        "u_nm": thickness_nm,
        "source": source,
        "seeds": ";".join(str(run.seed) for run in group),
        "n_incident": n_incident,
        "n_transmitted": sum(run.n_transmitted for run in group),
    }
    for metric in METRICS:
        total = sum(run.sums[metric] for run in group)
        total2 = sum(run.sums2[metric] for run in group)
        mean, se = base.mean_and_se(total, total2, n_incident)
        row[metric] = mean
        row[f"{metric}_se"] = se

    # The two partitions are disjoint for every incident ion, so their sums of
    # squares also partition exactly.
    for complement, total_metric, gt_metric in (
        ("flux_le14p2", "transmission", "flux_gt14p2"),
        ("sqrtE_le14p2", "sqrtE_all", "sqrtE_gt14p2"),
    ):
        total = sum(
            run.sums[total_metric] - run.sums[gt_metric] for run in group
        )
        total2 = sum(
            run.sums2[total_metric] - run.sums2[gt_metric] for run in group
        )
        mean, se = base.mean_and_se(total, total2, n_incident)
        row[complement] = mean
        row[f"{complement}_se"] = se
    return row


def build_continuous_curve(runs: list[ThresholdRun]) -> list[dict[str, object]]:
    rows: dict[float, dict[str, object]] = {
        0.0: {
            "u_nm": 0.0,
            "source": "analytic",
            "seeds": "",
            "n_incident": 0,
            "n_transmitted": 0,
            "transmission": 1.0,
            "transmission_se": 0.0,
            "flux_gt14p2": 0.0,
            "flux_gt14p2_se": 0.0,
            "flux_le14p2": 1.0,
            "flux_le14p2_se": 0.0,
            "sqrtE_all": 1.0,
            "sqrtE_all_se": 0.0,
            "sqrtE_gt14p2": 0.0,
            "sqrtE_gt14p2_se": 0.0,
            "sqrtE_le14p2": 1.0,
            "sqrtE_le14p2_se": 0.0,
            "sqrtE_sec_gt14p2": 0.0,
            "sqrtE_sec_gt14p2_se": 0.0,
            "sqrtE_sec_cap4_gt14p2": 0.0,
            "sqrtE_sec_cap4_gt14p2_se": 0.0,
            "sqrtE_csc_gt14p2": 0.0,
            "sqrtE_csc_gt14p2_se": 0.0,
        }
    }

    production = [run for run in runs if "production_100k" in run.family]
    production_by_u: dict[float, list[ThresholdRun]] = {}
    for run in production:
        production_by_u.setdefault(run.thickness_nm, []).append(run)
    if len(production_by_u) != 41 or any(
        len(group) != 2 for group in production_by_u.values()
    ):
        raise RuntimeError("Production campaign is not 41 thicknesses x 2 seeds")
    for u_nm, group in production_by_u.items():
        rows[u_nm] = totals_to_curve_row(group, "production_pooled", u_nm)

    for run in runs:
        if "recon" not in run.family or run.thickness_nm in rows:
            continue
        rows[run.thickness_nm] = totals_to_curve_row(
            [run], "recon", run.thickness_nm
        )
    return [rows[u_nm] for u_nm in sorted(rows)]


def centered_window(
    production_rows: list[dict[str, object]], metric: str, n_points: int
) -> tuple[float, float]:
    x = np.array([float(row["u_nm"]) for row in production_rows])
    y = np.array([float(row[metric]) for row in production_rows])
    imax = int(np.argmax(y))
    start = max(0, min(imax - n_points // 2, len(x) - n_points))
    selected = x[start : start + n_points]
    return float(selected[0]), float(selected[-1])


def safe_quadratic(
    production_rows: list[dict[str, object]],
    metric: str,
    lo: float,
    hi: float,
    seed: int,
) -> dict[str, object]:
    try:
        result = base.quadratic_peak(production_rows, metric, lo, hi, seed=seed)
        curvature = float(result["quadratic_coefficients"][0])
        result["concave"] = curvature < 0.0
        return result
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"window_u_nm": [lo, hi], "error": repr(exc), "concave": False}


def map_quadratic_peak(
    quadratic: dict[str, object], s: float, s_sd: float
) -> dict[str, float] | None:
    if "vertex_mc_median_u_nm" not in quadratic:
        return None
    u_nm = float(quadratic["vertex_mc_median_u_nm"])
    stat_sd = 0.5 * (
        float(quadratic["vertex_mc_84_u_nm"])
        - float(quadratic["vertex_mc_16_u_nm"])
    )
    return {
        "u_nm": u_nm,
        "u_MC_stat_sd_approx_nm": stat_sd,
        "t_at_new_s_nm": s * u_nm,
        "t_MC_stat_component_sd_nm": s * stat_sd,
        "t_new_s_local_covariance_component_sd_nm": u_nm * s_sd,
        "t_combined_MC_and_s_local_covariance_sd_nm": math.sqrt(
            (s * stat_sd) ** 2 + (u_nm * s_sd) ** 2
        ),
        "additional_repeatability_component_sd_nm": u_nm * base.S_REPEAT_SD,
    }


def peak_analysis(
    production_rows: list[dict[str, object]],
    metric: str,
    s: float,
    s_sd: float,
    *,
    primary_window: tuple[float, float] | None,
    seed: int,
) -> dict[str, object]:
    grid = base.approximate_flat_interval(production_rows, metric)
    if primary_window is None:
        primary_window = centered_window(production_rows, metric, 11)
    primary = safe_quadratic(
        production_rows, metric, primary_window[0], primary_window[1], seed
    )
    sensitivity: dict[str, object] = {}
    for offset, n_points in enumerate((7, 9, 11), start=1):
        lo, hi = centered_window(production_rows, metric, n_points)
        sensitivity[f"{n_points}_centered_points"] = safe_quadratic(
            production_rows, metric, lo, hi, seed + offset
        )
    grid_u = float(grid["grid_peak_u_nm"])
    return {
        **grid,
        "grid_peak_t_at_new_s_nm": s * grid_u,
        "approx_95pct_indistinguishable_t_at_new_s_nm": [
            s * float(grid["approx_95pct_indistinguishable_u_min_nm"]),
            s * float(grid["approx_95pct_indistinguishable_u_max_nm"]),
        ],
        "primary_local_quadratic": primary,
        "mapped_primary_local_quadratic": map_quadratic_peak(primary, s, s_sd),
        "centered_window_sensitivity": sensitivity,
    }


def load_driver_and_model_b() -> tuple[dict[str, object], np.ndarray]:
    rows = list(
        csv.DictReader(
            (HERE / "new_b4_table.csv").open(
                "r", encoding="utf-8-sig", newline=""
            )
        )
    )
    u_nm = np.array([float(row["u_nm"]) for row in rows])
    b4 = np.array([float(row["b4"]) for row in rows])
    fit = base.fit_with_frozen_effective_variance(
        "new_campaign_b4_table_reference_for_14p2", u_nm, b4
    )
    stored = json.loads((HERE / "analysis_results.json").read_text(encoding="utf-8"))
    stored_fit = stored["fits"]["new_campaign"]
    if abs(float(fit["parameters"]["s"]) - float(stored_fit["parameters"]["s"])) > 1e-10:
        raise AssertionError("Recomputed Model-B scale differs from stored audit")
    sigma = np.asarray(fit["sigma_eff"], dtype=float)
    if not np.allclose(
        sigma, np.asarray(stored_fit["sigma_eff"], dtype=float), rtol=0.0, atol=1e-12
    ):
        raise AssertionError("Recomputed Model-B effective variances differ")
    return fit, sigma


def curve_arrays(
    curve_rows: list[dict[str, object]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([float(row["u_nm"]) for row in curve_rows]),
        np.array([float(row[metric]) for row in curve_rows]),
    )


def model_comparison(
    chi2: float, n_parameters: int, model_b_fit: dict[str, object]
) -> dict[str, float]:
    model_b_chi2 = float(model_b_fit["chi2"])
    model_b_k = 4
    n_points = int(model_b_fit["n_points"])
    return {
        "delta_chi2_vs_Model_B": chi2 - model_b_chi2,
        "delta_AIC_vs_Model_B_common_sigma": (
            chi2 - model_b_chi2 + 2.0 * (n_parameters - model_b_k)
        ),
        "delta_BIC_vs_Model_B_common_sigma": (
            chi2
            - model_b_chi2
            + math.log(n_points) * (n_parameters - model_b_k)
        ),
    }


def fit_route_only(
    label: str,
    basis_description: str,
    u_nm: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    model_b_fit: dict[str, object],
) -> dict[str, object]:
    fit = base.fit_route_only_proxy(
        label, u_nm, values, sigma_override=sigma
    )
    fit["basis"] = basis_description
    fit["threshold_deg_from_membrane_normal"] = ANGLE_CUT_DEG
    fit["comparison_to_Model_B"] = model_comparison(
        float(fit["chi2"]), 3, model_b_fit
    )
    return fit


def fit_route_only_fixed_s(
    label: str,
    basis_description: str,
    u_nm: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    model_b_fit: dict[str, object],
    s_fixed: float,
) -> dict[str, object]:
    """Exact nonnegative weighted-linear fit of C0+A*Q(t/s_fixed)."""
    interpolator = PchipInterpolator(u_nm, values, extrapolate=False)
    t, y, _ = base.load_fit_data()
    q_at_data = np.nan_to_num(
        interpolator(np.clip(t / s_fixed, u_nm[0], u_nm[-1])), nan=0.0
    )
    design = np.column_stack([np.ones_like(t), q_at_data])
    weighted_design = design / sigma[:, None]
    weighted_y = y / sigma
    solution = lsq_linear(
        weighted_design,
        weighted_y,
        bounds=(np.zeros(2), np.full(2, np.inf)),
        method="trf",
        tol=1e-14,
        lsmr_tol="auto",
        max_iter=10_000,
    )
    if not solution.success:
        raise RuntimeError(f"Fixed-s route-only fit failed: {solution.message}")
    c0, amplitude = solution.x
    residual = (design @ solution.x - y) / sigma
    chi2 = float(residual @ residual)
    dof = len(t) - 2
    try:
        covariance = (
            np.linalg.inv(weighted_design.T @ weighted_design) * (chi2 / dof)
        )
        errors = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        covariance = np.full((2, 2), np.nan)
        errors = np.full(2, np.nan)

    tt = np.linspace(0.01, 230.0, 230_000)
    qq = np.nan_to_num(
        interpolator(np.clip(tt / s_fixed, u_nm[0], u_nm[-1])), nan=0.0
    )
    yy = c0 + amplitude * qq
    gaussian_deviance = float(
        chi2 + np.sum(np.log(2.0 * math.pi * sigma * sigma))
    )
    return {
        "label": label,
        "model": "I(t)=C0+A*Q(t/s_fixed), C0>=0, A>=0",
        "basis": basis_description,
        "threshold_deg_from_membrane_normal": ANGLE_CUT_DEG,
        "procedure": (
            "Exact nonnegative weighted-linear least squares with s fixed and "
            "the common frozen Model-B sigma_eff"
        ),
        "n_points": len(t),
        "parameters": {"C0": float(c0), "A": float(amplitude), "s_fixed": s_fixed},
        "parameter_se_local_covariance": {
            "C0": float(errors[0]),
            "A": float(errors[1]),
        },
        "parameter_covariance": covariance.tolist(),
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "sigma_eff": sigma.tolist(),
        "gaussian_deviance_with_frozen_sigma_eff": gaussian_deviance,
        "aic_with_frozen_sigma_eff": gaussian_deviance + 2.0 * 2,
        "bic_with_frozen_sigma_eff": gaussian_deviance + math.log(len(t)) * 2,
        "model_peak_t_nm": float(tt[int(np.argmax(yy))]),
        "model_peak_intensity": float(np.max(yy)),
        "q_at_fit_data": q_at_data.tolist(),
        "comparison_to_Model_B": model_comparison(chi2, 2, model_b_fit),
        "caveat": (
            "The fixed value of s comes from Model B fitted to the same "
            "intensity data and is a sensitivity test, not independent evidence."
        ),
    }


def fit_positive_mixture(
    label: str,
    external_basis: str,
    u_nm: np.ndarray,
    external_values: np.ndarray,
    direct_values: np.ndarray,
    sigma: np.ndarray,
    model_b_fit: dict[str, object],
    *,
    s_fixed: float | None,
) -> dict[str, object]:
    raw = base.fit_route_mixture_proxy(
        label,
        u_nm,
        external_values,
        u_nm,
        direct_values,
        s_fixed=s_fixed,
        sigma_override=sigma,
    )
    parameters = raw["parameters"]
    errors = raw["parameter_se_local_covariance"]
    ai = float(parameters["A_i_gt20"])
    aii = float(parameters["A_ii_le20"])
    renamed_parameters = {
        "C0": float(parameters["C0"]),
        "A_external_gt14p2": ai,
        "A_direct_le14p2": aii,
    }
    renamed_errors = {
        "C0": float(errors["C0"]),
        "A_external_gt14p2": float(errors["A_i_gt20"]),
        "A_direct_le14p2": float(errors["A_ii_le20"]),
    }
    if "s" in parameters:
        renamed_parameters["s"] = float(parameters["s"])
        renamed_errors["s"] = float(errors["s"])

    peak_t = float(raw["model_peak_t_nm"])
    effective_s = float(raw["effective_s"])
    peak_u = peak_t / effective_s
    ext_interp = PchipInterpolator(u_nm, external_values, extrapolate=False)
    dir_interp = PchipInterpolator(u_nm, direct_values, extrapolate=False)
    ext_basis_at_peak = float(ext_interp(np.clip(peak_u, u_nm[0], u_nm[-1])))
    dir_basis_at_peak = float(dir_interp(np.clip(peak_u, u_nm[0], u_nm[-1])))
    c0_component = float(parameters["C0"])
    external_component = ai * ext_basis_at_peak
    direct_component = aii * dir_basis_at_peak
    total = c0_component + external_component + direct_component
    above_baseline = external_component + direct_component

    raw["model"] = (
        "I(t)=C0+A_external*X_gt14p2(t/s)"
        "+A_direct*Q_sqrtE_le14p2(t/s); all amplitudes nonnegative"
    )
    raw["external_basis"] = external_basis
    raw["direct_basis"] = "sqrt(E_exit/E0) for theta<=14.2 deg"
    raw["threshold_deg_from_membrane_normal"] = ANGLE_CUT_DEG
    raw["parameters"] = renamed_parameters
    raw["parameter_se_local_covariance"] = renamed_errors
    raw["parameter_order_for_covariance"] = list(renamed_parameters)
    raw["amplitude_ratio_external_over_direct"] = ai / aii if aii > 0 else math.inf
    raw.pop("amplitude_ratio_Ai_over_Aii", None)
    raw["components_at_model_peak"] = {
        "t_peak_nm": peak_t,
        "u_peak_nm": peak_u,
        "C0_component": c0_component,
        "external_component": external_component,
        "direct_component": direct_component,
        "total": total,
        "baseline_fraction_of_total": c0_component / total,
        "external_fraction_of_total": external_component / total,
        "direct_fraction_of_total": direct_component / total,
        "external_fraction_of_above_baseline_signal": (
            external_component / above_baseline if above_baseline > 0 else math.nan
        ),
        "direct_fraction_of_above_baseline_signal": (
            direct_component / above_baseline if above_baseline > 0 else math.nan
        ),
        "warning": (
            "These are fitted basis-component contributions, not physical SE3 "
            "fractions; amplitudes absorb yield, interception, visibility, and "
            "collection efficiency."
        ),
    }
    n_parameters = 3 if s_fixed is not None else 4
    raw["comparison_to_Model_B"] = model_comparison(
        float(raw["data_chi2"]), n_parameters, model_b_fit
    )
    return raw


def population_at_experimental_thickness(
    curve_rows: list[dict[str, object]], s: float, t_nm: float
) -> dict[str, float]:
    u_nm = np.array([float(row["u_nm"]) for row in curve_rows])

    def interpolate(metric: str) -> float:
        values = np.array([float(row[metric]) for row in curve_rows])
        return float(PchipInterpolator(u_nm, values, extrapolate=False)(t_nm / s))

    transmission = interpolate("transmission")
    flux_gt = interpolate("flux_gt14p2")
    flux_le = interpolate("flux_le14p2")
    q_gt = interpolate("sqrtE_gt14p2")
    q_le = interpolate("sqrtE_le14p2")
    return {
        "experimental_t_nm": t_nm,
        "raw_SRIM_u_equals_t_over_s_nm": t_nm / s,
        "transmission_per_incident": transmission,
        "count_gt14p2_per_incident": flux_gt,
        "count_le14p2_per_incident": flux_le,
        "count_gt14p2_fraction_of_transmitted": flux_gt / transmission,
        "count_le14p2_fraction_of_transmitted": flux_le / transmission,
        "Q_sqrtE_gt14p2_per_incident": q_gt,
        "Q_sqrtE_le14p2_per_incident": q_le,
        "Q_gt14p2_over_Q_le14p2": q_gt / q_le,
        "Q_gt14p2_fraction_of_partitioned_Q": q_gt / (q_gt + q_le),
        "Q_le14p2_fraction_of_partitioned_Q": q_le / (q_gt + q_le),
    }


def comparison_with_20(
    results_20: dict[str, object],
    peaks_14: dict[str, object],
    route_14: dict[str, object],
    mix_free_14: dict[str, object],
    mix_fixed_14: dict[str, object],
    s_new: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    old_weighted = results_20["route_i_sqrtE_weighted"]["local_quadratic"]
    old_unweighted = results_20["route_i_unweighted"]["local_quadratic"]
    old_route = results_20["route_i_only_proxy_fits"][
        "sqrtE_weighted_gt20_common_Model_B_sigma"
    ]
    old_mix = results_20["route_i_plus_ii_mixture_sensitivity"][
        "common_Model_B_sigma_eff"
    ]
    old_free = old_mix["free_s"]
    old_fixed = old_mix["s_fixed_to_new_Model_B"]

    new_weighted_u = float(
        peaks_14["sqrtE"]["primary_local_quadratic"]["vertex_mc_median_u_nm"]
    )
    new_unweighted_u = float(
        peaks_14["unweighted"]["primary_local_quadratic"]["vertex_mc_median_u_nm"]
    )
    old_weighted_u = float(old_weighted["vertex_mc_median_u_nm"])
    old_unweighted_u = float(old_unweighted["vertex_mc_median_u_nm"])

    rows = [
        {
            "quantity": "sqrtE local peak raw u_nm",
            "cut_20deg": old_weighted_u,
            "cut_14p2deg": new_weighted_u,
            "change_14p2_minus_20": new_weighted_u - old_weighted_u,
        },
        {
            "quantity": "sqrtE local peak mapped t_nm at new s",
            "cut_20deg": s_new * old_weighted_u,
            "cut_14p2deg": s_new * new_weighted_u,
            "change_14p2_minus_20": s_new * (new_weighted_u - old_weighted_u),
        },
        {
            "quantity": "unweighted local peak raw u_nm",
            "cut_20deg": old_unweighted_u,
            "cut_14p2deg": new_unweighted_u,
            "change_14p2_minus_20": new_unweighted_u - old_unweighted_u,
        },
        {
            "quantity": "unweighted local peak mapped t_nm at new s",
            "cut_20deg": s_new * old_unweighted_u,
            "cut_14p2deg": s_new * new_unweighted_u,
            "change_14p2_minus_20": s_new * (new_unweighted_u - old_unweighted_u),
        },
        {
            "quantity": "sqrtE route-only common-sigma chi2",
            "cut_20deg": float(old_route["chi2"]),
            "cut_14p2deg": float(route_14["chi2"]),
            "change_14p2_minus_20": float(route_14["chi2"])
            - float(old_route["chi2"]),
        },
        {
            "quantity": "sqrtE route-only fitted s",
            "cut_20deg": float(old_route["parameters"]["s"]),
            "cut_14p2deg": float(route_14["parameters"]["s"]),
            "change_14p2_minus_20": float(route_14["parameters"]["s"])
            - float(old_route["parameters"]["s"]),
        },
        {
            "quantity": "sqrtE mixture free-s common-sigma chi2",
            "cut_20deg": float(old_free["data_chi2"]),
            "cut_14p2deg": float(mix_free_14["data_chi2"]),
            "change_14p2_minus_20": float(mix_free_14["data_chi2"])
            - float(old_free["data_chi2"]),
        },
        {
            "quantity": "sqrtE mixture free-s fitted s",
            "cut_20deg": float(old_free["effective_s"]),
            "cut_14p2deg": float(mix_free_14["effective_s"]),
            "change_14p2_minus_20": float(mix_free_14["effective_s"])
            - float(old_free["effective_s"]),
        },
        {
            "quantity": "sqrtE mixture fixed-s common-sigma chi2",
            "cut_20deg": float(old_fixed["data_chi2"]),
            "cut_14p2deg": float(mix_fixed_14["data_chi2"]),
            "change_14p2_minus_20": float(mix_fixed_14["data_chi2"])
            - float(old_fixed["data_chi2"]),
        },
    ]
    return {
        "same_Model_B_reference_s": s_new,
        "same_Model_B_reference_chi2": float(
            results_20["fits"]["new_campaign"]["chi2"]
        ),
        "table": rows,
        "interpretation": (
            "Lowering the split moves both route-only proxy peaks toward the "
            "observed 40-55 nm region and materially improves the route-only "
            "fit, but the common-error route-only model remains strongly "
            "disfavoured. Positive two-route mixtures remain competitive with "
            "or better than Model B."
        ),
    }, rows


def main() -> None:
    runs = discover_and_parse()
    curve_rows = build_continuous_curve(runs)
    base.write_csv(HERE / "threshold_14p2_pooled_curves.csv", curve_rows)

    production_rows = [
        row for row in curve_rows if row["source"] == "production_pooled"
    ]
    model_b_fit, common_sigma = load_driver_and_model_b()
    s_new = float(model_b_fit["parameters"]["s"])
    s_new_sd = float(model_b_fit["parameter_se_local_covariance"]["s"])

    peak_specs = {
        "unweighted": ("flux_gt14p2", (90.0, 140.0), 1401),
        "sqrtE": ("sqrtE_gt14p2", (60.0, 105.0), 1402),
        "horizontal_floor_sec": ("sqrtE_sec_gt14p2", None, 1403),
        "horizontal_floor_sec_cap4": ("sqrtE_sec_cap4_gt14p2", None, 1404),
        "ideal_vertical_tape_csc": ("sqrtE_csc_gt14p2", None, 1405),
    }
    peaks = {
        label: peak_analysis(
            production_rows,
            metric,
            s_new,
            s_new_sd,
            primary_window=window,
            seed=seed,
        )
        for label, (metric, window, seed) in peak_specs.items()
    }

    u_nm, q_gt = curve_arrays(curve_rows, "sqrtE_gt14p2")
    _, f_gt = curve_arrays(curve_rows, "flux_gt14p2")
    _, q_le = curve_arrays(curve_rows, "sqrtE_le14p2")
    _, q_floor = curve_arrays(curve_rows, "sqrtE_sec_gt14p2")
    _, q_floor_cap4 = curve_arrays(curve_rows, "sqrtE_sec_cap4_gt14p2")
    _, q_tape = curve_arrays(curve_rows, "sqrtE_csc_gt14p2")

    route_only = {
        "sqrtE": fit_route_only(
            "route_only_sqrtE_gt14p2_common_Model_B_sigma",
            "sqrt(E_exit/E0) for theta>14.2 deg",
            u_nm,
            q_gt,
            common_sigma,
            model_b_fit,
        ),
        "unweighted": fit_route_only(
            "route_only_unweighted_gt14p2_common_Model_B_sigma",
            "unit weight for theta>14.2 deg",
            u_nm,
            f_gt,
            common_sigma,
            model_b_fit,
        ),
        "horizontal_floor_sec": fit_route_only(
            "route_only_floor_sqrtE_sec_gt14p2_common_Model_B_sigma",
            "sqrt(E_exit/E0)*sec(theta) for theta>14.2 deg",
            u_nm,
            q_floor,
            common_sigma,
            model_b_fit,
        ),
        "horizontal_floor_sec_cap4": fit_route_only(
            "route_only_floor_sqrtE_minsec4_gt14p2_common_Model_B_sigma",
            "sqrt(E_exit/E0)*min(sec(theta),4) for theta>14.2 deg",
            u_nm,
            q_floor_cap4,
            common_sigma,
            model_b_fit,
        ),
        "ideal_vertical_tape_csc": fit_route_only(
            "route_only_tape_sqrtE_csc_gt14p2_common_Model_B_sigma",
            "sqrt(E_exit/E0)/sin(theta) for theta>14.2 deg",
            u_nm,
            q_tape,
            common_sigma,
            model_b_fit,
        ),
    }
    route_only["sqrtE_s_fixed_to_new_Model_B"] = fit_route_only_fixed_s(
        "route_only_sqrtE_gt14p2_fixed_s_common_Model_B_sigma",
        "sqrt(E_exit/E0) for theta>14.2 deg",
        u_nm,
        q_gt,
        common_sigma,
        model_b_fit,
        s_new,
    )

    mixtures: dict[str, dict[str, object]] = {}
    for basis_name, basis_description, values in (
        (
            "sqrtE",
            "sqrt(E_exit/E0) for theta>14.2 deg",
            q_gt,
        ),
        (
            "horizontal_floor_sec",
            "sqrt(E_exit/E0)*sec(theta) for theta>14.2 deg",
            q_floor,
        ),
        (
            "horizontal_floor_sec_cap4",
            "sqrt(E_exit/E0)*min(sec(theta),4) for theta>14.2 deg",
            q_floor_cap4,
        ),
        (
            "ideal_vertical_tape_csc",
            "sqrt(E_exit/E0)/sin(theta) for theta>14.2 deg; "
            "idealized selected azimuth only",
            q_tape,
        ),
    ):
        mixtures[basis_name] = {
            "free_s": fit_positive_mixture(
                f"{basis_name}_gt14p2_plus_direct_le14p2_free_s",
                basis_description,
                u_nm,
                values,
                q_le,
                common_sigma,
                model_b_fit,
                s_fixed=None,
            ),
            "s_fixed_to_new_Model_B": fit_positive_mixture(
                f"{basis_name}_gt14p2_plus_direct_le14p2_fixed_s",
                basis_description,
                u_nm,
                values,
                q_le,
                common_sigma,
                model_b_fit,
                s_fixed=s_new,
            ),
        }

    results_20 = json.loads((HERE / "analysis_results.json").read_text(encoding="utf-8"))
    threshold_comparison, comparison_rows = comparison_with_20(
        results_20,
        peaks,
        route_only["sqrtE"],
        mixtures["sqrtE"]["free_s"],
        mixtures["sqrtE"]["s_fixed_to_new_Model_B"],
        s_new,
    )
    base.write_csv(HERE / "threshold_14p2_vs_20.csv", comparison_rows)

    q_partition_error = max(
        abs(
            float(row["sqrtE_all"])
            - float(row["sqrtE_gt14p2"])
            - float(row["sqrtE_le14p2"])
        )
        for row in curve_rows
    )
    count_partition_error = max(
        abs(
            float(row["transmission"])
            - float(row["flux_gt14p2"])
            - float(row["flux_le14p2"])
        )
        for row in curve_rows
    )

    result = {
        "archive": str(base.ARCHIVE),
        "analysis_cut_deg_from_membrane_normal": ANGLE_CUT_DEG,
        "integrity": {
            "runs": len(runs),
            "incident_ions": sum(run.n_incident for run in runs),
            "transmitted_records": sum(run.n_transmitted for run in runs),
            "records_gt14p2": sum(run.n_gt_cut for run in runs),
            "exact_boundary_records": sum(run.exact_boundary_records for run in runs),
            "closest_record_to_threshold_deg": min(
                run.closest_angle_delta_deg for run in runs
            ),
            "max_Q_partition_roundoff": q_partition_error,
            "max_count_partition_roundoff": count_partition_error,
            "production_pooling": (
                "two independent 100k seeds pooled at 41 thicknesses; "
                "10k reconnaissance used only outside the production grid"
            ),
        },
        "definitions": {
            "unweighted_external": "sum[theta>14.2 deg]/N_incident",
            "sqrtE_external": (
                "sum sqrt(E_exit/E0)*1[theta>14.2 deg]/N_incident"
            ),
            "sqrtE_direct": (
                "sum sqrt(E_exit/E0)*1[theta<=14.2 deg]/N_incident"
            ),
            "horizontal_floor_sec": (
                "sum sqrt(E_exit/E0)*sec(theta)*1[theta>14.2 deg]/N_incident"
            ),
            "horizontal_floor_sec_cap4": (
                "same horizontal-floor proxy with min(sec(theta),4)"
            ),
            "ideal_vertical_tape_csc": (
                "sum sqrt(E_exit/E0)/sin(theta)*1[theta>14.2 deg]/N_incident"
            ),
        },
        "geometry_caveats": {
            "horizontal_floor": (
                "sec(theta) is the ideal planar path-length factor for a "
                "horizontal surface normal to the beam. It is not a "
                "material-specific SE yield and omits saturation and collection."
            ),
            "vertical_tape": (
                "1/sin(theta) assumes the ion's transverse direction is normal "
                "to a vertical tape plane. Actual interception and incidence "
                "also depend on azimuth, tape position/extent, shadowing, and "
                "detector visibility; none is available in the SRIM archive. "
                "This curve is therefore only a shape sensitivity, not a tape "
                "signal prediction."
            ),
            "mixture_amplitudes": (
                "All mixture amplitudes absorb unknown surface yield, "
                "intercepted fraction, detector visibility, and collection "
                "efficiency. Fitted components are not calibrated physical "
                "fractions."
            ),
        },
        "Model_B_common_error_reference": model_b_fit,
        "peaks": peaks,
        "population_partitions_at_experimental_thickness": {
            str(t_nm): population_at_experimental_thickness(
                curve_rows, s_new, t_nm
            )
            for t_nm in (40.0, 55.0)
        },
        "route_only_common_Model_B_sigma": route_only,
        "positive_external_plus_direct_mixtures_common_Model_B_sigma": mixtures,
        "comparison_with_20deg": threshold_comparison,
        "outputs": {
            "pooled_curves_csv": str(HERE / "threshold_14p2_pooled_curves.csv"),
            "comparison_csv": str(HERE / "threshold_14p2_vs_20.csv"),
            "results_json": str(HERE / "threshold_14p2_results.json"),
        },
    }
    (HERE / "threshold_14p2_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    compact = {
        "cut_deg": ANGLE_CUT_DEG,
        "s_new": s_new,
        "peaks": {
            label: {
                "grid_u_nm": value["grid_peak_u_nm"],
                "mapped_primary": value["mapped_primary_local_quadratic"],
            }
            for label, value in peaks.items()
        },
        "route_only_chi2_dof": {
            label: [fit["chi2"], fit["dof"]] for label, fit in route_only.items()
        },
        "mixture_chi2_dof": {
            label: {
                mode: [fit["data_chi2"], fit["data_dof"]]
                for mode, fit in fits.items()
            }
            for label, fits in mixtures.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Blind SE-emission model comparison: M-null, M-asym, and M-med.

Implements the campaign defined in ``DL/SE_model_spec_handoff.md`` with the
material-anisotropy deposition profiles of ``DL/medium_anisotropy_handoff.md``,
against the data preserved in this repository:

- Experimental data: ``data/reference_summaries/transmission_bayesian_summary.csv``
  plays the role of the spec's cross-normalized crystal files. Mode A (default)
  uses the carbon-normalized medians with symmetrized 68% errors. Mode B
  (``--normalization-nuisance``) uses the MoS2-posterior errors alone and adds
  one per-crystal normalization factor with a Gaussian prior whose width is that
  crystal's carbon-reference posterior relative width.
- Deposition profile: the ``IONIZ. by IONS`` column of
  ``data/SRIM 30 keV - Monolayer Long/IONIZ.txt`` (flat-extrapolated to z = 0).

All models share ``u = x / s`` and the escape integrals

    Y_f(u; lam, D) = int_0^u D(z) exp(-z/lam) dz
    Y_b(u; lam, D) = int_0^u D(z) exp(-(u - z)/lam) dz

with the back term entering as ``(1/C_r) * Y_b`` and one overall amplitude
``A`` (profiled analytically; two linear amplitudes for the C_r-free variant).
The material-anisotropy profiles fix beta = 1: a pure dilation of D is exactly
absorbed by the free stretch ``s`` (substitute (beta, s, lam) ->
(1, s/beta, beta*lam)), so beta is unidentifiable in these fits by construction.

Blind protocol: multistarts are drawn from loose bounds only; no previously
fitted values are used as starting points or priors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IONIZ = (
    REPOSITORY_ROOT
    / "data"
    / "SRIM 30 keV - Monolayer Long"
    / "IONIZ.txt"
)
DEFAULT_TRANSMISSION = (
    REPOSITORY_ROOT
    / "data"
    / "reference_summaries"
    / "transmission_bayesian_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "se_model_comparison"

ANGSTROM_PER_NM = 10.0
GRID_MAX_NM = 800.0
DEFAULT_DZ_NM = 0.25
CR_GRID = (1.0, 2.0)
DEFAULT_SEED = 2026
DEFAULT_STARTS = 60

CHI2_CAVEAT = (
    "Absolute chi2/nu is not interpretable as goodness of fit here (error "
    "model per DL/SE_model_spec_handoff.md); only delta-chi2 / delta-AIC "
    "between models fitted to identical data and error model are."
)
AIC_CONVENTION = "AIC = chi2_total + 2k (constant Gaussian term dropped)"

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "lam": (0.3, 60.0),
    "lam_f": (0.3, 60.0),
    "lam_b": (0.3, 60.0),
    "s": (0.3, 1.5),
    "rho": (0.0, 0.95),
    "lam_d": (5.0, 400.0),
    "chi0": (0.0, 1.0),
    "alpha": (0.05, 1.0),
    "n1": (0.8, 1.2),
    "n2": (0.8, 1.2),
    "n3": (0.8, 1.2),
}
LOG_SPACE_PARAMETERS = {"lam", "lam_f", "lam_b", "s", "lam_d", "alpha"}
NUISANCE_PARAMETERS = ("n1", "n2", "n3")


def portable_path(path: Path) -> str:
    """Represent repository files with portable forward-slash paths."""
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_ioniz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SRIM depth (nm) and ionization-by-ions (eV/nm) from IONIZ.txt.

    Keeps only lines that split into exactly three float-parseable tokens
    (depth, ionization by ions, ionization by recoils); SRIM's mantissa
    notation (``300100.E-04``) parses directly with :func:`float`. The profile
    is flat-extrapolated to z = 0 by prepending the first bin's value.

    The recoil column is excluded by design (a few percent of the ion channel
    in the reference case). Note for future readers: channeled ions suppress
    recoil ionization far more strongly than the electronic factor
    ``alpha*beta`` would suggest, so the recoil column must not be reused as a
    proxy for the channeled deposition profile.
    """
    depth_nm: list[float] = []
    ioniz: list[float] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        tokens = line.split()
        if len(tokens) != 3:
            continue
        try:
            depth, ions, _recoils = (float(token) for token in tokens)
        except ValueError:
            continue
        depth_nm.append(depth / ANGSTROM_PER_NM)
        ioniz.append(ions * ANGSTROM_PER_NM)
    if len(depth_nm) < 2:
        raise ValueError(f"{path} did not yield a usable ionization profile.")
    if depth_nm[0] <= 0.0 or np.any(np.diff(depth_nm) <= 0.0):
        raise ValueError(f"{path} depth grid must be positive and increasing.")
    depth_out = np.concatenate([[0.0], depth_nm])
    ioniz_out = np.concatenate([[ioniz[0]], ioniz])
    return depth_out, ioniz_out


@dataclass
class ExperimentData:
    """Cross-normalized SE intensities with per-crystal identity."""

    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    crystal: np.ndarray
    crystal_labels: list[str]
    mode: str
    norm_prior_sigma: np.ndarray | None


def load_experimental(path: Path, nuisance: bool) -> ExperimentData:
    """Load the Bayesian transmission summary in data mode A or B."""
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "thickness_nm",
        "normalized_median",
        "normalized_err_lower_68",
        "normalized_err_upper_68",
        "bayes_mu_mos2_median_0_255",
        "bayes_mu_mos2_ci16_0_255",
        "bayes_mu_mos2_ci84_0_255",
        "carbon_mu_median_0_255",
        "carbon_mu_ci16_0_255",
        "carbon_mu_ci84_0_255",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    labels = sorted(frame["dataset"].unique())
    crystal = frame["dataset"].map({label: i for i, label in enumerate(labels)})
    x = frame["thickness_nm"].to_numpy(dtype=float)

    if not nuisance:
        y = frame["normalized_median"].to_numpy(dtype=float)
        sigma = 0.5 * (
            frame["normalized_err_lower_68"].to_numpy(dtype=float)
            + frame["normalized_err_upper_68"].to_numpy(dtype=float)
        )
        prior = None
        mode = "A"
    else:
        carbon = frame["carbon_mu_median_0_255"].to_numpy(dtype=float)
        y = frame["bayes_mu_mos2_median_0_255"].to_numpy(dtype=float) / carbon
        sigma = (
            0.5
            * (
                frame["bayes_mu_mos2_ci84_0_255"].to_numpy(dtype=float)
                - frame["bayes_mu_mos2_ci16_0_255"].to_numpy(dtype=float)
            )
            / carbon
        )
        prior = np.empty(len(labels))
        for i, label in enumerate(labels):
            subset = frame[frame["dataset"] == label].iloc[0]
            prior[i] = (
                0.5
                * (
                    subset["carbon_mu_ci84_0_255"]
                    - subset["carbon_mu_ci16_0_255"]
                )
                / subset["carbon_mu_median_0_255"]
            )
        mode = "B"

    if np.any(sigma <= 0.0) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path} contains non-finite or non-positive errors.")
    return ExperimentData(
        x=x,
        y=y,
        sigma=sigma,
        crystal=crystal.to_numpy(dtype=int),
        crystal_labels=list(labels),
        mode=mode,
        norm_prior_sigma=prior,
    )


# ---------------------------------------------------------------------------
# Deposition profiles and escape integrals
# ---------------------------------------------------------------------------


class DepositionModel:
    """Piecewise-linear deposition profiles on a uniform SRIM-axis grid."""

    def __init__(
        self,
        srim_z: np.ndarray,
        srim_d: np.ndarray,
        dz: float = DEFAULT_DZ_NM,
        z_max: float = GRID_MAX_NM,
    ) -> None:
        self.srim_z = srim_z
        self.srim_d = srim_d
        self.dz = float(dz)
        self.z = np.arange(0.0, z_max + 0.5 * dz, dz)
        self.base = self.interp(self.z)

    def interp(self, z: np.ndarray) -> np.ndarray:
        """Baseline D(z): linear interpolation, zero beyond the SRIM range."""
        return np.interp(z, self.srim_z, self.srim_d, right=0.0)

    def profile_m2(self, rho: float, lam_d: float) -> np.ndarray:
        """M2 shift model: baseline advected by delta(z), Jacobian-conserving.

        delta(z) = delta_inf * (1 - exp(-z/lam_d)) with delta_inf = rho*lam_d;
        rho < 1 keeps the coordinate map monotone (positive Jacobian).
        """
        delta_inf = rho * lam_d
        decay = np.exp(-self.z / lam_d)
        delta = delta_inf * (1.0 - decay)
        jacobian = 1.0 - rho * decay
        return jacobian * self.interp(self.z - delta)

    def profile_m3(self, chi0: float, alpha: float) -> np.ndarray:
        """M3 two-component limit: (1-chi0)*D(z) + chi0*alpha*D(alpha*z)."""
        return (1.0 - chi0) * self.base + chi0 * alpha * self.interp(
            alpha * self.z
        )

    def profile_full(
        self,
        chi0: float,
        alpha: float,
        lam_d: float,
        n_quad: int = 300,
    ) -> np.ndarray:
        """Full three-term remapping model (beta = 1), verification only.

        Random, still-channeled, and dechanneled populations per
        DL/medium_anisotropy_handoff.md; the dechanneling integral is done by
        fixed-order Gauss-Legendre quadrature (the prototype's 300-400 point
        rule). Not fitted -- used to validate the M2/M3 reductions.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        z = self.z
        term_random = (1.0 - chi0) * self.base
        term_channeled = (
            chi0 * np.exp(-z / lam_d) * alpha * self.interp(alpha * z)
        )
        z_d = 0.5 * np.outer(z, nodes + 1.0)
        integrand = (
            np.exp(-z_d / lam_d)
            / lam_d
            * self.interp(alpha * z_d + (z[:, None] - z_d))
        )
        term_dechanneled = chi0 * 0.5 * z * (integrand @ weights)
        return term_random + term_channeled + term_dechanneled


def escape_integrals(
    z: np.ndarray,
    dz: float,
    profile: np.ndarray,
    lam: float,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Y_f(u) and Y_b(u) continuously in u (spec section 5).

    The profile is treated as piecewise linear on the uniform grid; each
    segment is integrated against the exponential kernel in closed form, so
    the integrals are exact for the discretized profile and continuous in the
    upper limit (no truncation to whole bins). Y_b uses the stable recursion
    Y_b(z_{i+1}) = Y_b(z_i)*exp(-dz/lam) + segment  -- never the overflowing
    exp(-u/lam)*int D exp(z/lam) form.
    """
    u = np.asarray(u, dtype=float)
    d0 = profile[:-1]
    slope = np.diff(profile) / dz
    x_seg = dz / lam
    decay_seg = np.exp(-x_seg)
    grow_seg = -np.expm1(-x_seg)  # 1 - exp(-dz/lam), accurately

    front_segments = np.exp(-z[:-1] / lam) * (
        d0 * lam * grow_seg
        + slope * (lam * lam * grow_seg - lam * dz * decay_seg)
    )
    front_nodes = np.concatenate([[0.0], np.cumsum(front_segments)])

    back_segments = lam * (
        d0 * grow_seg + slope * (dz - lam * grow_seg)
    )
    back_nodes = np.concatenate(
        [[0.0], lfilter([1.0], [1.0, -decay_seg], back_segments)]
    )

    u_eff = np.minimum(u, z[-1])
    index = np.clip(
        np.searchsorted(z, u_eff, side="right") - 1, 0, len(z) - 2
    )
    t = u_eff - z[index]
    d_left = profile[index]
    seg_slope = (profile[index + 1] - profile[index]) / dz
    decay_t = np.exp(-t / lam)
    grow_t = -np.expm1(-t / lam)

    y_front = front_nodes[index] + np.exp(-z[index] / lam) * (
        d_left * lam * grow_t
        + seg_slope * (lam * lam * grow_t - lam * t * decay_t)
    )
    y_back = back_nodes[index] * decay_t + lam * (
        d_left * grow_t + seg_slope * (t - lam * grow_t)
    )
    # Beyond the grid the profile is zero: Y_f is constant, Y_b decays.
    overshoot = u > z[-1]
    if np.any(overshoot):
        y_back = np.where(
            overshoot, y_back * np.exp(-(u - u_eff) / lam), y_back
        )
    return y_front, y_back


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitConfig:
    """One entry of the campaign: model structure plus C_r handling."""

    key: str
    model: str  # nfix | nfree | masym | mmed2 | mmed3
    label: str
    cr: float | None
    params: tuple[str, ...]


def build_configs(nuisance: bool) -> list[FitConfig]:
    """The nine campaign fits, optionally with per-crystal nuisances."""
    extra = NUISANCE_PARAMETERS if nuisance else ()
    configs = [
        FitConfig("nfix_cr1", "nfix", "M-null N-fix (C_r=1)", 1.0,
                  ("lam", "s") + extra),
        FitConfig("nfix_cr2", "nfix", "M-null N-fix (C_r=2)", 2.0,
                  ("lam", "s") + extra),
        FitConfig("nfree", "nfree", "M-null N-free", None,
                  ("lam", "s") + extra),
        FitConfig("masym_cr1", "masym", "M-asym (C_r=1)", 1.0,
                  ("lam_f", "lam_b", "s") + extra),
        FitConfig("masym_cr2", "masym", "M-asym (C_r=2)", 2.0,
                  ("lam_f", "lam_b", "s") + extra),
        FitConfig("mmed2_cr1", "mmed2", "M-med M2-shift (C_r=1)", 1.0,
                  ("lam", "s", "rho", "lam_d") + extra),
        FitConfig("mmed2_cr2", "mmed2", "M-med M2-shift (C_r=2)", 2.0,
                  ("lam", "s", "rho", "lam_d") + extra),
        FitConfig("mmed3_cr1", "mmed3", "M-med M3-two-comp (C_r=1)", 1.0,
                  ("lam", "s", "chi0", "alpha") + extra),
        FitConfig("mmed3_cr2", "mmed3", "M-med M3-two-comp (C_r=2)", 2.0,
                  ("lam", "s", "chi0", "alpha") + extra),
    ]
    return configs


def component_shapes(
    dep: DepositionModel,
    config: FitConfig,
    theta: dict[str, float],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Front and back escape components at measured thicknesses x."""
    u = np.asarray(x, dtype=float) / theta["s"]
    if config.model == "mmed2":
        profile = dep.profile_m2(theta["rho"], theta["lam_d"])
    elif config.model == "mmed3":
        profile = dep.profile_m3(theta["chi0"], theta["alpha"])
    else:
        profile = dep.base
    if config.model == "masym":
        y_front, _ = escape_integrals(
            dep.z, dep.dz, profile, theta["lam_f"], u
        )
        _, y_back = escape_integrals(
            dep.z, dep.dz, profile, theta["lam_b"], u
        )
    else:
        y_front, y_back = escape_integrals(
            dep.z, dep.dz, profile, theta["lam"], u
        )
    return y_front, y_back


def nuisance_factors(
    config: FitConfig,
    theta: dict[str, float],
    crystal: np.ndarray,
) -> np.ndarray:
    """Per-point normalization factors (all ones in data mode A)."""
    if NUISANCE_PARAMETERS[0] not in config.params:
        return np.ones(len(crystal))
    values = np.array([theta[name] for name in NUISANCE_PARAMETERS])
    return values[crystal]


def solve_amplitudes(
    config: FitConfig,
    y_front: np.ndarray,
    y_back: np.ndarray,
    factors: np.ndarray,
    data: ExperimentData,
) -> dict[str, float]:
    """Profile the conditionally-linear amplitude(s) by weighted LS."""
    weight = 1.0 / data.sigma**2
    if config.model == "nfree":
        col_f = factors * y_front
        col_b = factors * y_back
        gram = np.array(
            [
                [np.sum(weight * col_f * col_f),
                 np.sum(weight * col_f * col_b)],
                [np.sum(weight * col_f * col_b),
                 np.sum(weight * col_b * col_b)],
            ]
        )
        rhs = np.array(
            [np.sum(weight * col_f * data.y), np.sum(weight * col_b * data.y)]
        )
        try:
            a_front, a_back = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            (a_front, a_back), *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        cr = a_front / a_back if a_back != 0.0 else np.inf
        return {
            "A": float(a_front),
            "cr": float(cr),
            "a_back": float(a_back),
        }
    shape = factors * (y_front + y_back / config.cr)
    denominator = np.sum(weight * shape * shape)
    amplitude = (
        np.sum(weight * shape * data.y) / denominator
        if denominator > 0.0
        else 0.0
    )
    return {"A": float(amplitude)}


def predict(
    dep: DepositionModel,
    config: FitConfig,
    theta_full: dict[str, float],
    x: np.ndarray,
    crystal: np.ndarray | None,
) -> np.ndarray:
    """Model intensity in the physical basis (amplitude explicit)."""
    y_front, y_back = component_shapes(dep, config, theta_full, x)
    cr = theta_full["cr"] if config.model == "nfree" else config.cr
    prediction = theta_full["A"] * (y_front + y_back / cr)
    if crystal is not None:
        prediction = prediction * nuisance_factors(config, theta_full, crystal)
    return prediction


def residuals_full(
    dep: DepositionModel,
    config: FitConfig,
    data: ExperimentData,
    theta_full: dict[str, float],
) -> np.ndarray:
    """Weighted data residuals plus nuisance-prior residuals (mode B)."""
    prediction = predict(dep, config, theta_full, data.x, data.crystal)
    residual = (prediction - data.y) / data.sigma
    if data.norm_prior_sigma is not None:
        priors = np.array(
            [
                (theta_full[name] - 1.0) / data.norm_prior_sigma[i]
                for i, name in enumerate(NUISANCE_PARAMETERS)
            ]
        )
        residual = np.concatenate([residual, priors])
    return residual


def make_profiled_residual(
    dep: DepositionModel,
    config: FitConfig,
    data: ExperimentData,
):
    """Residual over the nonlinear parameters with amplitudes profiled."""

    def recover_full(vector: np.ndarray) -> dict[str, float]:
        theta = dict(zip(config.params, np.asarray(vector, dtype=float)))
        y_front, y_back = component_shapes(dep, config, theta, data.x)
        factors = nuisance_factors(config, theta, data.crystal)
        theta.update(solve_amplitudes(config, y_front, y_back, factors, data))
        return theta

    def residual(vector: np.ndarray) -> np.ndarray:
        theta = dict(zip(config.params, np.asarray(vector, dtype=float)))
        y_front, y_back = component_shapes(dep, config, theta, data.x)
        factors = nuisance_factors(config, theta, data.crystal)
        amplitudes = solve_amplitudes(config, y_front, y_back, factors, data)
        if config.model == "nfree":
            prediction = factors * (
                amplitudes["A"] * y_front + amplitudes["a_back"] * y_back
            )
        else:
            prediction = (
                amplitudes["A"] * factors * (y_front + y_back / config.cr)
            )
        out = (prediction - data.y) / data.sigma
        if data.norm_prior_sigma is not None:
            priors = np.array(
                [
                    (theta[name] - 1.0) / data.norm_prior_sigma[i]
                    for i, name in enumerate(NUISANCE_PARAMETERS)
                ]
            )
            out = np.concatenate([out, priors])
        return out

    residual.recover_full = recover_full
    return residual


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def draw_start(rng: np.random.Generator, name: str) -> float:
    """Dispersed start: log-uniform for scale-type parameters."""
    low, high = PARAMETER_BOUNDS[name]
    if name in NUISANCE_PARAMETERS:
        return rng.uniform(0.97, 1.03)
    if name in LOG_SPACE_PARAMETERS:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    span = high - low
    return float(rng.uniform(low + 0.02 * span, high - 0.05 * span))


def full_parameter_names(config: FitConfig) -> list[str]:
    """Physical-basis parameter ordering: amplitudes first."""
    names = ["A"]
    if config.model == "nfree":
        names.append("cr")
    return names + list(config.params)


def covariance_matrix(
    dep: DepositionModel,
    config: FitConfig,
    data: ExperimentData,
    theta_full: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Local covariance (J^T J)^-1 of the weighted residuals, physical basis.

    Unscaled by chi2/nu on purpose: the supplied errors define the weights and
    the chi2 caveat is reported alongside (spec section 6).
    """
    names = full_parameter_names(config)
    p_hat = np.array([theta_full[name] for name in names])
    n_res = len(data.x) + (0 if data.norm_prior_sigma is None else 3)
    jacobian = np.empty((n_res, len(names)))
    for j, name in enumerate(names):
        scale = 1.0
        if name in PARAMETER_BOUNDS:
            low, high = PARAMETER_BOUNDS[name]
            scale = high - low
        step = 1e-6 * max(abs(p_hat[j]), 0.01 * scale)
        for sign in (1.0, -1.0):
            perturbed = dict(theta_full)
            perturbed[name] = p_hat[j] + sign * step
            if sign > 0:
                upper = residuals_full(dep, config, data, perturbed)
            else:
                lower = residuals_full(dep, config, data, perturbed)
        jacobian[:, j] = (upper - lower) / (2.0 * step)
    normal = jacobian.T @ jacobian
    covariance = np.linalg.pinv(normal)
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    # A pinned/degenerate parameter has sigma = 0: its correlations are
    # undefined (0/0 or x/0) -- zero them rather than reporting DBL_MAX, and
    # clip pinv round-off that pushes |corr| microscopically above 1.
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    return covariance, correlation, names


def identifiability_flags(
    names: list[str],
    values: np.ndarray,
    sigma: np.ndarray,
    correlation: np.ndarray,
) -> list[str]:
    """Spec section 6 flags: strong correlations, huge errors, pinned bounds."""
    flags: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(correlation[i, j]) > 0.95:
                flags.append(
                    f"|corr({names[i]}, {names[j]})| = "
                    f"{abs(correlation[i, j]):.3f} > 0.95"
                )
    for i, name in enumerate(names):
        if sigma[i] > abs(values[i]) and name not in NUISANCE_PARAMETERS:
            flags.append(
                f"sigma({name}) = {sigma[i]:.3g} exceeds |{name}| = "
                f"{abs(values[i]):.3g}"
            )
        if name in PARAMETER_BOUNDS:
            low, high = PARAMETER_BOUNDS[name]
            span = high - low
            if values[i] - low < 1e-3 * span:
                flags.append(f"{name} pinned at lower bound {low}")
            elif high - values[i] < 1e-3 * span:
                flags.append(f"{name} pinned at upper bound {high}")
    return flags


def run_fit(
    dep: DepositionModel,
    config: FitConfig,
    data: ExperimentData,
    n_starts: int,
    seed: int,
    fit_index: int,
) -> dict[str, object]:
    """Multistart weighted least squares for one campaign entry."""
    rng = np.random.default_rng([seed, fit_index])
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in config.params])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in config.params])
    margin = 1e-9 * (upper - lower)
    residual = make_profiled_residual(dep, config, data)

    # Stage 1: cheap scouting pass over all dispersed starts.
    scouted = []
    for _ in range(n_starts):
        start = np.array([draw_start(rng, name) for name in config.params])
        start = np.clip(start, lower + margin, upper - margin)
        result = least_squares(
            residual,
            x0=start,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=400,
        )
        scouted.append((float(np.sum(result.fun**2)), result.x))
    scouted.sort(key=lambda item: item[0])

    # Stage 2: tight polish of the best scouting basins.
    best_result = None
    best_chi2 = np.inf
    for _, start in scouted[:5]:
        result = least_squares(
            residual,
            x0=np.clip(start, lower + margin, upper - margin),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=3000,
        )
        chi2 = float(np.sum(result.fun**2))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_result = result
    if best_result is None:
        raise RuntimeError(f"No optimization start converged for {config.key}")

    theta_full = residual.recover_full(best_result.x)
    all_res = residuals_full(dep, config, data, theta_full)
    n_data = len(data.x)
    chi2_data = float(np.sum(all_res[:n_data] ** 2))
    chi2_total = float(np.sum(all_res**2))
    k = len(full_parameter_names(config))
    dof = len(all_res) - k

    covariance, correlation, names = covariance_matrix(
        dep, config, data, theta_full
    )
    values = np.array([theta_full[name] for name in names])
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    flags = identifiability_flags(names, values, sigma, correlation)

    derived = derived_quantities(config, names, values, covariance)
    result_payload: dict[str, object] = {
        "key": config.key,
        "label": config.label,
        "model": config.model,
        "cr_mode": "free" if config.cr is None else f"fixed={config.cr:g}",
        "data_mode": data.mode,
        "parameter_names": names,
        "parameters": {n: float(v) for n, v in zip(names, values)},
        "parameter_sigma": {n: float(s) for n, s in zip(names, sigma)},
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "chi2_data": chi2_data,
        "chi2_total": chi2_total,
        "n_points": n_data,
        "k_parameters": k,
        "dof": dof,
        "reduced_chi2": chi2_total / dof if dof > 0 else np.nan,
        "aic": chi2_total + 2.0 * k,
        "aic_convention": AIC_CONVENTION,
        "chi2_caveat": CHI2_CAVEAT,
        "identifiability_flags": flags,
        "identified": not any(
            "sigma(" in f or "corr" in f for f in flags
        ),
        "derived": derived,
        "theta_vector": best_result.x.tolist(),
    }
    if config.model == "nfree":
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        order = np.argsort(eigenvalues)[::-1]
        result_payload["degeneracy_structure"] = {
            "correlation_eigenvalues": eigenvalues[order].tolist(),
            "leading_eigenvector": {
                n: float(v)
                for n, v in zip(names, eigenvectors[:, order[0]])
            },
        }
    return result_payload


def derived_quantities(
    config: FitConfig,
    names: list[str],
    values: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Measured-axis lengths (lam*s) and delta_inf with propagated errors."""
    index = {name: i for i, name in enumerate(names)}
    derived: dict[str, dict[str, float]] = {}

    def product(name_a: str, name_b: str, out_name: str) -> None:
        i, j = index[name_a], index[name_b]
        value = values[i] * values[j]
        gradient = np.zeros(len(names))
        gradient[i] = values[j]
        gradient[j] = values[i]
        variance = float(gradient @ covariance @ gradient)
        derived[out_name] = {
            "value": float(value),
            "sigma": float(np.sqrt(max(variance, 0.0))),
        }

    scale_pairs = {
        "nfix": [("lam", "s", "lam_times_s_nm")],
        "nfree": [("lam", "s", "lam_times_s_nm")],
        "masym": [
            ("lam_f", "s", "lam_f_times_s_nm"),
            ("lam_b", "s", "lam_b_times_s_nm"),
        ],
        "mmed2": [
            ("lam", "s", "lam_times_s_nm"),
            ("rho", "lam_d", "delta_inf_nm"),
        ],
        "mmed3": [("lam", "s", "lam_times_s_nm")],
    }
    for name_a, name_b, out_name in scale_pairs[config.model]:
        product(name_a, name_b, out_name)
    return derived


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def configure_style() -> None:
    """Compact journal-style Matplotlib defaults (as plot_model_fit.py)."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.0,
            "errorbar.capsize": 2,
        }
    )


FLAKE_STYLES = {
    0: {"label": "Flake 1", "marker": "o", "color": "#DC143C"},
    1: {"label": "Flake 2", "marker": "s", "color": "#1F77B4"},
    2: {"label": "Flake 3", "marker": "^", "color": "#006400"},
}
MODEL_STYLES = {
    "nfix": {"color": "black", "linestyle": "-", "label": "M-null N-fix"},
    "nfree": {"color": "0.5", "linestyle": "-", "label": "M-null N-free"},
    "masym": {"color": "#E69F00", "linestyle": "--", "label": "M-asym"},
    "mmed2": {"color": "#7B3294", "linestyle": "-.",
              "label": "M-med M2-shift"},
    "mmed3": {"color": "#A6611A", "linestyle": (0, (1, 1)),
              "label": "M-med M3-two-comp"},
}


def comparison_figure(
    dep: DepositionModel,
    configs: list[FitConfig],
    results: dict[str, dict[str, object]],
    data: ExperimentData,
    path: Path,
) -> None:
    """Two panels (C_r = 1 and 2), all fitted curves over the data."""
    import matplotlib.pyplot as plt

    configure_style()
    figure, axes_pair = plt.subplots(
        1, 2, figsize=(7.0, 3.1), sharey=True, sharex=True
    )
    x_dense = np.linspace(1.0, 220.0, 440)
    for axes, cr_value in zip(axes_pair, CR_GRID):
        for config in configs:
            if config.cr is not None and config.cr != cr_value:
                continue
            payload = results[config.key]
            theta_full = {
                name: payload["parameters"][name]
                for name in payload["parameter_names"]
            }
            curve = predict(dep, config, theta_full, x_dense, None)
            style = MODEL_STYLES[config.model]
            label = style["label"]
            if config.model == "nfree":
                label = (
                    f"{label} (C_r={payload['parameters']['cr']:.2f})"
                )
            axes.plot(
                x_dense,
                curve,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.1,
                label=label,
                zorder=2,
            )
        for crystal_id, style in FLAKE_STYLES.items():
            mask = data.crystal == crystal_id
            axes.errorbar(
                data.x[mask],
                data.y[mask],
                yerr=data.sigma[mask],
                fmt=style["marker"],
                color=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=4,
                linestyle="none",
                elinewidth=0.6,
                zorder=3,
                label=style["label"],
            )
        axes.set_xlim(0.0, 220.0)
        axes.set_ylim(0.0, 1.75)
        axes.set_xlabel(r"$t$ (nm)")
        axes.set_title(f"$C_r$ = {cr_value:g} grid", fontsize=9)
    axes_pair[0].set_ylabel(r"$I_\mathrm{n}$")
    axes_pair[0].legend(frameon=False, loc="lower left", handlelength=1.8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def residual_figure(
    dep: DepositionModel,
    configs: list[FitConfig],
    results: dict[str, dict[str, object]],
    data: ExperimentData,
    path: Path,
) -> None:
    """Weighted residuals vs thickness, one panel per fit, crystals marked."""
    import matplotlib.pyplot as plt

    configure_style()
    figure, axes_grid = plt.subplots(
        3, 3, figsize=(7.5, 6.6), sharex=True, sharey=True
    )
    for axes, config in zip(axes_grid.ravel(), configs):
        payload = results[config.key]
        theta_full = {
            name: payload["parameters"][name]
            for name in payload["parameter_names"]
        }
        prediction = predict(dep, config, theta_full, data.x, data.crystal)
        residual = (prediction - data.y) / data.sigma
        for crystal_id, style in FLAKE_STYLES.items():
            mask = data.crystal == crystal_id
            axes.plot(
                data.x[mask],
                residual[mask],
                style["marker"],
                color=style["color"],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=4,
                linestyle="none",
            )
        axes.axhline(0.0, color="0.6", linewidth=0.7)
        axes.set_title(
            f"{config.label}\n"
            rf"$\chi^2$={payload['chi2_total']:.2f}",
            fontsize=7.5,
        )
    for axes in axes_grid[-1, :]:
        axes.set_xlabel(r"$t$ (nm)")
    for axes in axes_grid[:, 0]:
        axes.set_ylabel(r"$r_i$")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def comparison_table(
    configs: list[FitConfig],
    results: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Model x C_r comparison with deltas relative to the best fit."""
    rows = []
    for config in configs:
        payload = results[config.key]
        rows.append(
            {
                "fit": config.key,
                "label": config.label,
                "data_mode": payload["data_mode"],
                "k": payload["k_parameters"],
                "chi2": payload["chi2_total"],
                "dof": payload["dof"],
                "aic": payload["aic"],
                "identified": payload["identified"],
                "n_flags": len(payload["identifiability_flags"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["delta_chi2"] = frame["chi2"] - frame["chi2"].min()
    frame["delta_aic"] = frame["aic"] - frame["aic"].min()
    return frame.sort_values("aic").reset_index(drop=True)


def convergence_check(
    srim_z: np.ndarray,
    srim_d: np.ndarray,
    config: FitConfig,
    data: ExperimentData,
    payload: dict[str, object],
    dz: float,
) -> dict[str, object]:
    """Refit the given result at dz/2 and report parameter shifts vs sigma."""
    dep_half = DepositionModel(srim_z, srim_d, dz=dz / 2.0)
    residual = make_profiled_residual(dep_half, config, data)
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in config.params])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in config.params])
    start = np.clip(
        np.array(payload["theta_vector"]),
        lower + 1e-9 * (upper - lower),
        upper - 1e-9 * (upper - lower),
    )
    refit = least_squares(
        residual,
        x0=start,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    theta_half = residual.recover_full(refit.x)
    shifts = {}
    worst = 0.0
    for name in payload["parameter_names"]:
        sigma = payload["parameter_sigma"][name]
        delta = abs(theta_half[name] - payload["parameters"][name])
        ratio = delta / sigma if sigma > 0 else np.inf
        shifts[name] = {"delta": float(delta), "delta_over_sigma": ratio}
        worst = max(worst, ratio)
    return {
        "fit": config.key,
        "dz_nm": dz,
        "dz_half_nm": dz / 2.0,
        "chi2_at_dz_half": float(np.sum(refit.fun**2)),
        "parameter_shifts": shifts,
        "max_delta_over_sigma": worst,
    }


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------


def run_campaign(
    ioniz_path: Path,
    data_path: Path,
    output_dir: Path,
    nuisance: bool,
    dz: float,
    n_starts: int,
    seed: int,
) -> None:
    """Fit all nine configurations, then write reports and figures."""
    srim_z, srim_d = load_ioniz(ioniz_path)
    dep = DepositionModel(srim_z, srim_d, dz=dz)
    data = load_experimental(data_path, nuisance=nuisance)
    configs = build_configs(nuisance)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    for index, config in enumerate(configs):
        print(f"Fitting {config.label} [{data.mode}] ...", flush=True)
        payload = run_fit(dep, config, data, n_starts, seed, index)
        payload["inputs"] = {
            "ioniz": portable_path(ioniz_path),
            "data": portable_path(data_path),
            "dz_nm": dz,
            "grid_max_nm": GRID_MAX_NM,
            "n_starts": n_starts,
            "seed": seed,
        }
        results[config.key] = payload
        fit_path = output_dir / f"fit_{config.key}_{data.mode}.json"
        fit_path.write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )

    table = comparison_table(configs, results)
    table_path = output_dir / f"comparison_table_{data.mode}.csv"
    table.to_csv(table_path, index=False)

    best_key = table.iloc[0]["fit"]
    best_config = next(c for c in configs if c.key == best_key)
    convergence = convergence_check(
        srim_z, srim_d, best_config, data, results[best_key], dz
    )

    figure_path = output_dir / f"comparison_{data.mode}.png"
    residual_path = output_dir / f"residuals_{data.mode}.png"
    comparison_figure(dep, configs, results, data, figure_path)
    residual_figure(dep, configs, results, data, residual_path)

    summary = {
        "data_mode": data.mode,
        "aic_convention": AIC_CONVENTION,
        "chi2_caveat": CHI2_CAVEAT,
        "table": table.to_dict(orient="records"),
        "convergence_check": convergence,
        "figures": [portable_path(figure_path), portable_path(residual_path)],
    }
    summary_path = output_dir / f"campaign_summary_{data.mode}.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
            default=lambda value: value.item()
            if hasattr(value, "item")
            else str(value),
        ),
        encoding="utf-8",
    )

    print(f"\n=== Comparison table (data mode {data.mode}) ===")
    columns = ["label", "k", "chi2", "dof", "aic", "delta_chi2", "delta_aic"]
    header = (
        f"{'fit':<26}{'k':>3}{'chi2':>10}{'dof':>5}{'AIC':>10}"
        f"{'dChi2':>9}{'dAIC':>9}  flags"
    )
    print(header)
    for _, row in table.iterrows():
        payload = results[row["fit"]]
        flags = payload["identifiability_flags"]
        flag_text = "; ".join(flags) if flags else "-"
        print(
            f"{row['label']:<26}{row['k']:>3}{row['chi2']:>10.3f}"
            f"{row['dof']:>5}{row['aic']:>10.3f}{row['delta_chi2']:>9.3f}"
            f"{row['delta_aic']:>9.3f}  {flag_text}"
        )
    print(f"\n{CHI2_CAVEAT}")
    print(
        f"dz/2 convergence ({best_key}): max |delta p| / sigma_p = "
        f"{convergence['max_delta_over_sigma']:.3g}"
    )
    print(f"Outputs: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ioniz",
        type=Path,
        default=DEFAULT_IONIZ,
        help=f"SRIM IONIZ.txt (default: {DEFAULT_IONIZ}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_TRANSMISSION,
        help=f"Bayesian transmission summary (default: {DEFAULT_TRANSMISSION}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--normalization-nuisance",
        action="store_true",
        help="Data mode B: per-crystal normalization nuisances with "
        "carbon-width Gaussian priors.",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=DEFAULT_DZ_NM,
        help=f"Uniform SRIM-axis grid step in nm (default: {DEFAULT_DZ_NM}).",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=DEFAULT_STARTS,
        help=f"Multistart count per fit (default: {DEFAULT_STARTS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for start dispersion (default: {DEFAULT_SEED}).",
    )
    return parser


def main() -> None:
    """Run the full campaign for the selected data mode."""
    matplotlib.use("Agg")
    args = build_parser().parse_args()
    run_campaign(
        ioniz_path=args.ioniz.resolve(),
        data_path=args.data.resolve(),
        output_dir=args.output_dir.resolve(),
        nuisance=args.normalization_nuisance,
        dz=args.dz,
        n_starts=args.starts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

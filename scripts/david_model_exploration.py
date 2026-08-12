"""Overlay the experimental SE intensity on the raw SRIM ionization profile.

This exploratory starting script compares two curves on a shared thickness axis:

1. The experimental normalized transmission intensity (all three flakes).
2. The SRIM ``IONIZ. by IONS`` depth profile (electronic energy loss of the
   30 keV primary beam), read directly from a SRIM-2013 ``IONIZ.txt`` output.

Each curve keeps its own y-axis so their shapes can be eyeballed together
without imposing any normalization or fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import interpolate, optimize, signal

SHOW_PLOT = True
DEBUG = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "DL"))
from thickness_sweep_ionization import (  # noqa: E402  (needs REPOSITORY_ROOT)
    DEFAULT_SWEEP_DIR,
    build_ionization_model,
    discover_sweep,
)
STYLE_FILE = REPOSITORY_ROOT / "DL" / "dlister_r01.mplstyle"
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
DEFAULT_PNG = REPOSITORY_ROOT / "outputs" / "david_model_exploration.png"
DEFAULT_PAPER_FIG = REPOSITORY_ROOT / "outputs" / "paper_model_comparison.pdf"

ANGSTROM_PER_NM = 10.0


def load_ionization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SRIM depth (nm) and the ionization-by-ions column (eV/Angstrom).

    The SRIM header and separators are skipped by keeping only lines that split
    into exactly three float-parseable tokens: depth, ionization by ions, and
    ionization by recoils. SRIM's mantissa notation (``300100.E-04``) parses
    directly with :func:`float`.
    """
    depth_ang: list[float] = []
    ioniz_ions: list[float] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        tokens = line.split()
        if len(tokens) != 3:
            continue
        try:
            depth, ions, _recoils = (float(token) for token in tokens)
        except ValueError:
            continue
        depth_ang.append(depth)
        ioniz_ions.append(ions)

    if len(depth_ang) < 2:
        raise ValueError(f"{path} did not yield a usable ionization profile.")
    return np.asarray(depth_ang) / ANGSTROM_PER_NM, np.asarray(ioniz_ions)


def load_him_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the measured HIM data as plain numpy arrays, sorted by thickness.

    Returns ``(eels_thickness, HIM_intensity, HIM_intensity_error, flake)``:
    EELS-measured thickness (nm), normalized HIM SE intensity, symmetrized 68%
    intensity error (same convention as fit_se_models.load_experimental), and
    the flake/dataset label for each point.
    """
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "thickness_nm",
        "normalized_median",
        "normalized_err_lower_68",
        "normalized_err_upper_68",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    order = np.argsort(frame["thickness_nm"].to_numpy(dtype=float))
    eels_thickness = frame["thickness_nm"].to_numpy(dtype=float)[order]
    him_intensity = frame["normalized_median"].to_numpy(dtype=float)[order]
    him_intensity_error = (
        0.5
        * (
            frame["normalized_err_lower_68"].to_numpy(dtype=float)
            + frame["normalized_err_upper_68"].to_numpy(dtype=float)
        )
    )[order]
    flake = frame["dataset"].to_numpy(dtype=str)[order]
    return eels_thickness, him_intensity, him_intensity_error, flake


def configure_style() -> None:
    """Apply David's style sheet (DL/dlister_r01.mplstyle)."""
    matplotlib.style.use(str(STYLE_FILE))


FLAKE_STYLES = {
    "Dataset 1": {"label": "Flake 1", "marker": "o", "color": "#DC143C"},
    "Dataset 2": {"label": "Flake 2", "marker": "s", "color": "#1F77B4"},
    "Dataset 3": {"label": "Flake 3", "marker": "^", "color": "#006400"},
}

SRIM_COLOR = "#555555"


def create_plot(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = DEFAULT_PNG,
    show_figure: bool = SHOW_PLOT,
) -> None:
    """Draw the measured data and SRIM ionization on twin y-axes.

    All data come in as plain numpy arrays; ``flake`` optionally labels each
    measured point (per-flake colors/markers), otherwise one neutral style.
    """
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()

    figure, axes = plt.subplots()
    srim_axes = axes.twinx()
    srim_axes.grid(False)  # one grid only; the twin axis is unaligned

    # Right axis: raw SRIM electronic-ionization profile vs depth.
    srim_axes.plot(
        ionization_thickness,
        ionization,
        color=SRIM_COLOR,
        linewidth=1.2,
        zorder=1,
        label="SRIM ioniz. by ions",
    )

    # Left axis: measured normalized intensity, one series per flake.
    if flake is None:
        groups = [(None, np.ones(len(eels_thickness), dtype=bool))]
    else:
        groups = [(label, flake == label) for label in np.unique(flake)]
    for dataset_label, mask in groups:
        style = FLAKE_STYLES.get(
            dataset_label,
            {"label": "Data", "marker": "o", "color": "#1F77B4"},
        )
        axes.errorbar(
            eels_thickness[mask],
            HIM_intensity[mask],
            yerr=HIM_intensity_error[mask],
            fmt=style["marker"],
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5,
            linestyle="none",
            elinewidth=0.7,
            zorder=3,
            label=style["label"],
        )

    axes.set_xlabel(r"$t$ (nm)")
    axes.set_ylabel(r"$I_\mathrm{n}$ (normalized intensity)")
    srim_axes.set_ylabel("SRIM ionization by ions (eV/Å)", color=SRIM_COLOR)
    srim_axes.tick_params(axis="y", labelcolor=SRIM_COLOR)
    axes.set_xlim(left=0.0)
    axes.set_ylim(bottom=0.0)
    srim_axes.set_ylim(bottom=0.0)

    data_handles, data_labels = axes.get_legend_handles_labels()
    srim_handles, srim_labels = srim_axes.get_legend_handles_labels()
    axes.legend(data_handles + srim_handles, data_labels + srim_labels)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, bbox_inches="tight")
    print(f"Saved exploration plot: {png_path}")
    print(
        f"SRIM ions profile: {len(ionization_thickness)} points, "
        f"depth {ionization_thickness.min():.2f}-"
        f"{ionization_thickness.max():.2f} nm"
    )

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


def fit_and_plot_models(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> None:
    """Fit the exploratory escape-depth models to the HIM data, printing and
    plotting each fit, then compare them under identical raw-sigma weighting."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)

    # Null
    # Same escape depth front/back, horizontal stretch, separate f/b collection coefficients
    ionization_interp = interpolate.interp1d(
        ionization_thickness,
        ionization,
        kind='cubic',
        fill_value=(ionization[0], ionization[-1]),
        bounds_error=False
    )

    def null_model(x_array, C_overall, C_fbratio, lambda_all, stretch, n=8192):
        # lambda_all: SE escape depth in nm (same units as x_array); unit-area kernel
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        a = dt / lambda_all
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a  # exact per-cell front kernel, valid for any lambda vs dt
        cell = np.exp(-t[:-1] / lambda_all) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back) / lambda_all)

    null_initial_guess = [0.02, 3.5, 2, 0.68]
    lambda_prior, lambda_prior_sigma = 2.0, 1  # nm; data barely constrain lambda

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, null_model(thickness_axes, *null_initial_guess))
    plt.title("Null Initial Guess")
    plt.show()

    def null_resid(p):
        r = (null_model(eels_thickness, *p) - HIM_intensity) / HIM_intensity_error
        return np.append(r, (p[2] - lambda_prior) / lambda_prior_sigma)  # Gaussian prior on lambda

    null_fit = optimize.least_squares(
        null_resid,
        null_initial_guess,
        bounds=([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]),  # lambda > 0: exp overflows if LM probes negative
        x_scale='jac',
    )
    null_opt = null_fit.x
    null_cov = np.linalg.inv(null_fit.jac.T @ null_fit.jac)  # absolute (residuals pre-scaled by sigma)
    chi2_p0, chi2_fit = (np.sum(null_resid(p)[:-1]**2) for p in (null_initial_guess, null_opt))
    dof = len(eels_thickness) - 3  # lambda is prior-dominated, ~not a free parameter

    print(f"Null fit\nOptimized:\n{null_opt}\n\nErrors:\n{np.sqrt(np.diag(null_cov))}\n\n"
          f"Covariance:\n{null_cov}\n\nchi2: p0={chi2_p0:.1f} fit={chi2_fit:.1f} "
          f"(reduced {chi2_fit/dof:.2f})\n\nMessage:\n{null_fit.message}")

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, null_model(thickness_axes, *null_opt))
    plt.title("Null Fit")
    plt.show()


    # Split lambda
    # Delta-electron momentum hypothesis: forward-deposited energy -> different effective
    # escape depths toward entrance (front) vs exit (back) surface
    def split_model(x_array, C_overall, C_fbratio, lambda_front, lambda_back, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        a_b = dt / lambda_back
        e_b, phi_b = np.exp(-a_b), -np.expm1(-a_b) / a_b
        c1, c0 = dt * (1 - phi_b) / a_b, dt * (phi_b - (1 - phi_b) / a_b)
        back, _ = signal.lfilter([c1, c0], [1, -e_b], s, zi=[-c1 * s[0]])
        a_f = dt / lambda_front
        e_f, phi_f = np.exp(-a_f), -np.expm1(-a_f) / a_f
        psi_f = (phi_f - e_f) / a_f
        cell = np.exp(-t[:-1] / lambda_front) * dt * ((phi_f - psi_f) * s[:-1] + psi_f * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        # each kernel unit-area so C's keep their meaning and lambdas stay decoupled from amplitude
        return np.interp(x_array, t, C_overall * (front / lambda_front + C_fbratio * back / lambda_back))

    split_initial_guess = [*null_opt[:2], null_opt[2], 5*null_opt[2], null_opt[3]]  # start from null fit

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, split_model(thickness_axes, *split_initial_guess))
    plt.title("Split Initial Guess")
    plt.show()

    def split_resid(p):
        r = (split_model(eels_thickness, *p) - HIM_intensity) / HIM_intensity_error
        return np.append(r, (p[2] - lambda_prior) / lambda_prior_sigma)  # prior on lambda_front only

    split_fit = optimize.least_squares(
        split_resid,
        split_initial_guess,
        bounds=([0, 0, 0.2, 0.2, 0.2], [np.inf, np.inf, 50, 50, 3]),
        x_scale='jac',
    )
    split_opt = split_fit.x
    split_cov = np.linalg.inv(split_fit.jac.T @ split_fit.jac)
    split_err = np.sqrt(np.diag(split_cov))
    split_corr = split_cov / np.outer(split_err, split_err)
    chi2_split = np.sum(split_resid(split_opt)[:-1]**2)  # strip the single prior row

    print(f"\n\n\nSplit fit\nOptimized:\n{split_opt}\n\nErrors:\n{split_err}\n\n"
          f"Correlation:\n{np.array2string(split_corr, precision=3, suppress_small=True)}\n\n"
          f"lambda_f - lambda_b = {split_opt[2]-split_opt[3]:.2f} +/- "
          f"{np.sqrt(split_cov[2,2] + split_cov[3,3] - 2*split_cov[2,3]):.2f}\n"
          f"chi2: null={chi2_fit:.1f} split={chi2_split:.1f} dchi2={chi2_fit-chi2_split:.2f} (1 dof)\n"
          f"eig(cov): {np.linalg.eigvalsh(split_cov)}\n\nMessage:\n{split_fit.message}")

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, null_model(thickness_axes, *null_opt), label='null')
    plt.plot(thickness_axes, split_model(thickness_axes, *split_opt), label='split')
    plt.legend()
    plt.title("Split-lambda Fit")
    plt.show()


    # D1p: M2 channeling-shift deposition, symmetric escape (equal f/b weight, single
    # amplitude), physical lambda >= 2 nm. The shift acts in raw SRIM coordinates
    # alongside the stretch: delta(z) = rho*lambda_d*(1 - exp(-z/lambda_d)), with
    # Jacobian 1 - rho*exp(-z/lambda_d) keeping the depth map monotone for rho < 1.
    # Escape stays in measured-nm space like null/split, so the 2 nm floor is stricter
    # than the report's raw-space bound (2 raw ~ 1.25 nm measured there).
    def d1p_model(x_array, A, lambda_phys, stretch, rho, lambda_d, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        z = t / stretch
        decay = np.exp(-z / lambda_d)
        s = (1 - rho * decay) * ionization_interp(z - rho * lambda_d * (1 - decay))
        a = dt / lambda_phys
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lambda_phys) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return np.interp(x_array, t, A * (front + back) / lambda_phys)

    d1p_shape_guess = [2.0, null_opt[3], 0.75, 16.6]  # lambda_phys, stretch, rho, lambda_d (report D1p)
    shape = d1p_model(eels_thickness, 1.0, *d1p_shape_guess)
    weight = 1 / HIM_intensity_error**2
    A_guess = np.sum(weight * shape * HIM_intensity) / np.sum(weight * shape**2)  # amplitude is linear
    d1p_initial_guess = [A_guess, *d1p_shape_guess]

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, d1p_model(thickness_axes, *d1p_initial_guess))
    plt.title("D1p Initial Guess")
    plt.show()

    def d1p_resid(p):
        r = (d1p_model(eels_thickness, *p) - HIM_intensity) / HIM_intensity_error
        return np.append(r, (p[1] - lambda_prior) / lambda_prior_sigma)  # Gaussian prior on lambda_phys

    d1p_fit = optimize.least_squares(
        d1p_resid,
        d1p_initial_guess,
        bounds=([0, 2.0, 0.2, 0, 5], [np.inf, 50, 3, 0.95, 400]),  # lambda_phys >= 2 nm physical
        x_scale='jac',
    )
    d1p_opt = d1p_fit.x
    d1p_cov = np.linalg.inv(d1p_fit.jac.T @ d1p_fit.jac)
    d1p_err = np.sqrt(np.diag(d1p_cov))
    d1p_corr = d1p_cov / np.outer(d1p_err, d1p_err)
    chi2_d1p = np.sum(d1p_resid(d1p_opt)[:-1]**2)

    delta_inf = d1p_opt[3] * d1p_opt[4]
    delta_inf_err = np.sqrt(d1p_opt[4]**2 * d1p_cov[3, 3] + d1p_opt[3]**2 * d1p_cov[4, 4]
                            + 2 * d1p_opt[3] * d1p_opt[4] * d1p_cov[3, 4])

    print(f"\n\n\nD1p fit\nOptimized:\n{d1p_opt}\n\nErrors:\n{d1p_err}\n\n"
          f"Correlation:\n{np.array2string(d1p_corr, precision=3, suppress_small=True)}\n\n"
          f"delta_inf = rho*lambda_d = {delta_inf:.2f} +/- {delta_inf_err:.2f} nm (report: 12.55 +/- 2.69)\n"
          f"chi2: D1p={chi2_d1p:.1f}\n\nMessage:\n{d1p_fit.message}")

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, d1p_model(thickness_axes, *d1p_opt))
    plt.title("D1p Fit")
    plt.show()


    # Head-to-head under identical raw-sigma weighting -- comparable within this table,
    # but not to the campaign's frozen-sigma_eff numbers in outputs/refined_report/.
    # dof counts prior-dominated lambdas as not free, matching the null dof convention.
    n_data = len(eels_thickness)
    print("\n\n\nModel comparison (raw-sigma weighting; prior rows excluded from chi2)")
    print(f"{'model':<6} {'chi2':>8} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    for name, chi2, k, k_eff in (
        ("null", chi2_fit, 4, 3),
        ("split", chi2_split, 5, 4),
        ("D1p", chi2_d1p, 5, 4),
    ):
        dof_m = n_data - k_eff
        print(f"{name:<6} {chi2:8.1f} {k:3d} {dof_m:4d} {chi2/dof_m:9.2f} {chi2 + 2*k:8.1f}")

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1, yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, null_model(thickness_axes, *null_opt), label='null')
    plt.plot(thickness_axes, split_model(thickness_axes, *split_opt), label='split')
    plt.plot(thickness_axes, d1p_model(thickness_axes, *d1p_opt), label='D1p')
    plt.legend()
    plt.title("Null vs Split vs D1p")
    plt.show()


def fit_and_plot_models_r02(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> None:
    """Fit escape-depth models with x-error-inflated weights, lambda_esc fixed as an
    assumed input (sweep LAMBDA_ESC 1.5/2.0/2.5 for the systematic)."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)
    LAMBDA_ESC = 2.0  # nm; data have no leverage below ~12 nm -> treat as input, not fit parameter

    ionization_interp = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False
    )

    def escape_terms(s, t, dt, lam):
        # unit-area exponential escape both ways; exact per-cell kernels, any lam vs dt
        a = dt / lam
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return front / lam, back / lam

    # Null: fixed symmetric escape depth, free front/back collection ratio, stretch
    def null_model(x_array, C_overall, C_fbratio, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        front, back = escape_terms(ionization_interp(t / stretch), t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    # Split: back face gets its own escape depth (nested: null at lambda_back = LAMBDA_ESC)
    def split_model(x_array, C_overall, C_fbratio, lambda_back, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
        _, back = escape_terms(s, t, dt, lambda_back)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    # D1p: conservative forward warp of deposition, symmetric escape, equal f/b weight.
    # Reparameterized to the identifiable pair (delta_inf = rho*lambda_d, rho); lambda_d derived.
    def d1p_model(x_array, A, stretch, delta_inf, rho, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        z, lambda_d = t / stretch, delta_inf / rho
        decay = np.exp(-z / lambda_d)
        s = (1 - rho * decay) * ionization_interp(z - delta_inf * (1 - decay))
        front, back = escape_terms(s, t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, A * (front + back))

    def wfit(model, p0, bounds, n_iter=3):
        # iteratively fold 10% thickness error into the weights via the local model slope
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2 + (slope * 0.1 * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, cov, np.sum(resid(fit.x)**2), sig, at_bound

    def report(name, opt, cov, chi2, at_bound):
        err = np.sqrt(np.diag(cov))
        corr = cov / np.outer(err, err)
        print(f"\n\n{name} fit\nOptimized:\n{opt}\n\nErrors:\n{err}\n\n"
              f"Correlation:\n{np.array2string(corr, precision=3, suppress_small=True)}\n\n"
              f"chi2 = {chi2:.1f}  (dof = {len(eels_thickness) - len(opt)}, "
              f"reduced = {chi2 / (len(eels_thickness) - len(opt)):.2f})")
        if at_bound.any():
            print(f"WARNING: params at bounds {np.where(at_bound)[0]} -- covariance invalid there")

    def plot(title, curves):
        plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1,
                     yerr=HIM_intensity_error, fmt='x')
        for label, y in curves:
            plt.plot(thickness_axes, y, label=label)
        if len(curves) > 1:
            plt.legend()
        plt.title(title)
        plt.show()

    # Null
    null_opt, null_cov, chi2_null, sig_null, nb = wfit(
        null_model, [0.02, 3.5, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Null", null_opt, null_cov, chi2_null, nb)
    plot("Null Fit", [("null", null_model(thickness_axes, *null_opt))])

    # Split
    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Split", split_opt, split_cov, chi2_split, sb)
    print(f"lambda_back - lambda_esc = {split_opt[2] - LAMBDA_ESC:.2f} +/- "
          f"{np.sqrt(split_cov[2, 2]):.2f}\n"
          f"dchi2 vs null = {chi2_null - chi2_split:.2f} (1 dof, nested)")

    # D1p
    d1p_opt, d1p_cov, chi2_d1p, _, db = wfit(
        d1p_model, [0.06, null_opt[2], 6.0, 0.7], ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    report("D1p", d1p_opt, d1p_cov, chi2_d1p, db)
    print(f"delta_inf = {d1p_opt[2]:.2f} +/- {np.sqrt(d1p_cov[2, 2]):.2f} nm, "
          f"lambda_d = delta_inf/rho = {d1p_opt[2] / d1p_opt[3]:.1f} nm")

    # Head-to-head on a COMMON weighting (null's converged sigma_eff), so chi2 is comparable
    print("\n\nModel comparison (common sigma_eff from null fit)")
    print(f"{'model':<6} {'chi2':>8} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    for name, model, opt in (("null", null_model, null_opt),
                             ("split", split_model, split_opt),
                             ("D1p", d1p_model, d1p_opt)):
        chi2 = np.sum(((model(eels_thickness, *opt) - HIM_intensity) / sig_null)**2)
        k = len(opt)
        print(f"{name:<6} {chi2:8.1f} {k:3d} {len(eels_thickness) - k:4d} "
              f"{chi2 / (len(eels_thickness) - k):9.2f} {chi2 + 2 * k:8.1f}")

    plot("Null vs Split vs D1p", [("null", null_model(thickness_axes, *null_opt)),
                                  ("split", split_model(thickness_axes, *split_opt)),
                                  ("D1p", d1p_model(thickness_axes, *d1p_opt))])
    if png_path:
        plt.savefig(png_path)

    return {
        "models": (("null", null_model, null_opt),
                   ("split", split_model, split_opt),
                   ("D1p", d1p_model, d1p_opt)),
        "sig_eff": sig_null,  # null's converged sigma_eff = the common weighting
    }


def fit_and_plot_models_r03(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> None:
    """r02 models on the thickness-aware deposition D(z; t) from the SRIM
    thickness sweep (DL/06_thickness_sweep_report.md). Depth and film thickness
    both map measured -> raw through the stretch: D(z/stretch, x/stretch).
    Because D now depends on each point's film thickness, the shared-grid
    cumulative trick is invalid; models integrate per point on [0, x]."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)
    LAMBDA_ESC = 2.0  # nm; data have no leverage below ~12 nm -> treat as input, not fit parameter

    # Finite-thickness deposition model (stage-1/AIC prints document its fit here)
    sweep = [(t, *load_ionization(p)) for t, p in discover_sweep(DEFAULT_SWEEP_DIR)]
    ionization_tdep, _ = build_ionization_model(ionization_thickness, ionization, sweep)

    def escape_terms(s, t, dt, lam):
        # unit-area exponential escape both ways; exact per-cell kernels, any lam vs dt
        a = dt / lam
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return front / lam, back / lam

    def per_point(x_array, deposition, combine, n=2048):
        # deposition(t, x) -> profile on grid t for a film of thickness x;
        # combine(front_end, back_end) -> model value at x
        x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            if x <= 0:
                continue
            t, dt = np.linspace(0, x, n, retstep=True)
            front, back = escape_terms(deposition(t, x), t, dt, LAMBDA_ESC)
            out[i] = combine(front[-1], back[-1])
        return out

    # Null: fixed symmetric escape depth, free front/back collection ratio, stretch
    def null_model(x_array, C_overall, C_fbratio, stretch):
        return per_point(
            x_array,
            lambda t, x: ionization_tdep(t / stretch, x / stretch),
            lambda f, b: C_overall * (f + C_fbratio * b),
        )

    # Split: back face gets its own escape depth (nested: null at lambda_back = LAMBDA_ESC)
    def split_model(x_array, C_overall, C_fbratio, lambda_back, stretch, n=2048):
        x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            if x <= 0:
                continue
            t, dt = np.linspace(0, x, n, retstep=True)
            s = ionization_tdep(t / stretch, x / stretch)
            front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
            _, back = escape_terms(s, t, dt, lambda_back)
            out[i] = C_overall * (front[-1] + C_fbratio * back[-1])
        return out

    # D1p: conservative forward warp of deposition, symmetric escape, equal f/b weight.
    # Reparameterized to the identifiable pair (delta_inf = rho*lambda_d, rho); lambda_d derived.
    def d1p_model(x_array, A, stretch, delta_inf, rho):
        lambda_d = delta_inf / rho

        def deposition(t, x):
            z = t / stretch
            decay = np.exp(-z / lambda_d)
            return (1 - rho * decay) * ionization_tdep(
                z - delta_inf * (1 - decay), x / stretch
            )

        return per_point(x_array, deposition, lambda f, b: A * (f + b))

    def wfit(model, p0, bounds, n_iter=3):
        # iteratively fold 10% thickness error into the weights via the local model slope
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2 + (slope * 0.1 * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, cov, np.sum(resid(fit.x)**2), sig, at_bound

    def report(name, opt, cov, chi2, at_bound):
        err = np.sqrt(np.diag(cov))
        corr = cov / np.outer(err, err)
        print(f"\n\n{name} fit\nOptimized:\n{opt}\n\nErrors:\n{err}\n\n"
              f"Correlation:\n{np.array2string(corr, precision=3, suppress_small=True)}\n\n"
              f"chi2 = {chi2:.1f}  (dof = {len(eels_thickness) - len(opt)}, "
              f"reduced = {chi2 / (len(eels_thickness) - len(opt)):.2f})")
        if at_bound.any():
            print(f"WARNING: params at bounds {np.where(at_bound)[0]} -- covariance invalid there")

    def plot(title, curves):
        plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1,
                     yerr=HIM_intensity_error, fmt='x')
        for label, y in curves:
            plt.plot(thickness_axes, y, label=label)
        if len(curves) > 1:
            plt.legend()
        plt.title(title)
        plt.show()

    # Null
    null_opt, null_cov, chi2_null, sig_null, nb = wfit(
        null_model, [0.02, 3.5, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Null (r03)", null_opt, null_cov, chi2_null, nb)
    plot("Null Fit (r03)", [("null", null_model(thickness_axes, *null_opt))])

    # Split
    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Split (r03)", split_opt, split_cov, chi2_split, sb)
    print(f"lambda_back - lambda_esc = {split_opt[2] - LAMBDA_ESC:.2f} +/- "
          f"{np.sqrt(split_cov[2, 2]):.2f}\n"
          f"dchi2 vs null = {chi2_null - chi2_split:.2f} (1 dof, nested)")

    # D1p
    d1p_opt, d1p_cov, chi2_d1p, _, db = wfit(
        d1p_model, [0.06, null_opt[2], 6.0, 0.7], ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    report("D1p (r03)", d1p_opt, d1p_cov, chi2_d1p, db)
    print(f"delta_inf = {d1p_opt[2]:.2f} +/- {np.sqrt(d1p_cov[2, 2]):.2f} nm, "
          f"lambda_d = delta_inf/rho = {d1p_opt[2] / d1p_opt[3]:.1f} nm")

    # Head-to-head on a COMMON weighting (null's converged sigma_eff), so chi2 is comparable
    print("\n\nModel comparison r03 (common sigma_eff from null fit)")
    print(f"{'model':<6} {'chi2':>8} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    for name, model, opt in (("null", null_model, null_opt),
                             ("split", split_model, split_opt),
                             ("D1p", d1p_model, d1p_opt)):
        chi2 = np.sum(((model(eels_thickness, *opt) - HIM_intensity) / sig_null)**2)
        k = len(opt)
        print(f"{name:<6} {chi2:8.1f} {k:3d} {len(eels_thickness) - k:4d} "
              f"{chi2 / (len(eels_thickness) - k):9.2f} {chi2 + 2 * k:8.1f}")

    plot("Null vs Split vs D1p (r03)", [("null", null_model(thickness_axes, *null_opt)),
                                        ("split", split_model(thickness_axes, *split_opt)),
                                        ("D1p", d1p_model(thickness_axes, *d1p_opt))])
    if png_path:
        plt.savefig(png_path)


def fit_and_plot_models_r04(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> None:
    """Channelling-model collection-ratio (C_fb) study, documenting the checks
    behind DL/07_paper_fit_report.md section 4 ("Why this model has no C_fb").

    Variants of the r02 channelling model I = A*(Y_f + C_fb*Y_b):
      1. equal weight (C_fb = 1, the paper-figure form)  -- baseline
      2. C_fb free                                        -- dchi2 ~ 0.3 for 1 dof,
         near-degenerate (|corr| ~ 1 between C_fb, A, rho), prefers C_fb ~ 1.6
      3. C_fb fixed to the ~3.0 the null/split fits measure (r02 convention)
         -- the warp dies (delta_inf -> 0), chi2 collapses onto the null
      4. C_fb fixed to the PHYSICAL collection ratio ~0.65 (r05 convention,
         = split C_fb * lambda_f / lambda_b; the r02 "3" bundles split's
         escape-depth ratio and is not transferable) -- channelling survives
         (dchi2 ~ +4 vs split) but strained: rho pins at the 0.95
         monotonicity bound and the fit is sensitive to the ratio's 1 sigma
         (chi2 17.0 at +1 sigma, 27.7 at -1 sigma).
      5. Effective C_fb by projection: the split family fitted to the
         noise-free equal-weight channelling curve lands at C_fb,eff ~ 0.65,
         lambda_b,eff ~ 9.8 nm with max mismatch ~0.005 in I_n (~sigma/5):
         the families are observationally degenerate on these data.

    Conclusion: the models are not distinguished by the collection ratio
    itself (both accept ~0.65-0.73); they differ in what carries the rest of
    the front/back imbalance -- long back-side transport (split) vs a
    near-maximal deposition warp at the monotonicity limit (channelling).
    """
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)
    LAMBDA_ESC = 2.0  # nm; same assumed input as r02

    ionization_interp = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False
    )

    def escape_terms(s, t, dt, lam):
        # unit-area exponential escape both ways; exact per-cell kernels, any lam vs dt
        a = dt / lam
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return front / lam, back / lam

    def null_model(x_array, C_overall, C_fbratio, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        front, back = escape_terms(ionization_interp(t / stretch), t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    def split_model(x_array, C_overall, C_fbratio, lambda_back, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
        _, back = escape_terms(s, t, dt, lambda_back)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    def d1p_cfb_model(x_array, A, C_fb, stretch, delta_inf, rho, n=8192):
        # r02 channelling warp with an explicit back-collection ratio C_fb
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        z, lambda_d = t / stretch, delta_inf / rho
        decay = np.exp(-z / lambda_d)
        s = (1 - rho * decay) * ionization_interp(z - delta_inf * (1 - decay))
        front, back = escape_terms(s, t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, A * (front + C_fb * back))

    def d1p_model(x_array, A, stretch, delta_inf, rho):
        return d1p_cfb_model(x_array, A, 1.0, stretch, delta_inf, rho)

    def wfit(model, p0, bounds, n_iter=3):
        # as r02, but pinv: the free-C_fb fit is intentionally near-degenerate
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2 + (slope * 0.1 * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.pinv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, cov, np.sum(resid(fit.x)**2), sig, at_bound

    def report(name, opt, cov, chi2, at_bound, names):
        err = np.sqrt(np.diag(cov))
        corr = cov / np.outer(err, err)
        print(f"\n\n{name}: chi2 = {chi2:.2f} (dof = {len(eels_thickness) - len(opt)}, "
              f"reduced = {chi2 / (len(eels_thickness) - len(opt)):.2f})")
        for nm, v, s_ in zip(names, opt, err):
            print(f"  {nm:<10} = {v:9.4f} +/- {s_:.4f}")
        print(f"  correlation:\n{np.array2string(corr, precision=2, suppress_small=True)}")
        if at_bound.any():
            print(f"  WARNING: params at bounds {[names[i] for i in np.where(at_bound)[0]]} "
                  f"-- covariance invalid there")

    # Reference fits: null and split provide the "instrumental" C_fb reading
    null_opt, null_cov, chi2_null, _, nb = wfit(
        null_model, [0.02, 3.5, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Null", null_opt, null_cov, chi2_null, nb, ["C_o", "C_fb", "s"])

    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Split", split_opt, split_cov, chi2_split, sb,
           ["C_o", "C_fb", "lambda_b", "s"])

    # 1. Baseline: equal-weight channelling (the paper-figure form)
    base_opt, base_cov, chi2_base, _, bb = wfit(
        d1p_model, [0.06, null_opt[2], 6.0, 0.7],
        ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    report("D1p equal-weight (baseline)", base_opt, base_cov, chi2_base, bb,
           ["A", "s", "delta_inf", "rho"])

    # 2. Free C_fb, from two starts to probe the degeneracy valley
    BOUNDS5 = ([0, 0, 0.2, 0, 0.05], [np.inf, np.inf, 3, 30, 0.95])
    NAMES5 = ["A", "C_fb", "s", "delta_inf", "rho"]
    free_a = wfit(d1p_cfb_model, [base_opt[0], 1.0, *base_opt[1:]], BOUNDS5)
    report("D1p free C_fb (start: warp fit)", free_a[0], free_a[1], free_a[2],
           free_a[4], NAMES5)
    free_b = wfit(d1p_cfb_model, [0.022, 3.0, 0.66, 0.5, 0.5], BOUNDS5)
    report("D1p free C_fb (start: null-like)", free_b[0], free_b[1], free_b[2],
           free_b[4], NAMES5)
    chi2_free = min(free_a[2], free_b[2])
    print(f"\nFree C_fb verdict: dchi2 vs equal-weight = {chi2_base - chi2_free:.2f} "
          f"(1 dof) -- unsupported; both starts converge to the same valley "
          f"(C_fb ~ {free_a[0][1]:.2f}), linearized errors unreliable there.")

    # 3./4. C_fb fixed: to the r02-convention null/split values (~3), and to
    # the physical collection ratio (r05 convention) with its +/- 1 sigma band
    cfb_phys = split_opt[1] * LAMBDA_ESC / split_opt[2]
    rel = np.sqrt(split_cov[1, 1] / split_opt[1]**2 + split_cov[2, 2] / split_opt[2]**2
                  - 2 * split_cov[1, 2] / (split_opt[1] * split_opt[2]))
    cfb_phys_err = cfb_phys * rel
    fixed_cases = (
        ("null-fitted, r02 conv.", null_opt[1]),
        ("split-fitted, r02 conv.", split_opt[1]),
        ("split collection ratio, r05 conv.", cfb_phys),
        ("collection ratio -1 sigma", cfb_phys - cfb_phys_err),
        ("collection ratio +1 sigma", cfb_phys + cfb_phys_err),
    )
    fixed_results = []
    for label, cfb in fixed_cases:
        def d1p_fixed(x_array, A, stretch, delta_inf, rho, _c=cfb):
            return d1p_cfb_model(x_array, A, _c, stretch, delta_inf, rho)

        f_opt, f_cov, f_chi2, _, fb = wfit(
            d1p_fixed, [0.06 / max(cfb, 0.5), 0.64, 10.0, 0.7],
            ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
        report(f"D1p with C_fb fixed = {cfb:.3f} ({label})",
               f_opt, f_cov, f_chi2, fb, ["A", "s", "delta_inf", "rho"])
        fixed_results.append((label, cfb, d1p_fixed, f_opt, f_chi2, fb.any()))
        print(f"  warp contribution: dchi2 vs null = {chi2_null - f_chi2:+.2f} "
              f"(2 dof); vs split (same k) = {f_chi2 - chi2_split:+.2f}")

    print("\n\nC_fb study summary (each fit under its own converged sigma_eff)")
    print(f"{'variant':<28} {'chi2':>7} {'k':>3}  note")
    print(f"{'null':<28} {chi2_null:7.1f} {3:3d}  C_fb = {null_opt[1]:.2f} "
          f"+/- {np.sqrt(null_cov[1, 1]):.2f}")
    print(f"{'split':<28} {chi2_split:7.1f} {4:3d}  C_fb = {split_opt[1]:.2f} "
          f"+/- {np.sqrt(split_cov[1, 1]):.2f}")
    print(f"{'D1p equal-weight':<28} {chi2_base:7.1f} {4:3d}  paper-figure form")
    print(f"{'D1p free C_fb':<28} {chi2_free:7.1f} {5:3d}  degenerate, "
          f"C_fb ~ {free_a[0][1]:.2f}")
    for label, cfb, _, _, f_chi2, pinned in fixed_results:
        if f_chi2 > 0.8 * chi2_null:
            note = "warp dead: collapses onto null"
        else:
            note = f"viable, dchi2 vs split = {f_chi2 - chi2_split:+.1f}"
        if pinned:
            note += "; rho at bound"
        print(f"{f'D1p C_fb = {cfb:.2f}':<28} {f_chi2:7.1f} {4:3d}  {note} ({label})")
    # 5. Effective C_fb by projection: fit the split family to the NOISE-FREE
    # equal-weight channelling curve at the measured thicknesses. The location
    # is the collection ratio a split analysis would report if channelling
    # were true; the residual measures how separable the families are.
    synth = d1p_model(eels_thickness, *base_opt)
    proj = optimize.least_squares(
        lambda p: (split_model(eels_thickness, *p) - synth) / HIM_intensity_error,
        [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        bounds=([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]),
        x_scale='jac', ftol=1e-12, xtol=1e-12)
    family_gap = split_model(eels_thickness, *proj.x) - synth
    print(f"\nEffective C_fb by projection (split family vs noise-free "
          f"channelling curve):\n"
          f"  C_fb,eff = {proj.x[1] * LAMBDA_ESC / proj.x[2]:.3f} (physical "
          f"convention), lambda_b,eff = {proj.x[2]:.2f} nm, s = {proj.x[3]:.3f}\n"
          f"  max family mismatch = {np.max(np.abs(family_gap)):.4f} in I_n "
          f"(median data sigma {np.median(HIM_intensity_error):.4f}); "
          f"expected family-discrimination dchi2 ~ "
          f"{np.sum((family_gap / HIM_intensity_error)**2):.1f}")

    print("\nConclusion: the r02-convention C_fb (~3) is NOT transferable (it "
          "bundles split's\nescape-depth ratio) and kills the warp. The physical "
          "collection ratio (~0.65,\nr05 convention) IS compatible with "
          "channelling -- at the cost of a near-maximal\nwarp (rho at the 0.95 "
          "monotonicity bound) and strong sensitivity to the ratio's\n1 sigma. "
          "The models differ in what carries the remaining imbalance: long "
          "back\ntransport (split) vs an extreme deposition warp (channelling) "
          "-- but by the\nprojection above they are observationally degenerate "
          "on these 15 points.")

    plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1,
                 yerr=HIM_intensity_error, fmt='x')
    plt.plot(thickness_axes, null_model(thickness_axes, *null_opt), label='null')
    plt.plot(thickness_axes, split_model(thickness_axes, *split_opt), label='split')
    plt.plot(thickness_axes, d1p_model(thickness_axes, *base_opt),
             label='D1p equal-weight')
    label, cfb, model_fx, opt_fx, _, _ = fixed_results[1]
    plt.plot(thickness_axes, model_fx(thickness_axes, *opt_fx), '--',
             label=f'D1p C_fb = {cfb:.2f} (collapses onto null)')
    label, cfb, model_fx, opt_fx, _, _ = fixed_results[2]
    plt.plot(thickness_axes, model_fx(thickness_axes, *opt_fx), ':',
             label=f'D1p C_fb = {cfb:.2f} (physical ratio)')
    plt.legend()
    plt.title("r04: channelling C_fb study")
    plt.show()
    if png_path:
        plt.savefig(png_path)


def fit_and_plot_models_r05(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> dict:
    """r02 with physically-normalized escape kernels: no 1/lambda in the yields.

    An SE born at a surface has the same escape odds regardless of its MFP;
    lambda only sets how fast those odds decay with depth. So the yield is the
    unnormalized sum Y = integral D(z/s) exp(-d/lambda) dz, and the physical
    yield grows with lambda. This is a pure reparameterization of r02 (same
    model manifold): identical chi2 and curves, with parameters mapped as
      C_o(r05)  = C_o(r02) / lambda_f
      C_fb(r05) = C_fb(r02) * lambda_f / lambda_b
    C_fb now reads directly as the back/front collection-and-generation ratio;
    for the split model expect ~0.65 (below 1, as back-side collection
    geometry demands). The null's C_fb is unchanged (lambda ratio = 1): the
    null still demands a ~3x back channel, which under this reading requires
    an implausibly large collection/generation asymmetry -- one more way to
    see it fail. See DL/07_paper_fit_report.md, Eq. (25) discussion."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)
    LAMBDA_ESC = 2.0  # nm; data have no leverage below ~12 nm -> treat as input, not fit parameter

    ionization_interp = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False
    )

    def escape_terms(s, t, dt, lam):
        # exponential escape both ways, exact per-cell kernels, any lam vs dt.
        # NO 1/lam normalization: surface escape odds are lam-independent, so
        # the yield amplitude grows with lam (see docstring).
        a = dt / lam
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return front, back

    # Null: fixed symmetric escape depth, free front/back collection ratio, stretch
    def null_model(x_array, C_overall, C_fbratio, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        front, back = escape_terms(ionization_interp(t / stretch), t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    # Split: back face gets its own escape depth (nested: null at lambda_back = LAMBDA_ESC)
    def split_model(x_array, C_overall, C_fbratio, lambda_back, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
        _, back = escape_terms(s, t, dt, lambda_back)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    # D1p: conservative forward warp of deposition, symmetric escape, equal f/b weight
    def d1p_model(x_array, A, stretch, delta_inf, rho, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        z, lambda_d = t / stretch, delta_inf / rho
        decay = np.exp(-z / lambda_d)
        s = (1 - rho * decay) * ionization_interp(z - delta_inf * (1 - decay))
        front, back = escape_terms(s, t, dt, LAMBDA_ESC)
        return np.interp(x_array, t, A * (front + back))

    def wfit(model, p0, bounds, n_iter=3):
        # iteratively fold 10% thickness error into the weights via the local model slope
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2 + (slope * 0.1 * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, cov, np.sum(resid(fit.x)**2), sig, at_bound

    def report(name, opt, cov, chi2, at_bound):
        err = np.sqrt(np.diag(cov))
        corr = cov / np.outer(err, err)
        print(f"\n\n{name} fit\nOptimized:\n{opt}\n\nErrors:\n{err}\n\n"
              f"Correlation:\n{np.array2string(corr, precision=3, suppress_small=True)}\n\n"
              f"chi2 = {chi2:.1f}  (dof = {len(eels_thickness) - len(opt)}, "
              f"reduced = {chi2 / (len(eels_thickness) - len(opt)):.2f})")
        if at_bound.any():
            print(f"WARNING: params at bounds {np.where(at_bound)[0]} -- covariance invalid there")

    def plot(title, curves):
        plt.errorbar(eels_thickness, HIM_intensity, xerr=eels_thickness*0.1,
                     yerr=HIM_intensity_error, fmt='x')
        for label, y in curves:
            plt.plot(thickness_axes, y, label=label)
        if len(curves) > 1:
            plt.legend()
        plt.title(title)
        plt.show()

    # Null
    null_opt, null_cov, chi2_null, sig_null, nb = wfit(
        null_model, [0.01, 3.5, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Null (r05)", null_opt, null_cov, chi2_null, nb)
    plot("Null Fit (r05)", [("null", null_model(thickness_axes, *null_opt))])

    # Split
    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Split (r05)", split_opt, split_cov, chi2_split, sb)
    print(f"C_fb is now the direct back/front collection-and-generation ratio: "
          f"{split_opt[1]:.3f} +/- {np.sqrt(split_cov[1, 1]):.3f} "
          f"(r02 equivalent: C_fb * lambda_b / lambda_f = "
          f"{split_opt[1] * split_opt[2] / LAMBDA_ESC:.2f})\n"
          f"lambda_back - lambda_esc = {split_opt[2] - LAMBDA_ESC:.2f} +/- "
          f"{np.sqrt(split_cov[2, 2]):.2f}\n"
          f"dchi2 vs null = {chi2_null - chi2_split:.2f} (1 dof, nested)")

    # D1p
    d1p_opt, d1p_cov, chi2_d1p, _, db = wfit(
        d1p_model, [0.03, null_opt[2], 6.0, 0.7], ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    report("D1p (r05)", d1p_opt, d1p_cov, chi2_d1p, db)
    print(f"delta_inf = {d1p_opt[2]:.2f} +/- {np.sqrt(d1p_cov[2, 2]):.2f} nm, "
          f"lambda_d = delta_inf/rho = {d1p_opt[2] / d1p_opt[3]:.1f} nm")

    # Head-to-head on a COMMON weighting (null's converged sigma_eff), so chi2 is comparable.
    # Pure reparameterization of r02: this table must match r02's exactly.
    print("\n\nModel comparison r05 (common sigma_eff from null fit; must match r02)")
    print(f"{'model':<6} {'chi2':>8} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    for name, model, opt in (("null", null_model, null_opt),
                             ("split", split_model, split_opt),
                             ("D1p", d1p_model, d1p_opt)):
        chi2 = np.sum(((model(eels_thickness, *opt) - HIM_intensity) / sig_null)**2)
        k = len(opt)
        print(f"{name:<6} {chi2:8.1f} {k:3d} {len(eels_thickness) - k:4d} "
              f"{chi2 / (len(eels_thickness) - k):9.2f} {chi2 + 2 * k:8.1f}")

    plot("Null vs Split vs D1p (r05)", [("null", null_model(thickness_axes, *null_opt)),
                                        ("split", split_model(thickness_axes, *split_opt)),
                                        ("D1p", d1p_model(thickness_axes, *d1p_opt))])
    if png_path:
        plt.savefig(png_path)

    return {
        "models": (("null", null_model, null_opt),
                   ("split", split_model, split_opt),
                   ("D1p", d1p_model, d1p_opt)),
        "sig_eff": sig_null,
    }


def fit_and_plot_models_r06(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    flake: np.ndarray | None = None,
    png_path: Path = None,
    show_figure: bool = True,
) -> None:
    """Free-lambda_esc study: what happens when the assumed 2 nm escape depth
    is fitted instead (physical kernels, production wfit weighting).

    Findings (2026-07-29):
      - Null with shared lambda FREE: lambda -> 8.3 +/- 1.0 nm and chi2 drops
        48.2 -> 15.5. The null's statistical failure is therefore conditional
        on the physical lambda_esc = 2 nm input. Its rescue is non-physical
        twice over: a shared escape depth ~4x the known SE escape depth on
        BOTH surfaces, and it retains C_fb ~ 3.1 (physical convention, equal
        lambdas) -- a back-collection advantage the geometry forbids.
      - Split with lambda_f also free is unidentified (C_fb = 9.6 +/- 8.5,
        lambda_f = 18 +/- 11): the data cannot constrain both escape depths
        and the collection ratio. This is why lambda_esc is an input.
    Paper wording must NOT claim that a free lambda_esc "converges near 2 nm";
    it converges to ~8 nm. The defensible statement: the symmetric model can
    only match the curve shape with a non-physical escape depth AND a
    non-physical collection ratio."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)

    ionization_interp = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False
    )

    def escape_terms(s, t, dt, lam):
        # physical normalization (no 1/lam), as in the production fits
        a = dt / lam
        e, phi = np.exp(-a), -np.expm1(-a) / a
        c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
        back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
        psi = (phi - e) / a
        cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
        front = np.concatenate(([0], np.cumsum(cell)))
        return front, back

    def null_free(x_array, C_o, C_fb, stretch, lam, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        front, back = escape_terms(ionization_interp(t / stretch), t, dt, lam)
        return np.interp(x_array, t, C_o * (front + C_fb * back))

    def split_free(x_array, C_o, C_fb, lam_f, lam_b, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        front, _ = escape_terms(s, t, dt, lam_f)
        _, back = escape_terms(s, t, dt, lam_b)
        return np.interp(x_array, t, C_o * (front + C_fb * back))

    def wfit(model, p0, bounds, n_iter=3):
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2 + (slope * 0.1 * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.pinv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, np.sqrt(np.diag(cov)), np.sum(resid(fit.x)**2), at_bound

    for label, model, p0, bounds, names in (
        ("Null, shared lambda FREE", null_free, [0.01, 3.0, 0.68, 2.0],
         ([0, 0, 0.2, 0.2], [np.inf, np.inf, 3, 50]),
         ["C_o", "C_fb", "s", "lambda"]),
        ("Split, lambda_f also FREE", split_free, [0.0111, 0.65, 2.0, 9.6, 0.625],
         ([0, 0, 0.2, 0.2, 0.2], [np.inf, np.inf, 50, 50, 3]),
         ["C_o", "C_fb", "lambda_f", "lambda_b", "s"]),
    ):
        opt, err, chi2, ab = wfit(model, p0, bounds)
        dof = len(eels_thickness) - len(opt)
        print(f"\n{label}: chi2 = {chi2:.1f} (dof {dof}, reduced {chi2 / dof:.2f})")
        for nm, v, e_ in zip(names, opt, err):
            print(f"  {nm:<8} = {v:9.4f} +/- {e_:.4f}")
        if ab.any():
            print(f"  WARNING: at bounds: {[names[i] for i in np.where(ab)[0]]}")
    print("\nFixed-lambda references (production): null chi2 = 48.2, split chi2 = 14.9")


PAPER_MODEL_STYLES = {
    # grey dashed strawman; blue/red + distinct linestyles are CVD-safe
    "null": {"label": "Symmetric Escape (null)", "color": "#686868", "linestyle": "--"},
    "split": {"label": "Asymmetric Escape", "color": "#2e0acc", "linestyle": "-"},
    "D1p": {"label": "Simplified Channelling", "color": "#cc0808", "linestyle": "-."},
}


def create_paper_figure(
    results: dict,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    pdf_path: Path = DEFAULT_PAPER_FIG,
    show_figure: bool = SHOW_PLOT,
) -> None:
    """Publication figure (APL single column): r02 fits over the pooled data,
    with normalized residuals below. Residuals use the null fit's converged
    sigma_eff for all models so the three panels' points are comparable."""
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()
    # column-format overrides on top of the house style: 11 pt text overpowers
    # a 3.37 in figure, and autolayout fights the shared-axis gridspec
    column_rc = {
        "figure.autolayout": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }
    with plt.rc_context(column_rc):
        figure, (ax_fit, ax_res) = plt.subplots(
            2, 1,
            figsize=(3.37, 3.6),  # 3.37 in = APL single-column width
            sharex=True,
            gridspec_kw={"height_ratios": (3, 1), "hspace": 0.07},
        )

        ax_fit.errorbar(
            eels_thickness,
            HIM_intensity,
            xerr=eels_thickness * 0.1,
            yerr=HIM_intensity_error,
            fmt="o",
            color="black",
            markerfacecolor="none",
            markersize=3.0,
            elinewidth=0.6,
            linestyle="none",
            zorder=3,
            label="Data",
        )

        curve_axis = np.linspace(0.0, 1.05 * np.max(eels_thickness), 400)
        sig_eff = results["sig_eff"]
        residual_extent = 0.0
        for name, model, opt in results["models"]:
            style = PAPER_MODEL_STYLES[name]
            ax_fit.plot(
                curve_axis,
                model(curve_axis, *opt),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                zorder=2,
                label=style["label"],
            )
            residuals = (HIM_intensity - model(eels_thickness, *opt)) / sig_eff
            residual_extent = max(residual_extent, np.max(np.abs(residuals)))
            ax_res.plot(
                eels_thickness,
                residuals,
                marker="o",
                markersize=2.2,
                linestyle="none",
                color=style["color"],
                zorder=2,
            )
        ax_res.axhline(0.0, color="#686868", linewidth=0.8, zorder=1)

        ax_fit.set_ylabel(r"$I_n$")
        ax_fit.set_xlim(0.0, curve_axis[-1])
        ax_fit.set_ylim(bottom=0.0)
        ax_fit.legend(loc="lower left", bbox_to_anchor=(0.10, 0.03))
        ax_res.set_xlabel(r"$t$ (nm)")
        ax_res.set_ylabel(r"$\Delta I_n/\sigma$")
        ax_res.set_ylim(-1.15 * residual_extent, 1.15 * residual_extent)

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(pdf_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
        print(f"Saved paper figure: {pdf_path} (+ .png preview)")

        if show_figure:
            plt.show()
        else:
            plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ioniz",
        type=Path,
        default=DEFAULT_IONIZ,
        help=f"SRIM IONIZ.txt output (default: {DEFAULT_IONIZ}).",
    )
    parser.add_argument(
        "--transmission",
        type=Path,
        default=DEFAULT_TRANSMISSION,
        help=f"Normalized transmission summary (default: {DEFAULT_TRANSMISSION}).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=DEFAULT_PNG,
        help=f"Output PNG (default: {DEFAULT_PNG}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure after saving it.",
    )
    return parser


def main() -> None:
    """Load the inputs as numpy arrays and create the comparison plot."""
    args = build_parser().parse_args()

    ionization_thickness, ionization = load_ionization(args.ioniz.resolve())
    eels_thickness, HIM_intensity, HIM_intensity_error, flake = load_him_data(
        args.transmission.resolve()
    )

##    create_plot(
##        ionization_thickness,
##        ionization,
##        eels_thickness,
##        HIM_intensity,
##        HIM_intensity_error,
##        flake,
##        png_path=args.png.resolve(),
##        show_figure=args.show,
##    )

    results = fit_and_plot_models_r04(
        ionization_thickness,
        ionization,
        eels_thickness,
        HIM_intensity,
        HIM_intensity_error,
        flake,
        png_path=args.png.resolve(),
        show_figure=args.show,
        )

    create_paper_figure(
        eels_thickness=eels_thickness,
        HIM_intensity=HIM_intensity,
        HIM_intensity_error=HIM_intensity_error,
        results=results,
        show_figure=args.show,
    )

    # Further exploration goes here -- everything above is plain numpy arrays.


if __name__ == "__main__":
    main()

"""Thickness-aware deposition cross-check for the paper model fits.

Refits the three models of scripts/paper_model_comparison.py (physical
escape kernels, identical weighting and bounds) with the finite-thickness
deposition model D(z; t) of docs/06_thickness_sweep_report.md in place of the
semi-infinite SRIM profile, and prints both variants side by side.

Both depth and film thickness map measured -> raw through the stretch:
S(z) = D(z/s; x/s) for a data point of measured thickness x. Because the
deposition then depends on each point's film thickness, the shared-grid
cumulative evaluation is invalid; every thickness is integrated on its own
[0, x] grid. The baseline (semi-infinite) variant runs through the same
per-point machinery so that the printed differences are purely the
deposition physics, not numerics.

Documented in docs/08_thickness_aware_fits_report.md.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from thickness_sweep_ionization import (  # noqa: E402  (needs REPOSITORY_ROOT)
    DEFAULT_SWEEP_DIR,
    build_ionization_model,
    discover_sweep,
)

STYLE_FILE = REPOSITORY_ROOT / "config" / "dlister_r01.mplstyle"
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
DEFAULT_FIG = REPOSITORY_ROOT / "outputs" / "thickness_aware_model_comparison.png"

ANGSTROM_PER_NM = 10.0
LAMBDA_ESC = 2.0  # nm; assumed input, as in the paper fits
THICKNESS_FRACTIONAL_ERROR = 0.10

MODEL_STYLES = {
    "null": {"label": "Symmetric Escape (null)", "color": "#686868", "linestyle": "--"},
    "split": {"label": "Asymmetric Escape", "color": "#2e0acc", "linestyle": "-"},
    "D1p": {"label": "Simplified Channelling", "color": "#cc0808", "linestyle": "-."},
}


def load_ionization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SRIM depth (nm) and the ionization-by-ions column (eV/Angstrom)."""
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


def load_him_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load thickness (nm), normalized intensity, and symmetrized 68% errors."""
    frame = pd.read_csv(path)
    required = {"thickness_nm", "normalized_median",
                "normalized_err_lower_68", "normalized_err_upper_68"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    order = np.argsort(frame["thickness_nm"].to_numpy(dtype=float))
    thickness = frame["thickness_nm"].to_numpy(dtype=float)[order]
    intensity = frame["normalized_median"].to_numpy(dtype=float)[order]
    error = (0.5 * (frame["normalized_err_lower_68"].to_numpy(dtype=float)
                    + frame["normalized_err_upper_68"].to_numpy(dtype=float)))[order]
    return thickness, intensity, error


def configure_style() -> None:
    """Apply David's style sheet (config/dlister_r01.mplstyle)."""
    matplotlib.style.use(str(STYLE_FILE))


def escape_terms(s, t, dt, lam):
    # exponential escape both ways; exact per-cell kernels, any lam vs dt.
    # Physical normalization (no 1/lam): surface escape odds are
    # lam-independent, so the yield amplitude grows with lam.
    a = dt / lam
    e, phi = np.exp(-a), -np.expm1(-a) / a
    c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
    back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
    psi = (phi - e) / a
    cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
    front = np.concatenate(([0], np.cumsum(cell)))
    return front, back


def build_models(dep):
    """The three paper models on a deposition lookup dep(z_raw, t_raw).

    dep takes raw-axis (SRIM) depth queries and the raw-axis film thickness;
    the baseline variant ignores the thickness argument. Every measured
    thickness x is integrated on its own [0, x] grid (n = 2048; the per-cell
    kernels are exact for any dt/lam)."""

    def null_model(x_array, C_o, C_fb, stretch, n=2048):
        x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            if x <= 0:
                continue
            t, dt = np.linspace(0, x, n, retstep=True)
            front, back = escape_terms(dep(t / stretch, x / stretch), t, dt, LAMBDA_ESC)
            out[i] = C_o * (front[-1] + C_fb * back[-1])
        return out

    def split_model(x_array, C_o, C_fb, lambda_back, stretch, n=2048):
        x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            if x <= 0:
                continue
            t, dt = np.linspace(0, x, n, retstep=True)
            s = dep(t / stretch, x / stretch)
            front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
            _, back = escape_terms(s, t, dt, lambda_back)
            out[i] = C_o * (front[-1] + C_fb * back[-1])
        return out

    def d1p_model(x_array, A, stretch, delta_inf, rho, n=2048):
        lambda_d = delta_inf / rho
        x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            if x <= 0:
                continue
            t, dt = np.linspace(0, x, n, retstep=True)
            z = t / stretch
            decay = np.exp(-z / lambda_d)
            s = (1 - rho * decay) * dep(z - delta_inf * (1 - decay), x / stretch)
            front, back = escape_terms(s, t, dt, LAMBDA_ESC)
            out[i] = A * (front[-1] + back[-1])
        return out

    return null_model, split_model, d1p_model


def run_fits(dep, label, eels_thickness, HIM_intensity, HIM_intensity_error,
             dense_max: float) -> dict:
    """Fit all three models on the given deposition lookup; print summaries."""
    thickness_axes = np.linspace(0, dense_max, 500)
    null_model, split_model, d1p_model = build_models(dep)

    def wfit(model, p0, bounds, n_iter=3):
        # iteratively fold 10% thickness error into the weights via the local model slope
        sig = HIM_intensity_error.copy()
        for _ in range(n_iter):
            resid = lambda p: (model(eels_thickness, *p) - HIM_intensity) / sig
            fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac',
                                         ftol=1e-12, xtol=1e-12)
            slope = np.interp(eels_thickness, thickness_axes,
                              np.gradient(model(thickness_axes, *fit.x), thickness_axes))
            sig = np.sqrt(HIM_intensity_error**2
                          + (slope * THICKNESS_FRACTIONAL_ERROR * eels_thickness)**2)
            p0 = fit.x
        cov = np.linalg.inv(fit.jac.T @ fit.jac)
        at_bound = (np.isclose(fit.x, bounds[0]) | np.isclose(fit.x, bounds[1]))
        return fit.x, cov, float(np.sum(resid(fit.x)**2)), sig, at_bound

    def report(title, names, opt, cov, chi2, at_bound):
        err = np.sqrt(np.diag(cov))
        dof = len(eels_thickness) - len(opt)
        print(f"\n{title} [{label}]: chi2 = {chi2:.1f}, dof = {dof}, "
              f"chi2/dof = {chi2 / dof:.2f}")
        for name, value, sigma in zip(names, opt, err):
            print(f"  {name:<9} = {value:9.4f} +/- {sigma:.4f}")
        if at_bound.any():
            print(f"  WARNING: params at bounds {np.where(at_bound)[0]} "
                  f"-- covariance invalid there")

    null_opt, null_cov, chi2_null, sig_null, nb = wfit(
        null_model, [0.01, 3.0, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Symmetric Escape (null)", ["C_o", "C_fb", "s"],
           null_opt, null_cov, chi2_null, nb)

    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Asymmetric Escape", ["C_o", "C_fb", "lambda_b", "s"],
           split_opt, split_cov, chi2_split, sb)
    print(f"  nested vs null: dchi2 = {chi2_null - chi2_split:.1f} (1 dof)")

    d1p_opt, d1p_cov, chi2_d1p, _, db = wfit(
        d1p_model, [0.03, null_opt[2], 6.0, 0.7],
        ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    report("Simplified Channelling", ["A", "s", "delta_inf", "rho"],
           d1p_opt, d1p_cov, chi2_d1p, db)

    print(f"\nModel comparison [{label}] (common sigma_eff from the null fit)")
    print(f"{'model':<24} {'chi2':>7} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    rows = {}
    for name, model, opt in (("null", null_model, null_opt),
                             ("split", split_model, split_opt),
                             ("D1p", d1p_model, d1p_opt)):
        chi2 = float(np.sum(((model(eels_thickness, *opt) - HIM_intensity) / sig_null)**2))
        k = len(opt)
        rows[name] = chi2
        print(f"{MODEL_STYLES[name]['label']:<24} {chi2:7.1f} {k:3d} "
              f"{len(eels_thickness) - k:4d} "
              f"{chi2 / (len(eels_thickness) - k):9.2f} {chi2 + 2 * k:8.1f}")

    return {
        "label": label,
        "models": (("null", null_model, null_opt, null_cov, chi2_null),
                   ("split", split_model, split_opt, split_cov, chi2_split),
                   ("D1p", d1p_model, d1p_opt, d1p_cov, chi2_d1p)),
        "sig_eff": sig_null,
        "common": rows,
    }


def shift_summary(base: dict, aware: dict) -> None:
    """Print per-parameter shifts between the two variants in units of sigma."""
    names = {"null": ["C_o", "C_fb", "s"],
             "split": ["C_o", "C_fb", "lambda_b", "s"],
             "D1p": ["A", "s", "delta_inf", "rho"]}
    print("\nParameter shifts, thickness-aware minus baseline (sigma = baseline error)")
    print(f"{'model':<7} {'param':<10} {'baseline':>10} {'t-aware':>10} {'shift/sigma':>12}")
    for (name, _, ob, cb, chib), (_, _, oa, _, chia) in zip(base["models"], aware["models"]):
        err = np.sqrt(np.diag(cb))
        for pname, vb, va, sig in zip(names[name], ob, oa, err):
            print(f"{name:<7} {pname:<10} {vb:10.4f} {va:10.4f} {(va - vb) / sig:12.2f}")
        print(f"{name:<7} {'chi2':<10} {chib:10.1f} {chia:10.1f}")


def create_figure(results: dict, eels_thickness, HIM_intensity, HIM_intensity_error,
                  fig_path: Path, show_figure: bool) -> None:
    """Paper-layout figure (fits + normalized residuals) for the t-aware fits."""
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()
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
            2, 1, figsize=(3.37, 3.6), sharex=True,
            gridspec_kw={"height_ratios": (3, 1), "hspace": 0.07})

        ax_fit.errorbar(
            eels_thickness, HIM_intensity,
            xerr=eels_thickness * THICKNESS_FRACTIONAL_ERROR,
            yerr=HIM_intensity_error,
            fmt="o", color="black", markerfacecolor="none", markersize=3.0,
            elinewidth=0.6, linestyle="none", zorder=3, label="Data")

        curve_axis = np.linspace(0.0, 1.05 * np.max(eels_thickness), 400)
        sig_eff = results["sig_eff"]
        residual_extent = 0.0
        for name, model, opt, _, _ in results["models"]:
            style = MODEL_STYLES[name]
            ax_fit.plot(curve_axis, model(curve_axis, *opt),
                        color=style["color"], linestyle=style["linestyle"],
                        linewidth=1.2, zorder=2, label=style["label"])
            residuals = (HIM_intensity - model(eels_thickness, *opt)) / sig_eff
            residual_extent = max(residual_extent, np.max(np.abs(residuals)))
            ax_res.plot(eels_thickness, residuals, marker="o", markersize=2.2,
                        linestyle="none", color=style["color"], zorder=2)
        ax_res.axhline(0.0, color="#686868", linewidth=0.8, zorder=1)

        ax_fit.set_ylabel(r"$I_n$")
        ax_fit.set_xlim(0.0, curve_axis[-1])
        ax_fit.set_ylim(bottom=0.0)
        ax_fit.legend(loc="lower left", bbox_to_anchor=(0.10, 0.03))
        ax_res.set_xlabel(r"$t$ (nm)")
        ax_res.set_ylabel(r"$\Delta I_n/\sigma$")
        ax_res.set_ylim(-1.15 * residual_extent, 1.15 * residual_extent)

        fig_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved thickness-aware comparison figure: {fig_path}")

        if show_figure:
            plt.show()
        else:
            plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ioniz", type=Path, default=DEFAULT_IONIZ,
                        help=f"Long-series SRIM IONIZ.txt (default: {DEFAULT_IONIZ}).")
    parser.add_argument("--transmission", type=Path, default=DEFAULT_TRANSMISSION,
                        help=f"Transmission summary (default: {DEFAULT_TRANSMISSION}).")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR,
                        help=f"Thickness-sweep root (default: {DEFAULT_SWEEP_DIR}).")
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG,
                        help=f"Output figure (default: {DEFAULT_FIG}).")
    parser.add_argument("--show", action="store_true",
                        help="Open the figure after saving it.")
    return parser


def main() -> None:
    """Fit both deposition variants, print the comparison, render the figure."""
    args = build_parser().parse_args()

    ionization_thickness, ionization = load_ionization(args.ioniz.resolve())
    eels_thickness, HIM_intensity, HIM_intensity_error = load_him_data(
        args.transmission.resolve())

    d_long = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False)

    sweep = [(t, *load_ionization(p))
             for t, p in discover_sweep(args.sweep_dir.resolve())]
    ionization_tdep, _ = build_ionization_model(
        ionization_thickness, ionization, sweep)

    dense_max = float(np.max(ionization_thickness))
    base = run_fits(lambda zq, x_raw: d_long(zq), "baseline: semi-infinite D",
                    eels_thickness, HIM_intensity, HIM_intensity_error, dense_max)
    aware = run_fits(ionization_tdep, "thickness-aware D(z; t)",
                     eels_thickness, HIM_intensity, HIM_intensity_error, dense_max)

    shift_summary(base, aware)
    create_figure(aware, eels_thickness, HIM_intensity, HIM_intensity_error,
                  fig_path=args.fig.resolve(), show_figure=args.show)


if __name__ == "__main__":
    main()

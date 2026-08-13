"""Production script for the paper model-comparison figure.

Fits the three models shown in the paper figure (docs/09_paper_fit_report_v2.md)
to the normalized HIM SE transmission data and renders the APL
single-column figure (fits + normalized residuals):

- Symmetric (null):        I = C_o [Y_f(lam_esc) + C_fb Y_b(lam_esc)], lam_esc = 2 nm
- Symmetric (lambda free): the same model with the shared escape depth fitted
- Asymmetric:              I = C_o [Y_f(lam_esc) + C_fb Y_b(lam_b)]

The free-lambda symmetric model is the null's only statistical rescue: it
reaches an acceptable fit, but at a shared escape depth ~4x the physical
value and a back/front collection ratio ~3 that the geometry forbids. The
asymmetric model fits equally well with every parameter physical. The
Simplified Channelling alternative is omitted: it is observationally
degenerate with the asymmetric model on these data (archived docs/07,
section 10.3); it remains in scripts/david_model_exploration.py and
scripts/thickness_aware_fits.py.

Yields use PHYSICAL escape kernels (no 1/lambda normalization): an SE born
at a surface has escape odds independent of its MFP, so the emitted count
grows with lambda and C_fb reads directly as the back/front
collection-and-generation ratio (expected < 1 from detector geometry).
The unit-area convention used during exploration (r02) gives identical
curves and chi2 with C_o -> C_o*lambda_f and C_fb -> C_fb*lambda_b/lambda_f.
The exploration history and all cross-checks live in
scripts/david_model_exploration.py (r02-r05) and docs/07_paper_fit_report.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import interpolate, optimize, signal, stats

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_PAPER_FIG = REPOSITORY_ROOT / "outputs" / "paper_model_comparison.pdf"

ANGSTROM_PER_NM = 10.0
LAMBDA_ESC = 2.0  # nm; assumed input (data have no leverage below ~12 nm)
THICKNESS_FRACTIONAL_ERROR = 0.10

MODEL_STYLES = {
    # grey/red/blue + distinct linestyles: validated CVD-safe trio
    "null": {"label": "Symmetric", "color": "#686868", "linestyle": "--"},
    "symfree": {"label": r"Symmetric ($\lambda$ free)",
                "color": "#cc0808", "linestyle": "-."},
    # dotted so the near-coincident red dash-dot curve stays visible beneath
    "split": {"label": "Asymmetric", "color": "#2e0acc", "linestyle": ":"},
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


def fit_models(
    ionization_thickness: np.ndarray,
    ionization: np.ndarray,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
) -> dict:
    """Run the three paper fits; return models, parameters, and sigma_eff."""
    thickness_axes = np.linspace(0, max(ionization_thickness), 500)

    ionization_interp = interpolate.interp1d(
        ionization_thickness, ionization, kind='cubic',
        fill_value=(ionization[0], ionization[-1]), bounds_error=False
    )

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

    def sym_model(x_array, C_overall, C_fbratio, stretch, lam, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        front, back = escape_terms(ionization_interp(t / stretch), t, dt, lam)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

    def null_model(x_array, C_overall, C_fbratio, stretch):
        return sym_model(x_array, C_overall, C_fbratio, stretch, LAMBDA_ESC)

    def split_model(x_array, C_overall, C_fbratio, lambda_back, stretch, n=8192):
        t, dt = np.linspace(0, np.max(x_array), n, retstep=True)
        s = ionization_interp(t / stretch)
        front, _ = escape_terms(s, t, dt, LAMBDA_ESC)
        _, back = escape_terms(s, t, dt, lambda_back)
        return np.interp(x_array, t, C_overall * (front + C_fbratio * back))

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
        corr = cov / np.outer(err, err)
        dof = len(eels_thickness) - len(opt)
        print(f"\n{title}: chi2 = {chi2:.1f}, dof = {dof}, "
              f"chi2/dof = {chi2 / dof:.2f}, "
              f"GoF p = {stats.chi2.sf(chi2, dof):.2g}")
        for name, value, sigma in zip(names, opt, err):
            print(f"  {name:<9} = {value:9.4f} +/- {sigma:.4f}")
        print(f"  correlation:\n"
              f"{np.array2string(corr, precision=3, suppress_small=True)}")
        if at_bound.any():
            print(f"  WARNING: params at bounds {np.where(at_bound)[0]} "
                  f"-- covariance invalid there")

    null_opt, null_cov, chi2_null, sig_null, nb = wfit(
        null_model, [0.01, 3.0, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    report("Symmetric (null, lambda_esc = 2 nm)", ["C_o", "C_fb", "s"],
           null_opt, null_cov, chi2_null, nb)

    symfree_opt, symfree_cov, chi2_symfree, _, fb = wfit(
        sym_model, [*null_opt, LAMBDA_ESC],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 3, 50]))
    report("Symmetric (lambda free)", ["C_o", "C_fb", "s", "lambda"],
           symfree_opt, symfree_cov, chi2_symfree, fb)
    print(f"  nested vs null: dchi2 = {chi2_null - chi2_symfree:.1f} (1 dof)\n"
          f"  the statistical rescue is non-physical twice over: lambda = "
          f"{symfree_opt[3]:.1f} nm shared\n  by BOTH surfaces (~4x the physical "
          f"SE escape depth), and C_fb = {symfree_opt[1]:.2f}\n  (equal lambdas, so "
          f"this is directly the back/front collection ratio -- geometry\n"
          f"  forbids > 1)")

    split_opt, split_cov, chi2_split, _, sb = wfit(
        split_model, [*null_opt[:2], LAMBDA_ESC, null_opt[2]],
        ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    report("Asymmetric", ["C_o", "C_fb", "lambda_b", "s"],
           split_opt, split_cov, chi2_split, sb)
    cfb, lam_b = split_opt[1], split_opt[2]
    print(f"  C_fb is the physical back/front collection-and-generation ratio; "
          f"unit-area (r02)\n  equivalent: C_fb * lambda_b / lambda_f = "
          f"{cfb * lam_b / LAMBDA_ESC:.2f} (see archived docs/07 section 10.1)\n"
          f"  lambda_b - lambda_esc = {lam_b - LAMBDA_ESC:.2f} +/- "
          f"{np.sqrt(split_cov[2, 2]):.2f} nm\n"
          f"  nested vs null: dchi2 = {chi2_null - chi2_split:.1f} (1 dof); "
          f"vs symmetric-free (same k, non-nested):\n  dAIC = "
          f"{chi2_split - chi2_symfree:+.1f}")

    print("\nModel comparison (common sigma_eff from the null fit)")
    print(f"{'model':<26} {'chi2':>7} {'k':>3} {'dof':>4} {'chi2/dof':>9} {'AIC':>8}")
    for name, model, opt in (("Symmetric (null)", null_model, null_opt),
                             ("Symmetric (lambda free)", sym_model, symfree_opt),
                             ("Asymmetric", split_model, split_opt)):
        chi2 = float(np.sum(((model(eels_thickness, *opt) - HIM_intensity) / sig_null)**2))
        k = len(opt)
        print(f"{name:<26} {chi2:7.1f} {k:3d} {len(eels_thickness) - k:4d} "
              f"{chi2 / (len(eels_thickness) - k):9.2f} {chi2 + 2 * k:8.1f}")

    return {
        "models": (("null", null_model, null_opt),
                   ("symfree", sym_model, symfree_opt),
                   ("split", split_model, split_opt)),
        "sig_eff": sig_null,
    }


def create_paper_figure(
    results: dict,
    eels_thickness: np.ndarray,
    HIM_intensity: np.ndarray,
    HIM_intensity_error: np.ndarray,
    pdf_path: Path = DEFAULT_PAPER_FIG,
    show_figure: bool = False,
) -> None:
    """APL single-column figure: fits on top, normalized residuals below.

    Residuals use the null fit's converged sigma_eff for all models so the
    three point sets are directly comparable."""
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
            xerr=eels_thickness * THICKNESS_FRACTIONAL_ERROR,
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
            style = MODEL_STYLES[name]
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
        print(f"\nSaved paper figure: {pdf_path} (+ .png preview)")

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
        "--pdf",
        type=Path,
        default=DEFAULT_PAPER_FIG,
        help=f"Output PDF (default: {DEFAULT_PAPER_FIG}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure after saving it.",
    )
    return parser


def main() -> None:
    """Fit the three models, print the paper numbers, render the figure."""
    args = build_parser().parse_args()

    ionization_thickness, ionization = load_ionization(args.ioniz.resolve())
    eels_thickness, HIM_intensity, HIM_intensity_error = load_him_data(
        args.transmission.resolve()
    )

    results = fit_models(
        ionization_thickness,
        ionization,
        eels_thickness,
        HIM_intensity,
        HIM_intensity_error,
    )
    create_paper_figure(
        results,
        eels_thickness,
        HIM_intensity,
        HIM_intensity_error,
        pdf_path=args.pdf.resolve(),
        show_figure=args.show,
    )


if __name__ == "__main__":
    main()

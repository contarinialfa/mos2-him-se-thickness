"""Compare the long SRIM ionization profile against the thickness sweep.

Overlays the quasi-infinite ("SRIM 30 keV - Monolayer Long") ``IONIZ. by
IONS`` depth profile on each run of the finite-thickness SRIM sweep
(``data/SRIM-Thickness-Series/<t>nm/IONIZ.txt``), one panel per simulated
thickness. Where the pair separates -- expected near the exit surface --
the missing downstream material is changing the deposition profile, which
is exactly the effect the long-profile-based models ignore.

Produces three figures: a per-thickness grid (one panel per sweep run), a
single overlay of all runs colored light->dark by simulated thickness, and
a validation figure for the finite-thickness ionization model built here:
D(z; t) = R(t, z/t) * D_long(z), where the long run carries the statistics
and the sweep only constrains the smoothed correction surface R.
"""

from __future__ import annotations

import argparse
import re
from math import ceil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate, optimize

SHOW_PLOT = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STYLE_FILE = REPOSITORY_ROOT / "config" / "dlister_r01.mplstyle"
DEFAULT_LONG_IONIZ = (
    REPOSITORY_ROOT
    / "data"
    / "SRIM 30 keV - Monolayer Long"
    / "IONIZ.txt"
)
DEFAULT_SWEEP_DIR = REPOSITORY_ROOT / "data" / "SRIM-Thickness-Series"
DEFAULT_PNG = REPOSITORY_ROOT / "outputs" / "thickness_sweep_ionization.png"
DEFAULT_OVERLAY_PNG = (
    REPOSITORY_ROOT / "outputs" / "thickness_sweep_ionization_overlay.png"
)
DEFAULT_INTERP_PNG = (
    REPOSITORY_ROOT / "outputs" / "thickness_sweep_interpolation.png"
)

ANGSTROM_PER_NM = 10.0

LONG_COLOR = "#555555"   # neutral reference grey, matches SRIM_COLOR elsewhere
SWEEP_COLOR = "#2e0acc"  # first color of the config/dlister_r01.mplstyle cycle


def load_ionization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SRIM depth (nm) and the ionization-by-ions column (eV/Angstrom).

    Same parser as scripts/david_model_exploration.py: the SRIM header and
    separators are skipped by keeping only lines that split into exactly
    three float-parseable tokens (depth, ionization by ions, by recoils).
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


def discover_sweep(sweep_dir: Path) -> list[tuple[float, Path]]:
    """Find the ``<t>nm/IONIZ.txt`` runs, sorted by simulated thickness."""
    runs: list[tuple[float, Path]] = []
    for folder in sweep_dir.iterdir():
        match = re.fullmatch(r"(\d+(?:\.\d+)?)nm", folder.name)
        ioniz = folder / "IONIZ.txt"
        if match and ioniz.is_file():
            runs.append((float(match.group(1)), ioniz))
    if not runs:
        raise ValueError(f"No <t>nm/IONIZ.txt runs found under {sweep_dir}.")
    return sorted(runs)


def configure_style() -> None:
    """Apply David's style sheet (config/dlister_r01.mplstyle)."""
    matplotlib.style.use(str(STYLE_FILE))


def create_plot(
    long_depth: np.ndarray,
    long_ioniz: np.ndarray,
    sweep: list[tuple[float, np.ndarray, np.ndarray]],
    png_path: Path = DEFAULT_PNG,
    show_figure: bool = SHOW_PLOT,
) -> None:
    """One panel per sweep thickness: finite-thickness run vs long profile.

    Each panel spans its own simulated thickness and autoscales in y so the
    exit-surface deviation stays visible even where it is a few percent.
    """
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()

    ncols = 3
    nrows = ceil(len(sweep) / ncols)
    figure, axes_grid = plt.subplots(
        nrows, ncols, figsize=(11, 3.0 * nrows), squeeze=False
    )
    axes_flat = axes_grid.ravel()

    for axes, (thickness_nm, depth, ioniz) in zip(axes_flat, sweep):
        axes.plot(
            long_depth,
            long_ioniz,
            color=LONG_COLOR,
            linestyle="--",
            linewidth=1.2,
            label="Monolayer Long",
        )
        axes.plot(
            depth,
            ioniz,
            color=SWEEP_COLOR,
            linewidth=1.5,
            label="Thickness series",
        )
        axes.set_xlim(0.0, thickness_nm)
        axes.set_title(f"{thickness_nm:g} nm")
    for axes in axes_flat[len(sweep):]:
        axes.set_visible(False)

    axes_flat[0].legend()
    figure.supxlabel(r"$t$ (nm)")
    figure.supylabel("SRIM ionization by ions (eV/Å)")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved thickness-sweep comparison: {png_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


def create_overlay_plot(
    long_depth: np.ndarray,
    long_ioniz: np.ndarray,
    sweep: list[tuple[float, np.ndarray, np.ndarray]],
    png_path: Path = DEFAULT_OVERLAY_PNG,
    show_figure: bool = SHOW_PLOT,
) -> None:
    """All sweep runs on one axes, colored light->dark by simulated thickness.

    Cividis is truncated at the light end (0.78 -> 0) so the thinnest run
    stays legible on white; identity is also carried by the legend and by
    each curve terminating at its own simulated thickness.
    """
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()

    figure, axes = plt.subplots(figsize=(9, 6))
    axes.plot(
        long_depth,
        long_ioniz,
        color=LONG_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Monolayer Long",
    )
    ramp = matplotlib.cm.cividis(np.linspace(0.78, 0.0, len(sweep)))
    for color, (thickness_nm, depth, ioniz) in zip(ramp, sweep):
        axes.plot(
            depth,
            ioniz,
            color=color,
            linewidth=1.5,
            label=f"{thickness_nm:g} nm",
        )

    axes.set_xlim(0.0, max(thickness for thickness, _, _ in sweep))
    axes.set_xlabel(r"$t$ (nm)")
    axes.set_ylabel("SRIM ionization by ions (eV/Å)")
    axes.legend(loc="lower left", ncol=2)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved thickness-sweep overlay: {png_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


def ramp_color(thickness_nm: float) -> tuple:
    """Truncated-cividis color for a thickness, log-mapped over 10-300 nm.

    Thin = light, thick = dark; the ramp stops at 0.78 so the lightest
    step stays legible on white (same truncation as the overlay figure).
    """
    span = np.log(thickness_nm / 10.0) / np.log(300.0 / 10.0)
    return matplotlib.cm.cividis(0.78 * (1.0 - float(np.clip(span, 0.0, 1.0))))


ARTIFACT_RATIO_FLOOR = 0.8  # observed body suppression never drops below ~0.92

# Deficit model: missing backscatter from the material beyond the exit face.
# d_inf(z, t) = A(z) * K(t - z) is the deficit relative to an infinite
# target: A(z) = A0 + A1 z + A2 z^2 grows with depth (slower ions scatter
# more) and the kernel K decays with distance-to-exit u = t - z, K(0) = 1.
# Returned ratios are normalized to the long run, R(z, t) =
# (1 - d_inf(z, t)) / (1 - d_inf(z, t_long)), so R = 1 at t_long exactly.
# theta = [A0, A1, A2, *kernel params]; the kernel family is chosen by AIC.
KERNELS = {
    "exponential": {
        "fn": lambda u, z, k: np.exp(-u / k[0]),
        "names": ["lam"],
        "init": [30.0],
        "lo": [1.0],
        "hi": [500.0],
    },
    "power": {
        "fn": lambda u, z, k: (1.0 + u / k[0]) ** (-k[1]),
        "names": ["lam", "p"],
        "init": [30.0, 2.0],
        "lo": [1.0, 0.1],
        "hi": [500.0, 10.0],
    },
    "double-exponential": {
        "fn": lambda u, z, k: (1.0 - k[2]) * np.exp(-u / k[0])
        + k[2] * np.exp(-u / k[1]),
        "names": ["lam1", "lam2", "w"],
        "init": [10.0, 80.0, 0.3],
        "lo": [1.0, 10.0, 0.0],
        "hi": [100.0, 1000.0, 1.0],
    },
    # lam falling with depth (slower ions -> shorter backscatter return),
    # suggested by the stage-1 per-run lam trend at 100-200 nm
    "exponential-zlam": {
        "fn": lambda u, z, k: np.exp(-u / (k[0] * np.exp(k[1] * z))),
        "names": ["lam0", "kappa"],
        "init": [25.0, -0.002],
        "lo": [1.0, -0.02],
        "hi": [500.0, 0.02],
    },
}


def _row_noise(row: np.ndarray) -> float:
    """Per-bin MC noise of a ratio row from second differences (trend-immune)."""
    return float(np.std(np.diff(row, 2)) / np.sqrt(6.0))


def _deficit_inf(z: np.ndarray, u: np.ndarray, theta, kernel_fn) -> np.ndarray:
    amplitude = theta[0] + theta[1] * z + theta[2] * z**2
    return amplitude * kernel_fn(u, z, theta[3:])


def _ratio_model(z, t, theta, kernel_fn, t_long: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (1.0 - _deficit_inf(z, t - z, theta, kernel_fn)) / (
        1.0 - _deficit_inf(z, t_long - z, theta, kernel_fn)
    )


def _fit_per_run(
    z: np.ndarray, thickness_nm: float, ratio: np.ndarray, sigma: float
) -> tuple[np.ndarray, float]:
    """Stage 1: per-run exponential deficit fit (family check + global seed)."""
    u = thickness_nm - z
    deficit = 1.0 - ratio

    def residual(p):
        return (p[0] * np.exp(-u / p[1]) - deficit) / sigma

    x0 = [max(float(np.max(deficit)), 0.02), max(thickness_nm / 2.0, 5.0)]
    fit = optimize.least_squares(
        residual, x0, bounds=([0.0, 0.5], [1.0, 1000.0]), x_scale="jac"
    )
    rms = float(np.sqrt(np.mean((fit.fun * sigma) ** 2)))
    return fit.x, rms


def _fit_global(
    points: tuple, kernel_name: str, t_long: float, x0=None
) -> tuple:
    """Stage 2: weighted least-squares fit of one kernel candidate to all runs."""
    z, t, ratio, sigma = points
    kernel = KERNELS[kernel_name]
    if x0 is None:
        x0 = [0.05, 5e-4, 1e-6, *kernel["init"]]
    bounds = (
        [0.0, -1e-2, -1e-4, *kernel["lo"]],
        [0.5, 1e-2, 1e-4, *kernel["hi"]],
    )

    def residual(theta):
        return (_ratio_model(z, t, theta, kernel["fn"], t_long) - ratio) / sigma

    fit = optimize.least_squares(residual, x0, bounds=bounds, x_scale="jac")
    return fit, float(np.sum(fit.fun**2))


def build_ionization_model(
    long_depth: np.ndarray,
    long_ioniz: np.ndarray,
    sweep: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple:
    """Build D(z; t) = R(t, z/t) * D_long(z) from the long run and the sweep.

    Returns ``(ionization, model)`` where ``ionization(z_array, thickness_nm)``
    evaluates the curve for an arbitrary flake thickness and ``model`` holds
    the pieces (ratio grid, smoothed surface, D_long) for validation/plots.

    Conventions: the final depth bin of each sweep run is dropped (SRIM
    boundary artifact); the long run is itself the exact t = 300 nm member
    (R = 1); the thickness argument is clamped to the [10, 300] nm data
    range while z always spans the requested film (zero outside [0, t]).
    """
    d_long = interpolate.interp1d(
        long_depth,
        long_ioniz,
        kind="cubic",
        fill_value=(long_ioniz[0], long_ioniz[-1]),
        bounds_error=False,
    )
    f_grid = np.arange(1, 100) / 100.0  # shared fractional grid; bin 100 dropped

    t_long = float(long_depth.max())
    t_list, raw_list, noise_list, stage1 = [], [], [], []
    z_pts, t_pts, r_pts, s_pts = [], [], [], []
    for thickness_nm, depth, ioniz in sweep:
        fraction = depth[:-1] / thickness_nm
        if len(fraction) != len(f_grid) or np.max(np.abs(fraction - f_grid)) > 0.005:
            raise ValueError(
                f"{thickness_nm:g} nm run is not on the common j/100 fractional grid."
            )
        raw = ioniz[:-1] / d_long(depth[:-1])
        valid = len(raw)
        while valid > 0 and raw[valid - 1] < ARTIFACT_RATIO_FLOOR:
            valid -= 1
        if valid < len(raw):
            print(
                f"  note: trimmed {len(raw) - valid} extra artifact bin(s) "
                f"from the {thickness_nm:g} nm run"
            )
        sigma = _row_noise(raw[:valid])
        z_run = depth[:-1][:valid]
        run_params, run_rms = _fit_per_run(z_run, thickness_nm, raw[:valid], sigma)
        stage1.append((thickness_nm, run_params[0], run_params[1], run_rms / sigma))

        t_list.append(thickness_nm)
        # SRIM bins carry a ~0.01 A offset; re-sample onto the exact grid
        raw_list.append(np.interp(f_grid, fraction, raw))
        noise_list.append(sigma)
        z_pts.append(z_run)
        t_pts.append(np.full(valid, thickness_nm))
        r_pts.append(raw[:valid])
        s_pts.append(np.full(valid, sigma))
    t_list.append(t_long)
    raw_list.append(np.ones_like(f_grid))
    noise_list.append(0.0)

    t_nodes = np.asarray(t_list)
    ratios = np.vstack(raw_list)
    noise = np.asarray(noise_list)
    points = tuple(np.concatenate(a) for a in (z_pts, t_pts, r_pts, s_pts))

    print("\nStage 1 -- per-run exponential deficit fits d(u) = a*exp(-u/lam):")
    print(f"  {'t (nm)':>7} {'a':>8} {'lam (nm)':>9} {'rms/sigma':>10}")
    for thickness_nm, a_run, lam_run, quality in stage1:
        print(f"  {thickness_nm:7g} {a_run:8.4f} {lam_run:9.2f} {quality:10.2f}")

    print("Stage 2 -- global kernel candidates (weighted fit to all runs):")
    n_points = len(points[0])
    fits = {}
    for name, kernel in KERNELS.items():
        fit, chi2 = _fit_global(points, name, t_long)
        k = 3 + len(kernel["names"])
        fits[name] = (fit, chi2, chi2 + 2 * k)
        print(
            f"  {name:<19} k={k}  chi2/dof = {chi2 / (n_points - k):6.3f}  "
            f"AIC = {chi2 + 2 * k:8.1f}"
        )
    kernel_name = min(fits, key=lambda name: fits[name][2])
    best_fit = fits[kernel_name][0]
    theta = best_fit.x
    kernel_fn = KERNELS[kernel_name]["fn"]
    covariance = np.linalg.pinv(best_fit.jac.T @ best_fit.jac)
    theta_err = np.sqrt(np.diag(covariance))
    names = ["A0", "A1", "A2", *KERNELS[kernel_name]["names"]]
    print(f"Selected kernel: {kernel_name}")
    for name, value, error in zip(names, theta, theta_err):
        print(f"  {name:<4} = {value: .4e} +/- {error:.1e}")

    def ratio_fn(z, t):
        return _ratio_model(z, t, theta, kernel_fn, t_long)

    def ionization(z_array: np.ndarray, thickness_nm: float) -> np.ndarray:
        z = np.asarray(z_array, dtype=float)
        t_shape = float(np.clip(thickness_nm, t_nodes[0], t_long))
        r = ratio_fn(z * (t_shape / thickness_nm), t_shape)  # fraction-preserving clamp
        return np.where((z >= 0.0) & (z <= thickness_nm), r * d_long(z), 0.0)

    model = {
        "t_nodes": t_nodes,
        "f_grid": f_grid,
        "ratios": ratios,
        "noise": noise,
        "points": points,
        "stage1": stage1,
        "kernel_name": kernel_name,
        "kernel_fn": kernel_fn,
        "theta": theta,
        "theta_err": theta_err,
        "ratio_fn": ratio_fn,
        "t_long": t_long,
        "d_long": d_long,
    }
    return ionization, model


def validate_model(model: dict) -> None:
    """Sanity checks, per-run agreement, and leave-one-out refits."""
    t_nodes, f_grid = model["t_nodes"], model["f_grid"]
    ratios, noise, ratio_fn = model["ratios"], model["noise"], model["ratio_fn"]
    t_long = model["t_long"]

    body = (f_grid >= 0.1) & (f_grid <= 0.8)
    r_long = ratio_fn(f_grid * t_long, t_long)
    r_thin = ratio_fn(f_grid * t_nodes[0], t_nodes[0])
    print(
        f"\nModel sanity: max |R({t_long:g} nm) - 1| = "
        f"{np.max(np.abs(r_long - 1.0)):.4f}; "
        f"body mean R({t_nodes[0]:g} nm) = {np.mean(r_thin[body]):.4f}; "
        f"thickness clamped to [{t_nodes[0]:g}, {t_long:g}] nm"
    )

    print("Per-run agreement (global model vs fit points; rms includes MC noise):")
    z_all, t_all, r_all, _ = model["points"]
    for i, thickness_nm in enumerate(t_nodes[:-1]):
        run = t_all == thickness_nm
        deviation = ratio_fn(z_all[run], thickness_nm) - r_all[run]
        print(
            f"  {thickness_nm:5.0f} nm: rms {np.sqrt(np.mean(deviation**2)):.4f}, "
            f"max |dev| {np.max(np.abs(deviation)):.4f}, "
            f"MC noise sigma {noise[i]:.4f}"
        )

    print("Leave-one-out validation (global model refitted without each run):")
    z_all, t_all, r_all, s_all = model["points"]
    for i in range(1, len(t_nodes) - 2):  # interior sweep runs
        thickness_nm = t_nodes[i]
        keep = t_all != thickness_nm
        fit, _ = _fit_global(
            (z_all[keep], t_all[keep], r_all[keep], s_all[keep]),
            model["kernel_name"],
            t_long,
            x0=model["theta"],
        )
        prediction = _ratio_model(
            f_grid * thickness_nm, thickness_nm, fit.x, model["kernel_fn"], t_long
        )
        deviation = prediction - ratios[i]
        print(
            f"  {thickness_nm:5.0f} nm: rms dev {np.sqrt(np.mean(deviation**2)):.4f}, "
            f"max |dev| {np.max(np.abs(deviation)):.4f}, "
            f"MC noise sigma {noise[i]:.4f}"
        )


def create_interpolation_plot(
    ionization,
    model: dict,
    png_path: Path = DEFAULT_INTERP_PNG,
    show_figure: bool = SHOW_PLOT,
) -> None:
    """Left: raw vs smoothed ratio curves. Right: interpolated D(z; t) family."""
    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style()

    figure, (ax_ratio, ax_family) = plt.subplots(1, 2, figsize=(12, 5))

    t_nodes, f_grid = model["t_nodes"], model["f_grid"]
    for thickness_nm, row in zip(t_nodes[:-1], model["ratios"]):
        color = ramp_color(thickness_nm)
        ax_ratio.plot(f_grid, row, color=color, linewidth=0.6, alpha=0.35)
        smooth = model["ratio_fn"](f_grid * thickness_nm, thickness_nm)
        ax_ratio.plot(
            f_grid, smooth, color=color, linewidth=1.8, label=f"{thickness_nm:g} nm"
        )
    ax_ratio.axhline(
        1.0, color=LONG_COLOR, linestyle="--", linewidth=1.2, label="Monolayer Long"
    )
    ax_ratio.set_xlabel(r"$z/t$ (fractional depth)")
    ax_ratio.set_ylabel(r"$R = D_\mathrm{finite}/D_\mathrm{long}$")
    ax_ratio.legend(loc="lower left", ncol=2, fontsize="small")

    z_long = np.linspace(0.0, t_nodes[-1], 600)
    ax_family.plot(
        z_long,
        model["d_long"](z_long),
        color=LONG_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Monolayer Long",
    )
    for thickness_nm in (15.0, 25.0, 45.0, 60.0, 85.0, 125.0, 175.0, 250.0):
        z = np.linspace(0.0, thickness_nm, 400)
        ax_family.plot(
            z,
            ionization(z, thickness_nm),
            color=ramp_color(thickness_nm),
            linewidth=1.5,
            label=f"{thickness_nm:g} nm",
        )
    ax_family.set_xlabel(r"$z$ (nm)")
    ax_family.set_ylabel("SRIM ionization by ions (eV/Å)")
    ax_family.legend(loc="lower left", ncol=2, fontsize="small")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved interpolation validation figure: {png_path}")

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
        default=DEFAULT_LONG_IONIZ,
        help=f"Long-series SRIM IONIZ.txt (default: {DEFAULT_LONG_IONIZ}).",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help=f"Thickness-sweep root directory (default: {DEFAULT_SWEEP_DIR}).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=DEFAULT_PNG,
        help=f"Output PNG for the panel grid (default: {DEFAULT_PNG}).",
    )
    parser.add_argument(
        "--overlay-png",
        type=Path,
        default=DEFAULT_OVERLAY_PNG,
        help=f"Output PNG for the overlay (default: {DEFAULT_OVERLAY_PNG}).",
    )
    parser.add_argument(
        "--interp-png",
        type=Path,
        default=DEFAULT_INTERP_PNG,
        help=f"Output PNG for the interpolation figure (default: {DEFAULT_INTERP_PNG}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure after saving it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    long_depth, long_ioniz = load_ionization(args.ioniz.resolve())
    print(
        f"Long profile: {len(long_depth)} points, "
        f"depth {long_depth.min():.2f}-{long_depth.max():.2f} nm"
    )

    sweep: list[tuple[float, np.ndarray, np.ndarray]] = []
    for thickness_nm, ioniz_path in discover_sweep(args.sweep_dir.resolve()):
        depth, ioniz = load_ionization(ioniz_path)
        sweep.append((thickness_nm, depth, ioniz))
        print(
            f"Sweep {thickness_nm:g} nm: {len(depth)} points, "
            f"depth {depth.min():.2f}-{depth.max():.2f} nm"
        )

    create_plot(
        long_depth,
        long_ioniz,
        sweep,
        png_path=args.png.resolve(),
        show_figure=args.show,
    )
    create_overlay_plot(
        long_depth,
        long_ioniz,
        sweep,
        png_path=args.overlay_png.resolve(),
        show_figure=args.show,
    )

    ionization, model = build_ionization_model(long_depth, long_ioniz, sweep)
    validate_model(model)
    create_interpolation_plot(
        ionization,
        model,
        png_path=args.interp_png.resolve(),
        show_figure=args.show,
    )


if __name__ == "__main__":
    main()

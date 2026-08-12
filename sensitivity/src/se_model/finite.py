"""Finite-thickness SRIM source profiles for the three paper models.

The committed SRIM sweep contains otherwise-identical 30 keV He simulations
for 10, 20, 30, 40, 50, 70, 100, 150, and 200 nm MoS2 targets, plus the
high-statistics 300 nm ``Monolayer Long`` run.  This module follows the
retained-bin and exit-boundary-artifact treatment documented in
``DL/thickness_sweep_ionization.py``:

* the final bin of every finite run is always discarded;
* additional trailing bins are discarded while their finite/long ratio is
  below ``ARTIFACT_RATIO_FLOOR``;
* the retained ratios are fitted by the same bounded deficit-kernel family,
  with the kernel selected by AIC;
* requested raw target thickness is clamped fraction-preservingly to the
  simulated 10--300 nm interval, and the source is zero outside the physical
  film and beyond the 300 nm long-profile support.

Both SRIM electronic-ionization columns are available.  ``include_recoils``
selects either ion ionization alone or the sum of ion and target-recoil
ionization; it is a source sensitivity, not a fitted efficiency parameter.

The public source is evaluated in measured coordinates as

``S(z; t, s) = D(z / s; t / s) / s``.

The Jacobian keeps the deposition units explicit.  Since every paper model
has a free overall amplitude, it changes that amplitude only, not the fitted
shape or measured-axis escape lengths.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize

from .data import load_srim
from .deposition import IonizationProfile
from .models import ModelPrediction, ModelSpec, ParameterValues, predict_model

EXPECTED_SWEEP_THICKNESSES_NM = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    70.0,
    100.0,
    150.0,
    200.0,
)
ARTIFACT_RATIO_FLOOR = 0.8
_FRACTIONAL_GRID = np.arange(1, 100, dtype=float) / 100.0


@dataclass(frozen=True)
class SweepRunDiagnostics:
    """Boundary treatment and Monte Carlo noise for one finite SRIM run."""

    thickness_nm: float
    retained_bins: int
    dropped_final_bins: int
    dropped_extra_artifact_bins: int
    ratio_noise: float


@dataclass(frozen=True)
class KernelFitDiagnostics:
    """Fit result for one finite-thickness deficit-kernel candidate."""

    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    chi2: float
    parameter_count: int
    aic: float
    success: bool
    at_bound: tuple[str, ...]
    observation_count: int
    degrees_of_freedom: int
    reduced_chi2: float
    attempted_starts: int
    successful_starts: int
    near_best_starts: int
    best_start_index: int


@dataclass(frozen=True)
class FiniteSourceDiagnostics:
    """Complete deterministic record of the source-profile construction."""

    include_recoils: bool
    long_ioniz_path: Path
    sweep_directory: Path
    raw_thickness_bounds_nm: tuple[float, float]
    runs: tuple[SweepRunDiagnostics, ...]
    kernel_fits: tuple[KernelFitDiagnostics, ...]
    selected_kernel: str


@dataclass(frozen=True)
class _KernelDefinition:
    names: tuple[str, ...]
    initial: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]


_KERNELS: dict[str, _KernelDefinition] = {
    "exponential": _KernelDefinition(
        names=("lambda_nm",),
        initial=(30.0,),
        lower=(1.0,),
        upper=(500.0,),
    ),
    "power": _KernelDefinition(
        names=("lambda_nm", "power"),
        initial=(30.0, 2.0),
        lower=(1.0, 0.1),
        upper=(500.0, 10.0),
    ),
    "double_exponential": _KernelDefinition(
        names=("lambda_1_nm", "lambda_2_nm", "weight"),
        initial=(10.0, 80.0, 0.3),
        lower=(1.0, 10.0, 0.0),
        upper=(100.0, 1000.0, 1.0),
    ),
    "exponential_z_lambda": _KernelDefinition(
        names=("lambda_0_nm", "kappa_per_nm"),
        initial=(25.0, -0.002),
        lower=(1.0, -0.02),
        upper=(500.0, 0.02),
    ),
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_origin(
    depth_nm: np.ndarray, deposition_ev_per_nm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the synthetic constant-origin node added by ``load_srim``."""

    if depth_nm[0] == 0.0:
        return depth_nm[1:].copy(), deposition_ev_per_nm[1:].copy()
    return depth_nm.copy(), deposition_ev_per_nm.copy()


def _kernel(
    name: str, distance_to_exit_nm: np.ndarray, depth_nm: np.ndarray, p: np.ndarray
) -> np.ndarray:
    if name == "exponential":
        return np.exp(-distance_to_exit_nm / p[0])
    if name == "power":
        return np.power(1.0 + distance_to_exit_nm / p[0], -p[1])
    if name == "double_exponential":
        return (1.0 - p[2]) * np.exp(-distance_to_exit_nm / p[0]) + p[2] * np.exp(
            -distance_to_exit_nm / p[1]
        )
    if name == "exponential_z_lambda":
        local_length = p[0] * np.exp(p[1] * depth_nm)
        return np.exp(-distance_to_exit_nm / local_length)
    raise KeyError(f"Unknown finite-thickness kernel {name!r}.")


def _deficit(
    depth_nm: np.ndarray,
    distance_to_exit_nm: np.ndarray,
    theta: np.ndarray,
    kernel_name: str,
) -> np.ndarray:
    amplitude = theta[0] + theta[1] * depth_nm + theta[2] * depth_nm**2
    return amplitude * _kernel(kernel_name, distance_to_exit_nm, depth_nm, theta[3:])


def _ratio_model(
    depth_nm: np.ndarray,
    thickness_nm: np.ndarray | float,
    theta: np.ndarray,
    kernel_name: str,
    long_thickness_nm: float,
) -> np.ndarray:
    depth = np.asarray(depth_nm, dtype=float)
    thickness = np.asarray(thickness_nm, dtype=float)
    numerator = 1.0 - _deficit(depth, thickness - depth, theta, kernel_name)
    denominator = 1.0 - _deficit(depth, long_thickness_nm - depth, theta, kernel_name)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError(
            "Finite-thickness correction has a nonpositive normalization."
        )
    return numerator / denominator


def _row_noise(ratio: np.ndarray) -> float:
    if ratio.size < 3:
        raise ValueError("At least three retained bins are needed to estimate noise.")
    sigma = float(np.std(np.diff(ratio, n=2)) / np.sqrt(6.0))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Finite-run ratio noise is nonpositive or non-finite.")
    return sigma


def _fit_kernel(
    name: str,
    depth_nm: np.ndarray,
    thickness_nm: np.ndarray,
    ratio: np.ndarray,
    sigma: np.ndarray,
    long_thickness_nm: float,
) -> tuple[np.ndarray, KernelFitDiagnostics]:
    definition = _KERNELS[name]
    initial = np.asarray((0.05, 5e-4, 1e-6, *definition.initial), dtype=float)
    lower = np.asarray((0.0, -1e-2, -1e-4, *definition.lower), dtype=float)
    upper = np.asarray((0.5, 1e-2, 1e-4, *definition.upper), dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        predicted = _ratio_model(depth_nm, thickness_nm, theta, name, long_thickness_nm)
        return (predicted - ratio) / sigma

    # The correction is nonlinear and some candidates contain broad length
    # bounds.  A single start can make the AIC ranking depend on its seed.
    # Use a deterministic, name-specific multistart cloud around the
    # documented anchor and retain every successful solution for diagnostics.
    start_count = 16
    digest = hashlib.sha256(f"finite-kernel-v1\0{name}".encode()).digest()
    generator = np.random.default_rng(
        int.from_bytes(digest[:8], "little", signed=False)
    )
    starts = [initial]
    span = upper - lower
    for _ in range(start_count - 1):
        random_point = lower + span * generator.beta(1.5, 1.5, size=initial.size)
        mixing = generator.uniform(0.35, 0.75)
        candidate = mixing * initial + (1.0 - mixing) * random_point
        starts.append(np.clip(candidate, lower, upper))

    successful: list[tuple[int, optimize.OptimizeResult, float]] = []
    for start_index, start in enumerate(starts):
        try:
            result = optimize.least_squares(
                residual,
                start,
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=20_000,
            )
        except (FloatingPointError, ValueError):
            continue
        chi2 = float(result.fun @ result.fun)
        if result.success and np.isfinite(chi2):
            successful.append((start_index, result, chi2))
    if not successful:
        raise RuntimeError(f"All finite-thickness kernel starts failed for {name}.")

    best_start_index, result, chi2 = min(successful, key=lambda item: item[2])
    parameter_count = result.x.size
    observation_count = ratio.size
    degrees_of_freedom = observation_count - parameter_count
    parameter_names = ("A0", "A1_per_nm", "A2_per_nm2", *definition.names)
    bound_tolerance = 1e-7 * np.maximum(1.0, np.abs(result.x))
    at_lower = np.abs(result.x - lower) <= bound_tolerance
    at_upper = np.abs(result.x - upper) <= bound_tolerance
    at_bound = tuple(
        parameter_names[index] for index in np.flatnonzero(at_lower | at_upper)
    )
    diagnostics = KernelFitDiagnostics(
        name=name,
        parameter_names=parameter_names,
        parameters=tuple(float(value) for value in result.x),
        chi2=chi2,
        parameter_count=parameter_count,
        aic=chi2 + 2.0 * parameter_count,
        success=bool(result.success),
        at_bound=at_bound,
        observation_count=observation_count,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi2=(
            chi2 / degrees_of_freedom
            if degrees_of_freedom > 0
            else float("nan")
        ),
        attempted_starts=start_count,
        successful_starts=len(successful),
        near_best_starts=sum(
            candidate_chi2 <= chi2 + 1e-6 * max(1.0, abs(chi2))
            for _, _, candidate_chi2 in successful
        ),
        best_start_index=best_start_index,
    )
    return result.x, diagnostics


@dataclass(frozen=True)
class FiniteThicknessSource:
    """Thickness-dependent SRIM deposition with a model-compatible interface.

    ``deposition`` and ``__call__`` return eV per measured nm.  The
    ``measured_components`` method deliberately matches the interface consumed
    by :func:`se_model.models.predict_model`, so all three existing
    :class:`~se_model.models.ModelSpec` families can be fitted without changing
    their implementation.
    """

    long_profile: IonizationProfile
    correction_parameters: np.ndarray
    correction_kernel: str
    raw_thickness_bounds_nm: tuple[float, float]
    diagnostics: FiniteSourceDiagnostics
    integration_step_nm: float = 0.05
    minimum_cells: int = 64
    maximum_cells: int = 8192

    def __post_init__(self) -> None:
        theta = np.asarray(self.correction_parameters, dtype=float)
        if theta.ndim != 1 or theta.size < 4 or not np.all(np.isfinite(theta)):
            raise ValueError("correction_parameters must be a finite vector.")
        if self.correction_kernel not in _KERNELS:
            raise ValueError(f"Unknown correction kernel {self.correction_kernel!r}.")
        minimum, maximum = self.raw_thickness_bounds_nm
        if not 0.0 < minimum < maximum:
            raise ValueError("raw_thickness_bounds_nm must be positive and ordered.")
        if not np.isfinite(self.integration_step_nm) or self.integration_step_nm <= 0.0:
            raise ValueError("integration_step_nm must be finite and positive.")
        if self.minimum_cells < 2 or self.maximum_cells < self.minimum_cells:
            raise ValueError("Integration-cell bounds are invalid.")
        object.__setattr__(self, "correction_parameters", theta.copy())

    @property
    def include_recoils(self) -> bool:
        return self.diagnostics.include_recoils

    @property
    def integration_cells(self) -> int:
        """Fixed integration count used for every model evaluation.

        Holding the grid size fixed is necessary for a smooth numerical
        derivative in the errors-in-variables likelihood. Choosing
        ``ceil(max(t) / step)`` separately for each call creates a grid jump
        when a perturbed thickness crosses a cell boundary.
        """

        target = int(
            np.ceil(self.raw_thickness_bounds_nm[1] / self.integration_step_nm)
        )
        return min(max(target, self.minimum_cells), self.maximum_cells)

    def ratio_raw(
        self,
        depth_raw_nm: np.ndarray | float,
        thickness_raw_nm: np.ndarray | float,
    ) -> np.ndarray:
        """Return the bounded finite/long correction on the raw SRIM axis."""

        depth, thickness = np.broadcast_arrays(
            np.asarray(depth_raw_nm, dtype=float),
            np.asarray(thickness_raw_nm, dtype=float),
        )
        if (
            not np.all(np.isfinite(depth))
            or not np.all(np.isfinite(thickness))
            or np.any(depth < 0.0)
            or np.any(thickness < 0.0)
        ):
            raise ValueError("Raw depth and thickness must be finite and nonnegative.")

        result = np.zeros_like(depth, dtype=float)
        inside = (thickness > 0.0) & (depth <= thickness)
        if not np.any(inside):
            return result

        minimum, maximum = self.raw_thickness_bounds_nm
        requested_t = thickness[inside]
        shape_t = np.clip(requested_t, minimum, maximum)
        shape_z = depth[inside] * shape_t / requested_t
        ratio = _ratio_model(
            shape_z,
            shape_t,
            self.correction_parameters,
            self.correction_kernel,
            maximum,
        )
        if not np.all(np.isfinite(ratio)):
            raise FloatingPointError("Finite-thickness ratio is non-finite.")
        tolerance = 5e-8
        if np.any(ratio < -tolerance) or np.any(ratio > 1.0 + tolerance):
            raise FloatingPointError(
                "Finite-thickness correction left its physical [0, 1] bounds."
            )
        result[inside] = np.clip(ratio, 0.0, 1.0)
        return result

    def deposition_raw(
        self,
        depth_raw_nm: np.ndarray | float,
        thickness_raw_nm: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate eV/raw-nm deposition on the SRIM coordinate system."""

        depth, thickness = np.broadcast_arrays(
            np.asarray(depth_raw_nm, dtype=float),
            np.asarray(thickness_raw_nm, dtype=float),
        )
        ratio = self.ratio_raw(depth, thickness)
        source = self.long_profile.evaluate(depth)
        return np.where(depth <= thickness, ratio * source, 0.0)

    def deposition(
        self,
        depth_measured_nm: np.ndarray | float,
        thickness_measured_nm: np.ndarray | float,
        scale: float,
    ) -> np.ndarray:
        """Evaluate ``D(z/s; t/s)/s`` in eV per measured nm."""

        scale_value = float(scale)
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("scale must be finite and positive.")
        depth, thickness = np.broadcast_arrays(
            np.asarray(depth_measured_nm, dtype=float),
            np.asarray(thickness_measured_nm, dtype=float),
        )
        if (
            not np.all(np.isfinite(depth))
            or not np.all(np.isfinite(thickness))
            or np.any(depth < 0.0)
            or np.any(thickness < 0.0)
        ):
            raise ValueError(
                "Measured depth and thickness must be finite and nonnegative."
            )
        return (
            self.deposition_raw(depth / scale_value, thickness / scale_value)
            / scale_value
        )

    def __call__(
        self,
        depth_measured_nm: np.ndarray | float,
        thickness_measured_nm: np.ndarray | float,
        scale: float,
    ) -> np.ndarray:
        return self.deposition(depth_measured_nm, thickness_measured_nm, scale)

    def measured_components(
        self,
        thickness_nm: np.ndarray | float,
        scale: float,
        lambda_front_nm: float,
        lambda_back_nm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return front/back escape integrals for one or more film thicknesses."""

        thickness = np.asarray(thickness_nm, dtype=float)
        if not np.all(np.isfinite(thickness)) or np.any(thickness < 0.0):
            raise ValueError("thickness_nm must be finite and nonnegative.")
        scale_value = float(scale)
        lambda_front = float(lambda_front_nm)
        lambda_back = float(lambda_back_nm)
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("scale must be finite and positive.")
        if (
            not np.isfinite(lambda_front)
            or lambda_front <= 0.0
            or not np.isfinite(lambda_back)
            or lambda_back <= 0.0
        ):
            raise ValueError("Escape lengths must be finite and positive.")

        flat = thickness.reshape(-1)
        front = np.zeros_like(flat)
        back = np.zeros_like(flat)
        nonzero = flat > 0.0
        if np.any(nonzero):
            selected = flat[nonzero]
            cells = self.integration_cells
            fraction = np.linspace(0.0, 1.0, cells + 1)
            grid = selected[:, None] * fraction[None, :]
            deposition = self.deposition(grid, selected[:, None], scale_value)
            widths = selected / cells

            def coefficients(
                escape_length: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                a = widths / escape_length
                decay = np.exp(-a)
                phi = -np.expm1(-a) / a
                psi = (phi - decay) / a
                small = a < 1e-4
                if np.any(small):
                    value = a[small]
                    phi[small] = (
                        1.0
                        - value / 2.0
                        + value**2 / 6.0
                        - value**3 / 24.0
                        + value**4 / 120.0
                    )
                    psi[small] = (
                        0.5
                        - value / 3.0
                        + value**2 / 8.0
                        - value**3 / 30.0
                        + value**4 / 144.0
                    )
                return decay, phi, psi

            _, phi_front, psi_front = coefficients(lambda_front)
            _, phi_back, psi_back = coefficients(lambda_back)
            left = deposition[:, :-1]
            right = deposition[:, 1:]

            front_cells = (
                np.exp(-grid[:, :-1] / lambda_front)
                * widths[:, None]
                * ((phi_front - psi_front)[:, None] * left + psi_front[:, None] * right)
            )
            back_cells = (
                np.exp(-(selected[:, None] - grid[:, 1:]) / lambda_back)
                * widths[:, None]
                * ((phi_back - psi_back)[:, None] * right + psi_back[:, None] * left)
            )
            front[nonzero] = np.sum(front_cells, axis=1)
            back[nonzero] = np.sum(back_cells, axis=1)
        return front.reshape(thickness.shape), back.reshape(thickness.shape)


def load_finite_source(
    *,
    repository_root: Path | str | None = None,
    long_ioniz_path: Path | str | None = None,
    sweep_directory: Path | str | None = None,
    include_recoils: bool = False,
    integration_step_nm: float = 0.05,
    minimum_cells: int = 64,
    maximum_cells: int = 8192,
) -> FiniteThicknessSource:
    """Load the committed nine-run sweep and build a bounded finite source."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else _repository_root()
    )
    long_path = (
        Path(long_ioniz_path).resolve()
        if long_ioniz_path is not None
        else root / "data" / "SRIM 30 keV - Monolayer Long" / "IONIZ.txt"
    )
    sweep_dir = (
        Path(sweep_directory).resolve()
        if sweep_directory is not None
        else root / "data" / "SRIM-Thickness-Series"
    )
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing long SRIM profile: {long_path}")
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Missing SRIM thickness sweep: {sweep_dir}")

    long_z, long_deposition = load_srim(long_path, include_recoils=include_recoils)
    long_profile = IonizationProfile(long_z, long_deposition)
    long_raw_z, long_raw_deposition = _strip_origin(long_z, long_deposition)
    long_thickness = float(long_raw_z[-1])

    point_z: list[np.ndarray] = []
    point_t: list[np.ndarray] = []
    point_ratio: list[np.ndarray] = []
    point_sigma: list[np.ndarray] = []
    run_diagnostics: list[SweepRunDiagnostics] = []

    for thickness in EXPECTED_SWEEP_THICKNESSES_NM:
        folder = sweep_dir / f"{thickness:g}nm"
        path = folder / "IONIZ.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing committed sweep run: {path}")
        run_z_with_origin, run_d_with_origin = load_srim(
            path, include_recoils=include_recoils
        )
        run_z, run_deposition = _strip_origin(run_z_with_origin, run_d_with_origin)
        if run_z.size != 100:
            raise ValueError(f"{path}: expected 100 SRIM bins, found {run_z.size}.")

        body_z = run_z[:-1]
        body_deposition = run_deposition[:-1]
        fraction = body_z / thickness
        if (
            fraction.size != _FRACTIONAL_GRID.size
            or np.max(np.abs(fraction - _FRACTIONAL_GRID)) > 0.005
        ):
            raise ValueError(
                f"{path}: bins are not on the common j/100 fractional grid."
            )
        long_at_run = np.interp(
            body_z,
            long_raw_z,
            long_raw_deposition,
            left=long_raw_deposition[0],
            right=0.0,
        )
        if np.any(long_at_run <= 0.0):
            raise ValueError(f"{path}: long-profile denominator is nonpositive.")
        ratios = body_deposition / long_at_run

        valid = ratios.size
        while valid > 0 and ratios[valid - 1] < ARTIFACT_RATIO_FLOOR:
            valid -= 1
        if valid < 3:
            raise ValueError(f"{path}: fewer than three retained physical bins.")
        retained_ratio = ratios[:valid]
        sigma = _row_noise(retained_ratio)
        run_diagnostics.append(
            SweepRunDiagnostics(
                thickness_nm=thickness,
                retained_bins=valid,
                dropped_final_bins=1,
                dropped_extra_artifact_bins=ratios.size - valid,
                ratio_noise=sigma,
            )
        )
        point_z.append(body_z[:valid])
        point_t.append(np.full(valid, thickness))
        point_ratio.append(retained_ratio)
        point_sigma.append(np.full(valid, sigma))

    z_all = np.concatenate(point_z)
    t_all = np.concatenate(point_t)
    ratio_all = np.concatenate(point_ratio)
    sigma_all = np.concatenate(point_sigma)

    fitted: dict[str, tuple[np.ndarray, KernelFitDiagnostics]] = {}
    for name in _KERNELS:
        fitted[name] = _fit_kernel(
            name,
            z_all,
            t_all,
            ratio_all,
            sigma_all,
            long_thickness,
        )
    selected_name = min(fitted, key=lambda name: fitted[name][1].aic)
    selected_theta = fitted[selected_name][0]

    # Validate the selected correction throughout its supported 10--300 nm
    # domain rather than only at the retained fit points.
    validation_t = np.linspace(EXPECTED_SWEEP_THICKNESSES_NM[0], long_thickness, 41)
    validation_fraction = np.linspace(0.0, 1.0, 101)
    mesh_t, mesh_f = np.meshgrid(validation_t, validation_fraction, indexing="ij")
    validation_z = mesh_t * mesh_f
    validation_ratio = _ratio_model(
        validation_z,
        mesh_t,
        selected_theta,
        selected_name,
        long_thickness,
    )
    tolerance = 5e-8
    if (
        not np.all(np.isfinite(validation_ratio))
        or np.any(validation_ratio < -tolerance)
        or np.any(validation_ratio > 1.0 + tolerance)
    ):
        raise RuntimeError(
            "Selected finite-thickness correction is not bounded on its "
            "supported domain."
        )

    diagnostics = FiniteSourceDiagnostics(
        include_recoils=bool(include_recoils),
        long_ioniz_path=long_path,
        sweep_directory=sweep_dir,
        raw_thickness_bounds_nm=(
            EXPECTED_SWEEP_THICKNESSES_NM[0],
            long_thickness,
        ),
        runs=tuple(run_diagnostics),
        kernel_fits=tuple(fitted[name][1] for name in _KERNELS),
        selected_kernel=selected_name,
    )
    return FiniteThicknessSource(
        long_profile=long_profile,
        correction_parameters=selected_theta,
        correction_kernel=selected_name,
        raw_thickness_bounds_nm=diagnostics.raw_thickness_bounds_nm,
        diagnostics=diagnostics,
        integration_step_nm=integration_step_nm,
        minimum_cells=minimum_cells,
        maximum_cells=maximum_cells,
    )


def predict(
    spec: ModelSpec,
    source: FiniteThicknessSource,
    x: np.ndarray | float,
    params: ParameterValues,
    *,
    return_components: bool = False,
) -> np.ndarray | ModelPrediction:
    """Evaluate any existing paper ``ModelSpec`` with a finite source.

    The signature mirrors :func:`se_model.models.predict_model`, allowing an
    orchestration script to swap semi-infinite and finite sources without
    changing parameter order or model-family logic.
    """

    return predict_model(
        spec,
        source,  # type: ignore[arg-type] -- intentional compatible protocol
        x,
        params,
        return_components=return_components,
    )


def finite_source_metadata(source: FiniteThicknessSource) -> Mapping[str, object]:
    """Return a compact serialization-friendly provenance mapping."""

    selected = next(
        fit
        for fit in source.diagnostics.kernel_fits
        if fit.name == source.diagnostics.selected_kernel
    )
    return {
        "include_recoils": source.include_recoils,
        "long_ioniz_path": str(source.diagnostics.long_ioniz_path),
        "sweep_directory": str(source.diagnostics.sweep_directory),
        "raw_thickness_bounds_nm": list(source.raw_thickness_bounds_nm),
        "expected_sweep_thicknesses_nm": list(EXPECTED_SWEEP_THICKNESSES_NM),
        "artifact_ratio_floor": ARTIFACT_RATIO_FLOOR,
        "selected_kernel": selected.name,
        "selected_parameter_names": list(selected.parameter_names),
        "selected_parameters": list(selected.parameters),
        "selected_chi2": selected.chi2,
        "selected_aic": selected.aic,
        "selected_observation_count": selected.observation_count,
        "selected_degrees_of_freedom": selected.degrees_of_freedom,
        "selected_reduced_chi2": selected.reduced_chi2,
        "kernel_candidates": [
            {
                "name": fit.name,
                "parameter_names": list(fit.parameter_names),
                "parameters": list(fit.parameters),
                "chi2": fit.chi2,
                "parameter_count": fit.parameter_count,
                "aic": fit.aic,
                "success": fit.success,
                "at_bound": list(fit.at_bound),
                "observation_count": fit.observation_count,
                "degrees_of_freedom": fit.degrees_of_freedom,
                "reduced_chi2": fit.reduced_chi2,
                "attempted_starts": fit.attempted_starts,
                "successful_starts": fit.successful_starts,
                "near_best_starts": fit.near_best_starts,
                "best_start_index": fit.best_start_index,
            }
            for fit in source.diagnostics.kernel_fits
        ],
        "sweep_run_diagnostics": [
            {
                "thickness_nm": run.thickness_nm,
                "retained_bins": run.retained_bins,
                "dropped_final_bins": run.dropped_final_bins,
                "dropped_extra_artifact_bins": run.dropped_extra_artifact_bins,
                "ratio_noise": run.ratio_noise,
            }
            for run in source.diagnostics.runs
        ],
        "integration_step_nm": source.integration_step_nm,
        "integration_cells": source.integration_cells,
        "minimum_cells": source.minimum_cells,
        "maximum_cells": source.maximum_cells,
    }

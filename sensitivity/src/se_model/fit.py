"""Deterministic bounded fitting for the full-covariance SE likelihood."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, minimize

from .likelihood import LikelihoodEvaluation, evaluate_gaussian_eiv
from .models import ModelSpec

try:
    from .models import predict_model
except ImportError:
    from .models import prediction as predict_model

if TYPE_CHECKING:
    from .data import ObservationData
    from .deposition import IonizationProfile

FloatArray = NDArray[np.float64]


def _json_compatible(value: Any) -> Any:
    """Replace non-finite floats with JSON ``null`` recursively."""

    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True)
class StartRecord:
    """Complete record of one deterministic optimization start."""

    index: int
    start_params: OrderedDict[str, float]
    start_objective: float | None
    final_params: OrderedDict[str, float] | None
    objective: float | None
    success: bool
    accepted: bool
    status: int | None
    message: str
    n_function_evaluations: int | None
    n_iterations: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "index": self.index,
            "start_params": dict(self.start_params),
            "start_objective": self.start_objective,
            "final_params": (
                None if self.final_params is None else dict(self.final_params)
            ),
            "objective": self.objective,
            "success": self.success,
            "accepted": self.accepted,
            "status": self.status,
            "message": self.message,
            "n_function_evaluations": self.n_function_evaluations,
            "n_iterations": self.n_iterations,
        }


@dataclass(frozen=True)
class FitResult:
    """Fit, likelihood, uncertainty, and optimization diagnostics."""

    model_key: str
    parameter_names: tuple[str, ...]
    free_parameter_names: tuple[str, ...]
    params: OrderedDict[str, float]
    param_vector: FloatArray
    fixed_params: OrderedDict[str, float]
    likelihood: LikelihoodEvaluation
    deviance: float
    log_likelihood: float
    n_observations: int
    n_free_parameters: int
    aic: float
    aicc: float
    bic: float
    covariance: FloatArray
    correlation: FloatArray
    hessian_log: FloatArray
    hessian_rank: int
    hessian_condition_number: float
    hessian_positive_definite: bool
    at_lower_bound: tuple[str, ...]
    at_upper_bound: tuple[str, ...]
    at_any_bound: tuple[str, ...]
    effective_lower_bounds: OrderedDict[str, float]
    effective_upper_bounds: OrderedDict[str, float]
    optimizer_method: str
    seed: int
    n_starts: int
    best_start_index: int
    starts: tuple[StartRecord, ...]
    success: bool
    message: str

    def to_dict(
        self,
        *,
        include_likelihood_arrays: bool = True,
        include_start_records: bool = True,
    ) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload: dict[str, Any] = {
            "model_key": self.model_key,
            "parameter_names": list(self.parameter_names),
            "free_parameter_names": list(self.free_parameter_names),
            "params": dict(self.params),
            "param_vector": self.param_vector.tolist(),
            "fixed_params": dict(self.fixed_params),
            "likelihood": self.likelihood.to_dict(
                include_arrays=include_likelihood_arrays
            ),
            "deviance": self.deviance,
            "log_likelihood": self.log_likelihood,
            "n_observations": self.n_observations,
            "n_free_parameters": self.n_free_parameters,
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
            "covariance": self.covariance.tolist(),
            "correlation": self.correlation.tolist(),
            "hessian_log": self.hessian_log.tolist(),
            "hessian_rank": self.hessian_rank,
            "hessian_condition_number": self.hessian_condition_number,
            "hessian_positive_definite": self.hessian_positive_definite,
            "at_lower_bound": list(self.at_lower_bound),
            "at_upper_bound": list(self.at_upper_bound),
            "at_any_bound": list(self.at_any_bound),
            "effective_lower_bounds": dict(self.effective_lower_bounds),
            "effective_upper_bounds": dict(self.effective_upper_bounds),
            "optimizer_method": self.optimizer_method,
            "seed": self.seed,
            "n_starts": self.n_starts,
            "best_start_index": self.best_start_index,
            "success": self.success,
            "message": self.message,
        }
        if include_start_records:
            payload["starts"] = [record.to_dict() for record in self.starts]
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result to JSON."""

        return json.dumps(
            _json_compatible(self.to_dict()),
            indent=indent,
            allow_nan=False,
        )


@dataclass(frozen=True)
class ProfilePoint:
    """One conditional fit in a parameter profile."""

    value: float
    success: bool
    deviance: float | None
    delta_deviance: float | None
    result: FitResult | None
    message: str

    def to_dict(self, *, include_fit: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload: dict[str, Any] = {
            "value": self.value,
            "success": self.success,
            "deviance": self.deviance,
            "delta_deviance": self.delta_deviance,
            "message": self.message,
        }
        if include_fit and self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload


@dataclass(frozen=True)
class ProfileResult:
    """Conditional-deviance profile for one physical parameter."""

    model_key: str
    parameter: str
    reference_deviance: float
    points: tuple[ProfilePoint, ...]

    def to_dict(self, *, include_fits: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "model_key": self.model_key,
            "parameter": self.parameter,
            "reference_deviance": self.reference_deviance,
            "points": [
                point.to_dict(include_fit=include_fits) for point in self.points
            ],
        }


def _first_attribute(instance: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(instance, name):
            return getattr(instance, name)
    joined = ", ".join(names)
    raise AttributeError(
        f"{type(instance).__name__} must provide one of these attributes: {joined}."
    )


def _observation_arrays(
    data: ObservationData,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    x = np.asarray(
        _first_attribute(data, ("x", "thickness_nm", "thickness")), dtype=float
    )
    y = np.asarray(
        _first_attribute(data, ("y", "intensity", "normalized_intensity")),
        dtype=float,
    )
    covariance = np.asarray(
        _first_attribute(
            data,
            (
                "y_cov",
                "y_covariance",
                "covariance_y",
                "covariance",
                "intensity_covariance",
            ),
        ),
        dtype=float,
    )
    x_sigma = np.asarray(
        _first_attribute(
            data,
            ("x_sigma", "thickness_sigma_nm", "thickness_sigma"),
        ),
        dtype=float,
    )

    if x.ndim != 1 or y.ndim != 1 or x_sigma.ndim != 1:
        raise ValueError("x, y, and x_sigma must be one-dimensional.")
    if x.size != y.size or x.size != x_sigma.size:
        raise ValueError("x, y, and x_sigma must have equal lengths.")
    if covariance.shape != (x.size, x.size):
        raise ValueError(
            f"y covariance has shape {covariance.shape}; expected {(x.size, x.size)}."
        )
    if not all(np.all(np.isfinite(array)) for array in (x, y, covariance, x_sigma)):
        raise ValueError("Observation data contain a non-finite value.")
    if np.any(x_sigma < 0.0):
        raise ValueError("x_sigma must be nonnegative.")
    return x, y, covariance, x_sigma


def _ordered_values(
    value: Mapping[str, float] | Sequence[float],
    names: tuple[str, ...],
    label: str,
) -> OrderedDict[str, float]:
    if isinstance(value, Mapping):
        missing = [name for name in names if name not in value]
        extra = [name for name in value if name not in names]
        if missing or extra:
            raise ValueError(f"{label}: missing={missing}, extra={extra}.")
        return OrderedDict((name, float(value[name])) for name in names)

    vector = np.asarray(value, dtype=float)
    if vector.shape != (len(names),):
        raise ValueError(f"{label} has shape {vector.shape}; expected {(len(names),)}.")
    return OrderedDict(zip(names, (float(item) for item in vector), strict=True))


def _specification(
    spec: ModelSpec,
) -> tuple[
    str,
    tuple[str, ...],
    OrderedDict[str, float],
    OrderedDict[str, float],
    OrderedDict[str, float],
]:
    key = str(spec.key)
    names = tuple(str(name) for name in spec.parameter_names)
    if len(names) != len(set(names)) or not names:
        raise ValueError("ModelSpec.parameter_names must be nonempty and unique.")

    lower = _ordered_values(spec.lower_bounds, names, "lower_bounds")
    upper = _ordered_values(spec.upper_bounds, names, "upper_bounds")
    initial = _ordered_values(spec.initial, names, "initial")

    for name in names:
        if lower[name] < 0.0:
            raise ValueError(f"{name}: lower bound must be nonnegative.")
        if not upper[name] > lower[name]:
            raise ValueError(f"{name}: upper bound must exceed lower bound.")
        if not np.isfinite(initial[name]) or initial[name] <= 0.0:
            raise ValueError(f"{name}: initial value must be finite and positive.")
        if not lower[name] <= initial[name] <= upper[name]:
            raise ValueError(f"{name}: initial value lies outside its bounds.")
    return key, names, lower, upper, initial


def _effective_lower_bound(lower: float, upper: float, initial: float) -> float:
    if lower > 0.0:
        return lower
    candidates = [initial * 1e-12, np.finfo(float).tiny]
    if np.isfinite(upper):
        candidates.append(upper * 1e-15)
    return max(candidates)


def _effective_upper_bound(upper: float, initial: float) -> float:
    if np.isfinite(upper):
        return upper
    # An infinite physical upper bound is represented by an extremely broad,
    # finite computational bound.  This prevents exp(log_parameter) overflow
    # without constraining any scientifically relevant region.
    return initial * 1e12


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _model_prediction(
    spec: ModelSpec,
    profile: IonizationProfile,
    x: FloatArray,
    params: OrderedDict[str, float],
) -> FloatArray:
    if "return_components" in inspect.signature(predict_model).parameters:
        prediction = predict_model(
            spec,
            profile,
            x,
            params,
            return_components=False,
        )
    else:
        prediction = predict_model(spec, profile, x, params)
    result = np.asarray(prediction, dtype=float)
    if result.shape != x.shape:
        raise ValueError(
            f"predict_model returned shape {result.shape}; expected {x.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("predict_model returned a non-finite value.")
    return result


def _model_slope(
    spec: ModelSpec,
    profile: IonizationProfile,
    x: FloatArray,
    params: OrderedDict[str, float],
    *,
    relative_step: float,
    absolute_step: float,
) -> FloatArray:
    if relative_step <= 0.0 or absolute_step <= 0.0:
        raise ValueError("Derivative steps must be positive.")
    step = np.maximum(absolute_step, relative_step * np.maximum(np.abs(x), 1.0))
    lower_x = x - step
    upper_x = x + step

    slope = np.empty_like(x)
    central = lower_x >= 0.0
    if np.any(central):
        upper_prediction = _model_prediction(spec, profile, upper_x[central], params)
        lower_prediction = _model_prediction(spec, profile, lower_x[central], params)
        slope[central] = (upper_prediction - lower_prediction) / (2.0 * step[central])
    if np.any(~central):
        upper_prediction = _model_prediction(spec, profile, upper_x[~central], params)
        center_prediction = _model_prediction(spec, profile, x[~central], params)
        slope[~central] = (upper_prediction - center_prediction) / step[~central]
    return slope


def _numerical_hessian(
    objective: Any,
    optimum: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    relative_step: float = 2e-4,
    absolute_step: float = 2e-5,
) -> FloatArray:
    """Central finite-difference Hessian inside the optimization bounds."""

    dimension = optimum.size
    if dimension == 0:
        return np.empty((0, 0), dtype=float)

    distance_lower = optimum - lower
    distance_upper = upper - optimum
    nominal = np.maximum(
        absolute_step, relative_step * np.maximum(np.abs(optimum), 1.0)
    )
    step = nominal.copy()
    for index in range(dimension):
        finite_distance = min(distance_lower[index], distance_upper[index])
        if not np.isfinite(finite_distance):
            finite_distance = max(distance_lower[index], distance_upper[index])
        if np.isfinite(finite_distance):
            step[index] = min(step[index], 0.25 * finite_distance)
        if not np.isfinite(step[index]) or step[index] <= np.finfo(float).eps:
            raise RuntimeError(
                "Cannot evaluate a central Hessian because parameter "
                f"{index} is too close to a bound."
            )

    center = float(objective(optimum))
    if not np.isfinite(center):
        raise RuntimeError("Objective is non-finite at the fitted optimum.")

    hessian = np.empty((dimension, dimension), dtype=float)
    for i in range(dimension):
        plus = optimum.copy()
        minus = optimum.copy()
        plus[i] += step[i]
        minus[i] -= step[i]
        hessian[i, i] = (
            float(objective(plus)) - 2.0 * center + float(objective(minus))
        ) / step[i] ** 2

        for j in range(i):
            pp = optimum.copy()
            pm = optimum.copy()
            mp = optimum.copy()
            mm = optimum.copy()
            pp[i] += step[i]
            pp[j] += step[j]
            pm[i] += step[i]
            pm[j] -= step[j]
            mp[i] -= step[i]
            mp[j] += step[j]
            mm[i] -= step[i]
            mm[j] -= step[j]
            value = (
                float(objective(pp))
                - float(objective(pm))
                - float(objective(mp))
                + float(objective(mm))
            ) / (4.0 * step[i] * step[j])
            hessian[i, j] = value
            hessian[j, i] = value

    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("Numerical Hessian contains a non-finite value.")
    return 0.5 * (hessian + hessian.T)


def _covariance_diagnostics(
    hessian_log_free: FloatArray,
    free_indices: list[int],
    physical_vector: FloatArray,
    total_dimension: int,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float, bool]:
    full_hessian = np.zeros((total_dimension, total_dimension), dtype=float)
    covariance = np.zeros((total_dimension, total_dimension), dtype=float)

    if not free_indices:
        correlation = np.eye(total_dimension, dtype=float)
        return full_hessian, covariance, correlation, 0, float("nan"), True

    selector = np.ix_(free_indices, free_indices)
    full_hessian[selector] = hessian_log_free
    eigenvalues = np.linalg.eigvalsh(hessian_log_free)
    tolerance = (
        max(abs(float(eigenvalues[0])), abs(float(eigenvalues[-1])), 1.0)
        * max(hessian_log_free.shape)
        * np.finfo(float).eps
        * 100.0
    )
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(np.abs(eigenvalues) > tolerance))
    positive_definite = bool(np.all(positive))
    positive_eigenvalues = eigenvalues[positive]
    condition = (
        float(positive_eigenvalues[-1] / positive_eigenvalues[0])
        if positive_eigenvalues.size == eigenvalues.size and eigenvalues.size
        else float("inf")
    )

    # The optimized objective is -2 log L, so Cov(log p) = 2 H_D^{-1}.
    covariance_log = 2.0 * np.linalg.pinv(
        hessian_log_free,
        rcond=1e-12,
        hermitian=True,
    )
    jacobian = np.diag(physical_vector[free_indices])
    covariance_physical = jacobian @ covariance_log @ jacobian
    covariance[selector] = 0.5 * (covariance_physical + covariance_physical.T)

    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.zeros_like(covariance)
    np.divide(
        covariance,
        denominator,
        out=correlation,
        where=denominator > 0.0,
    )
    for index in range(total_dimension):
        correlation[index, index] = 1.0
    return (
        full_hessian,
        covariance,
        correlation,
        rank,
        condition,
        positive_definite,
    )


def fit_model(
    data: ObservationData,
    profile: IonizationProfile,
    spec: ModelSpec,
    *,
    seed: int = 20260730,
    n_starts: int = 32,
    fixed_params: Mapping[str, float] | None = None,
    additional_starts: (
        Sequence[Mapping[str, float] | Sequence[float]] | None
    ) = None,
    frozen_covariance: ArrayLike | None = None,
    derivative_relative_step: float = 1e-4,
    derivative_absolute_step: float = 1e-4,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: Mapping[str, Any] | None = None,
) -> FitResult:
    """Fit one model with deterministic multistart optimization.

    All free physical parameters are represented in logarithmic coordinates.
    Every start is recorded; failed or internally inconsistent optima are
    rejected rather than silently considered.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least one.")
    key, names, lower, upper, initial = _specification(spec)
    x, y, y_covariance, x_sigma = _observation_arrays(data)
    fixed = OrderedDict()
    for name, value in (fixed_params or {}).items():
        if name not in names:
            raise KeyError(f"Unknown fixed parameter {name!r}.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"Fixed parameter {name} must be finite and positive.")
        if not lower[name] <= numeric <= upper[name]:
            raise ValueError(f"Fixed parameter {name} lies outside its bounds.")
        fixed[name] = numeric

    free_names = tuple(name for name in names if name not in fixed)
    free_indices = [names.index(name) for name in free_names]
    effective_lower = OrderedDict(
        (
            name,
            _effective_lower_bound(lower[name], upper[name], initial[name]),
        )
        for name in names
    )
    effective_upper = OrderedDict(
        (name, _effective_upper_bound(upper[name], initial[name])) for name in names
    )
    log_lower = np.asarray(
        [np.log(effective_lower[name]) for name in free_names], dtype=float
    )
    log_upper = np.asarray(
        [np.log(effective_upper[name]) for name in free_names],
        dtype=float,
    )

    frozen = (
        None
        if frozen_covariance is None
        else np.asarray(frozen_covariance, dtype=float)
    )

    def assemble(log_free: FloatArray) -> OrderedDict[str, float]:
        free_values = {
            name: float(np.exp(value))
            for name, value in zip(free_names, log_free, strict=True)
        }
        return OrderedDict(
            (
                name,
                fixed[name] if name in fixed else free_values[name],
            )
            for name in names
        )

    def evaluate(log_free: FloatArray) -> LikelihoodEvaluation:
        params = assemble(np.asarray(log_free, dtype=float))
        prediction = _model_prediction(spec, profile, x, params)
        slope = _model_slope(
            spec,
            profile,
            x,
            params,
            relative_step=derivative_relative_step,
            absolute_step=derivative_absolute_step,
        )
        return evaluate_gaussian_eiv(
            y,
            prediction,
            y_covariance,
            x_sigma,
            slope,
            frozen_covariance=frozen,
        )

    def objective(log_free: FloatArray) -> float:
        try:
            value = evaluate(log_free).deviance
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    base_initial = OrderedDict(initial)
    base_initial.update(fixed)
    base_log = np.asarray(
        [np.log(base_initial[name]) for name in free_names], dtype=float
    )
    base_log = np.maximum(base_log, log_lower)
    base_log = np.minimum(base_log, log_upper)

    stable_key = key + "|" + "|".join(f"{name}={fixed[name]:.17g}" for name in fixed)
    generator = np.random.default_rng(_stable_seed(seed, stable_key))
    starts = [base_log]
    for supplied in additional_starts or ():
        if len(starts) >= n_starts:
            break
        supplied_values = _ordered_values(
            supplied,
            names,
            "additional_start",
        )
        supplied_values.update(fixed)
        vector = np.asarray([supplied_values[name] for name in names], dtype=float)
        physical_lower = np.asarray([lower[name] for name in names], dtype=float)
        physical_upper = np.asarray([upper[name] for name in names], dtype=float)
        if (
            not np.all(np.isfinite(vector))
            or np.any(vector <= 0.0)
            or np.any(vector < physical_lower)
            or np.any(vector > physical_upper)
        ):
            raise ValueError(
                "Every additional start must be finite, positive, and within "
                "the model bounds."
            )
        candidate = np.asarray(
            [np.log(supplied_values[name]) for name in free_names],
            dtype=float,
        )
        if not any(
            np.allclose(candidate, existing, rtol=0.0, atol=1e-14)
            for existing in starts
        ):
            starts.append(candidate)
    while len(starts) < n_starts:
        candidate = np.empty(len(free_names), dtype=float)
        for index, name in enumerate(free_names):
            sample_lower = max(log_lower[index], base_log[index] - 6.0)
            sample_upper = min(log_upper[index], base_log[index] + 6.0)
            candidate[index] = generator.uniform(sample_lower, sample_upper)
        starts.append(candidate)

    options = {
        "ftol": 1e-12,
        "gtol": 1e-8,
        "maxiter": 3000,
        "maxls": 50,
    }
    if optimizer_options:
        options.update(dict(optimizer_options))

    records: list[StartRecord] = []
    accepted: list[tuple[int, OptimizeResult, LikelihoodEvaluation]] = []
    bounds = list(zip(log_lower, log_upper, strict=True))
    for index, start in enumerate(starts):
        start_params = assemble(start)
        start_objective: float | None = None
        try:
            start_objective = objective(start)
            if free_names:
                result = minimize(
                    objective,
                    start,
                    method=optimizer_method,
                    bounds=bounds,
                    options=options,
                )
                final_log = np.asarray(result.x, dtype=float)
            else:
                result = OptimizeResult(
                    x=np.empty(0),
                    fun=objective(np.empty(0)),
                    success=True,
                    status=0,
                    message="No free parameters.",
                    nfev=1,
                    nit=0,
                )
                final_log = np.empty(0)

            final_params = assemble(final_log)
            final_evaluation = evaluate(final_log)
            objective_consistent = np.isclose(
                float(result.fun),
                final_evaluation.deviance,
                rtol=1e-8,
                atol=1e-8,
            )
            nonworsening = bool(
                np.isfinite(start_objective)
                and final_evaluation.deviance
                <= start_objective
                + 1e-8 * max(1.0, abs(start_objective))
            )
            accepted_result = bool(
                result.success
                and objective_consistent
                and nonworsening
                and np.isfinite(final_evaluation.deviance)
            )
            message = str(result.message)
            if result.success and not objective_consistent:
                message += " Final objective/evaluation mismatch."
            if result.success and not nonworsening:
                message += " Final objective is worse than its start."
            if accepted_result:
                accepted.append((index, result, final_evaluation))
            records.append(
                StartRecord(
                    index=index,
                    start_params=start_params,
                    start_objective=float(start_objective),
                    final_params=final_params,
                    objective=float(final_evaluation.deviance),
                    success=bool(result.success),
                    accepted=accepted_result,
                    status=int(result.status),
                    message=message,
                    n_function_evaluations=int(result.nfev),
                    n_iterations=int(result.nit),
                )
            )
        except Exception as error:  # Every failed start remains auditable.
            records.append(
                StartRecord(
                    index=index,
                    start_params=start_params,
                    start_objective=(
                        float(start_objective)
                        if start_objective is not None
                        and np.isfinite(start_objective)
                        else None
                    ),
                    final_params=None,
                    objective=None,
                    success=False,
                    accepted=False,
                    status=None,
                    message=f"{type(error).__name__}: {error}",
                    n_function_evaluations=None,
                    n_iterations=None,
                )
            )

    if not accepted:
        messages = "; ".join(
            f"start {record.index}: {record.message}" for record in records
        )
        raise RuntimeError(f"All optimization starts failed for {key}: {messages}")

    best_index, best_optimizer, final_evaluation = min(
        accepted, key=lambda item: item[2].deviance
    )
    final_log = np.asarray(best_optimizer.x, dtype=float)
    final_params = assemble(final_log)
    # This is deliberately recomputed after model selection so every reported
    # quantity comes from precisely the final parameter vector.
    final_evaluation = evaluate(final_log)
    final_deviance = final_evaluation.deviance

    physical_vector = np.asarray(list(final_params.values()), dtype=float)
    try:
        hessian_free = _numerical_hessian(
            objective,
            final_log,
            log_lower,
            log_upper,
        )
        (
            hessian_log,
            covariance,
            correlation,
            hessian_rank,
            hessian_condition,
            hessian_positive_definite,
        ) = _covariance_diagnostics(
            hessian_free,
            free_indices,
            physical_vector,
            len(names),
        )
        hessian_message = ""
    except RuntimeError as error:
        shape = (len(names), len(names))
        hessian_log = np.full(shape, np.nan)
        covariance = np.full(shape, np.nan)
        correlation = np.full(shape, np.nan)
        hessian_rank = 0
        hessian_condition = float("inf")
        hessian_positive_definite = False
        hessian_message = f" Hessian unavailable: {error}"

    lower_names = tuple(
        name
        for name in free_names
        if np.isclose(
            final_params[name],
            effective_lower[name],
            rtol=1e-5,
            atol=0.0,
        )
    )
    upper_names = tuple(
        name
        for name in free_names
        if np.isclose(
            final_params[name],
            effective_upper[name],
            rtol=1e-5,
            atol=0.0,
        )
    )
    at_any = tuple(name for name in names if name in lower_names or name in upper_names)

    n_observations = y.size
    parameter_count = len(free_names)
    aic = final_deviance + 2.0 * parameter_count
    denominator = n_observations - parameter_count - 1
    aicc = (
        aic + 2.0 * parameter_count * (parameter_count + 1) / denominator
        if denominator > 0
        else float("inf")
    )
    bic = final_deviance + parameter_count * np.log(n_observations)
    message = (
        f"Accepted {len(accepted)}/{n_starts} starts; best start {best_index}."
        f"{hessian_message}"
    )

    return FitResult(
        model_key=key,
        parameter_names=names,
        free_parameter_names=free_names,
        params=final_params,
        param_vector=physical_vector,
        fixed_params=fixed,
        likelihood=final_evaluation,
        deviance=final_deviance,
        log_likelihood=final_evaluation.log_likelihood,
        n_observations=n_observations,
        n_free_parameters=parameter_count,
        aic=float(aic),
        aicc=float(aicc),
        bic=float(bic),
        covariance=covariance,
        correlation=correlation,
        hessian_log=hessian_log,
        hessian_rank=hessian_rank,
        hessian_condition_number=hessian_condition,
        hessian_positive_definite=hessian_positive_definite,
        at_lower_bound=lower_names,
        at_upper_bound=upper_names,
        at_any_bound=at_any,
        effective_lower_bounds=effective_lower,
        effective_upper_bounds=effective_upper,
        optimizer_method=optimizer_method,
        seed=int(seed),
        n_starts=n_starts,
        best_start_index=best_index,
        starts=tuple(records),
        success=True,
        message=message,
    )


def profile_parameter(
    data: ObservationData,
    profile: IonizationProfile,
    spec: ModelSpec,
    parameter: str,
    grid: ArrayLike,
    *,
    reference_fit: FitResult | None = None,
    seed: int = 20260730,
    n_starts: int = 16,
    frozen_covariance: ArrayLike | None = None,
    derivative_relative_step: float = 1e-4,
    derivative_absolute_step: float = 1e-4,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: Mapping[str, Any] | None = None,
) -> ProfileResult:
    """Profile one physical parameter by refitting all remaining parameters."""

    key, names, lower, upper, _ = _specification(spec)
    if parameter not in names:
        raise KeyError(f"Unknown profile parameter {parameter!r}.")
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("grid must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Every profile-grid value must be finite and positive.")

    if reference_fit is None:
        reference_fit = fit_model(
            data,
            profile,
            spec,
            seed=seed,
            n_starts=max(n_starts, 1),
            frozen_covariance=frozen_covariance,
            derivative_relative_step=derivative_relative_step,
            derivative_absolute_step=derivative_absolute_step,
            optimizer_method=optimizer_method,
            optimizer_options=optimizer_options,
        )
    if reference_fit.model_key != key:
        raise ValueError("reference_fit belongs to a different model.")
    reference_deviance = reference_fit.deviance

    points: list[ProfilePoint] = []
    for value in values:
        numeric = float(value)
        if not lower[parameter] <= numeric <= upper[parameter]:
            points.append(
                ProfilePoint(
                    value=numeric,
                    success=False,
                    deviance=None,
                    delta_deviance=None,
                    result=None,
                    message="Grid value lies outside the model bounds.",
                )
            )
            continue
        try:
            result = fit_model(
                data,
                profile,
                spec,
                seed=_stable_seed(seed, f"profile:{key}:{parameter}:{numeric.hex()}"),
                n_starts=n_starts,
                fixed_params={parameter: numeric},
                frozen_covariance=frozen_covariance,
                derivative_relative_step=derivative_relative_step,
                derivative_absolute_step=derivative_absolute_step,
                optimizer_method=optimizer_method,
                optimizer_options=optimizer_options,
            )
            points.append(
                ProfilePoint(
                    value=numeric,
                    success=True,
                    deviance=result.deviance,
                    delta_deviance=result.deviance - reference_deviance,
                    result=result,
                    message=result.message,
                )
            )
        except Exception as error:
            points.append(
                ProfilePoint(
                    value=numeric,
                    success=False,
                    deviance=None,
                    delta_deviance=None,
                    result=None,
                    message=f"{type(error).__name__}: {error}",
                )
            )

    return ProfileResult(
        model_key=key,
        parameter=parameter,
        reference_deviance=reference_deviance,
        points=tuple(points),
    )

"""The three paper forward models, without model-selection interpretation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .deposition import IonizationProfile

ParameterValues = Mapping[str, float] | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class ModelPrediction:
    """Total prediction and its two additive surface components."""

    total: np.ndarray
    front: np.ndarray
    back: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification of one paper model.

    ``parameter_names`` is the authoritative physical parameter order used by
    optimizers and serialized outputs.  Every escape length is measured-axis
    EELS nm; ``scale`` is the dimensionless EELS-to-SRIM horizontal scale.
    """

    key: str
    label: str
    parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    initial: tuple[float, ...]
    escape_mode: str
    lambda_front_nm: float = 2.0

    def __post_init__(self) -> None:
        size = len(self.parameter_names)
        if not (
            len(self.lower_bounds)
            == len(self.upper_bounds)
            == len(self.initial)
            == size
        ):
            raise ValueError(f"{self.key}: parameter metadata lengths do not match.")
        if len(set(self.parameter_names)) != size:
            raise ValueError(f"{self.key}: parameter names must be unique.")
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        initial = np.asarray(self.initial, dtype=float)
        if np.any(lower >= upper):
            raise ValueError(f"{self.key}: every lower bound must be below upper.")
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError(f"{self.key}: initial values must lie within bounds.")
        if self.escape_mode not in {"fixed_symmetric", "free_symmetric", "asymmetric"}:
            raise ValueError(f"{self.key}: unknown escape mode {self.escape_mode}.")
        if not np.isfinite(self.lambda_front_nm) or self.lambda_front_nm <= 0.0:
            raise ValueError(f"{self.key}: lambda_front_nm must be positive.")

    def parameters(self, values: ParameterValues) -> dict[str, float]:
        """Return a validated name-to-value mapping in physical units."""

        if isinstance(values, Mapping):
            missing = set(self.parameter_names).difference(values)
            extra = set(values).difference(self.parameter_names)
            if missing or extra:
                raise ValueError(
                    f"{self.key}: parameter keys differ; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}."
                )
            vector = np.asarray(
                [values[name] for name in self.parameter_names], dtype=float
            )
        else:
            vector = np.asarray(values, dtype=float)
            if vector.ndim != 1 or len(vector) != len(self.parameter_names):
                raise ValueError(
                    f"{self.key}: expected {len(self.parameter_names)} parameters "
                    f"in order {self.parameter_names}."
                )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{self.key}: parameters must be finite.")
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if np.any(vector < lower) or np.any(vector > upper):
            raise ValueError(f"{self.key}: parameters lie outside declared bounds.")
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, vector)
        }


def paper_model_specs(lambda_front_nm: float = 2.0) -> tuple[ModelSpec, ...]:
    """Build the fixed-symmetric, free-symmetric, and asymmetric models."""

    common_names = ("C_o", "C_b", "scale")
    common_lower = (0.0, 0.0, 0.2)
    common_upper = (np.inf, np.inf, 3.0)
    return (
        ModelSpec(
            key="symmetric_fixed",
            label="Symmetric",
            parameter_names=common_names,
            lower_bounds=common_lower,
            upper_bounds=common_upper,
            # Reviewed S0 anchors.  These are optimizer starts, not fixed
            # values or priors; deterministic multistart fitting still
            # explores the declared bounds.
            initial=(7.38289e-4, 2.93341, 0.656248),
            escape_mode="fixed_symmetric",
            lambda_front_nm=lambda_front_nm,
        ),
        ModelSpec(
            key="symmetric_free",
            label=r"Symmetric ($\lambda$ free)",
            parameter_names=common_names + ("lambda_shared_nm",),
            lower_bounds=common_lower + (0.2,),
            upper_bounds=common_upper + (50.0,),
            initial=(1.63512e-4, 3.04172, 0.619060, 8.53015),
            escape_mode="free_symmetric",
            lambda_front_nm=lambda_front_nm,
        ),
        ModelSpec(
            key="asymmetric",
            label="Asymmetric effective response",
            parameter_names=common_names + ("lambda_back_nm",),
            lower_bounds=common_lower + (0.2,),
            upper_bounds=common_upper + (50.0,),
            initial=(6.85118e-4, 0.608305, 0.604148, 10.0111),
            escape_mode="asymmetric",
            lambda_front_nm=lambda_front_nm,
        ),
    )


PAPER_MODEL_SPECS = paper_model_specs()


def _escape_lengths(spec: ModelSpec, parameters: Mapping[str, float]) -> tuple:
    if spec.escape_mode == "fixed_symmetric":
        return spec.lambda_front_nm, spec.lambda_front_nm
    if spec.escape_mode == "free_symmetric":
        shared = parameters["lambda_shared_nm"]
        return shared, shared
    return spec.lambda_front_nm, parameters["lambda_back_nm"]


def components(
    spec: ModelSpec,
    profile: IonizationProfile,
    thickness_nm: np.ndarray | float,
    parameter_values: ParameterValues,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the additive collected front and back model components."""

    parameters = spec.parameters(parameter_values)
    lambda_front, lambda_back = _escape_lengths(spec, parameters)
    raw_front, raw_back = profile.measured_components(
        thickness_nm=thickness_nm,
        scale=parameters["scale"],
        lambda_front_nm=lambda_front,
        lambda_back_nm=lambda_back,
    )
    front = parameters["C_o"] * raw_front
    back = parameters["C_o"] * parameters["C_b"] * raw_back
    return front, back


def prediction(
    spec: ModelSpec,
    profile: IonizationProfile,
    thickness_nm: np.ndarray | float,
    parameter_values: ParameterValues,
) -> np.ndarray:
    """Evaluate total collected intensity for one paper model."""

    front, back = components(spec, profile, thickness_nm, parameter_values)
    return front + back


def predict_model(
    spec: ModelSpec,
    profile: IonizationProfile,
    x: np.ndarray | float,
    params: ParameterValues,
    *,
    return_components: bool = False,
) -> np.ndarray | ModelPrediction:
    """Canonical compatibility wrapper for fitting and reporting code.

    With ``return_components=False`` (the default), return the total model
    intensity array.  With ``return_components=True``, return a
    :class:`ModelPrediction` carrying ``total``, ``front``, and ``back`` arrays
    in a named structure.
    """

    front, back = components(spec, profile, x, params)
    total = front + back
    if return_components:
        return ModelPrediction(total=total, front=front, back=back)
    return total

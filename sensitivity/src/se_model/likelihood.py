"""Gaussian full-covariance likelihood with first-order x-error propagation.

The primary likelihood retains the complete covariance of the measured
intensities and propagates independent x-axis uncertainties through the local
model slope.  Because that propagated covariance depends on the model
parameters, the log-determinant is part of the objective.

A frozen-covariance mode is provided only for sensitivity analyses.  It uses
the same full Gaussian likelihood, but holds the supplied covariance fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CholeskyResult:
    """Result of a positive-definite Cholesky factorization."""

    factor: FloatArray
    jitter: float
    attempts: int
    scale: float


@dataclass(frozen=True)
class LikelihoodEvaluation:
    """Complete numerical record for one likelihood evaluation."""

    deviance: float
    log_likelihood: float
    mahalanobis: float
    log_determinant: float
    n_observations: int
    residual: FloatArray
    prediction: FloatArray
    slope: FloatArray
    whitened_residual: FloatArray
    covariance: FloatArray
    base_covariance_id: str
    covariance_id: str
    covariance_mode: str
    cholesky_jitter: float
    cholesky_attempts: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload: dict[str, Any] = {
            "deviance": self.deviance,
            "log_likelihood": self.log_likelihood,
            "mahalanobis": self.mahalanobis,
            "log_determinant": self.log_determinant,
            "n_observations": self.n_observations,
            "base_covariance_id": self.base_covariance_id,
            "covariance_id": self.covariance_id,
            "covariance_mode": self.covariance_mode,
            "cholesky_jitter": self.cholesky_jitter,
            "cholesky_attempts": self.cholesky_attempts,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
        }
        if include_arrays:
            payload.update(
                {
                    "residual": self.residual.tolist(),
                    "prediction": self.prediction.tolist(),
                    "slope": self.slope.tolist(),
                    "whitened_residual": self.whitened_residual.tolist(),
                    "covariance": self.covariance.tolist(),
                }
            )
        return payload


def covariance_identifier(covariance: ArrayLike) -> str:
    """Return a stable content identifier for a covariance matrix."""

    matrix = np.ascontiguousarray(np.asarray(covariance, dtype="<f8"))
    if matrix.ndim != 2:
        raise ValueError("A covariance matrix must be two-dimensional.")
    digest = hashlib.sha256()
    digest.update(b"se-model-covariance-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _as_vector(value: ArrayLike, name: str, size: int | None = None) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {vector.shape}.")
    if size is not None and vector.size != size:
        raise ValueError(f"{name} has length {vector.size}; expected {size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value.")
    return vector


def _as_covariance(value: ArrayLike, size: int, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} has shape {matrix.shape}; expected {(size, size)}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite value.")

    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if asymmetry > 1e-10 * scale:
        raise ValueError(
            f"{name} is not symmetric within tolerance "
            f"(maximum asymmetry {asymmetry:.3g})."
        )
    return 0.5 * (matrix + matrix.T)


def stable_cholesky(
    covariance: ArrayLike,
    *,
    initial_relative_jitter: float = 1e-12,
    maximum_relative_jitter: float = 1e-6,
    maximum_attempts: int = 8,
) -> CholeskyResult:
    """Factor a covariance matrix, adding trace-scaled jitter if necessary.

    The unmodified matrix is always attempted first.  Any nonzero jitter is
    returned and must be reported; it is not silently hidden.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance contains a non-finite value.")
    matrix = 0.5 * (matrix + matrix.T)

    diagonal_scale = float(np.mean(np.diag(matrix)))
    spectral_scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    scale = max(diagonal_scale, spectral_scale, np.finfo(float).tiny)
    identity = np.eye(matrix.shape[0], dtype=float)

    jitters = [0.0]
    relative = initial_relative_jitter
    while len(jitters) < maximum_attempts and relative <= maximum_relative_jitter:
        jitters.append(relative * scale)
        relative *= 10.0

    last_error: np.linalg.LinAlgError | None = None
    for attempt, jitter in enumerate(jitters, start=1):
        try:
            factor = np.linalg.cholesky(matrix + jitter * identity)
        except np.linalg.LinAlgError as error:
            last_error = error
            continue
        return CholeskyResult(
            factor=factor,
            jitter=float(jitter),
            attempts=attempt,
            scale=scale,
        )

    raise np.linalg.LinAlgError(
        "Covariance is not positive definite after "
        f"{len(jitters)} Cholesky attempts; maximum absolute jitter was "
        f"{jitters[-1]:.3g}."
    ) from last_error


def evaluate_gaussian_eiv(
    observed: ArrayLike,
    prediction: ArrayLike,
    y_covariance: ArrayLike,
    x_sigma: ArrayLike,
    slope: ArrayLike,
    *,
    frozen_covariance: ArrayLike | None = None,
) -> LikelihoodEvaluation:
    """Evaluate the full Gaussian likelihood.

    In primary mode,

    ``Omega = V_y + diag((dm/dx * sigma_x)**2)``.

    If ``frozen_covariance`` is supplied, it is used as ``Omega`` directly.
    This mode is intended for a clearly labelled sensitivity analysis.
    """

    y = _as_vector(observed, "observed")
    n_observations = y.size
    mean = _as_vector(prediction, "prediction", n_observations)
    derivative = _as_vector(slope, "slope", n_observations)
    sigma_x = _as_vector(x_sigma, "x_sigma", n_observations)
    if np.any(sigma_x < 0.0):
        raise ValueError("x_sigma must be nonnegative.")

    base_covariance = _as_covariance(y_covariance, n_observations, "y_covariance")
    base_covariance_id = covariance_identifier(base_covariance)

    if frozen_covariance is None:
        propagated_variance = np.square(derivative * sigma_x)
        covariance = base_covariance + np.diag(propagated_variance)
        covariance_mode = "parameter_dependent_eiv"
    else:
        covariance = _as_covariance(
            frozen_covariance, n_observations, "frozen_covariance"
        )
        covariance_mode = "frozen_sensitivity"

    covariance = 0.5 * (covariance + covariance.T)
    factorization = stable_cholesky(covariance)
    effective_covariance = covariance.copy()
    if factorization.jitter:
        effective_covariance += factorization.jitter * np.eye(n_observations)

    residual = y - mean
    whitened = np.linalg.solve(factorization.factor, residual)
    mahalanobis = float(whitened @ whitened)
    log_determinant = float(2.0 * np.sum(np.log(np.diag(factorization.factor))))
    normalizing_constant = float(n_observations * np.log(2.0 * np.pi))
    deviance = mahalanobis + log_determinant + normalizing_constant
    eigenvalues = np.linalg.eigvalsh(effective_covariance)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    condition_number = (
        float(max_eigenvalue / min_eigenvalue) if min_eigenvalue > 0.0 else float("inf")
    )

    if not np.isfinite(deviance):
        raise FloatingPointError("The Gaussian deviance is non-finite.")

    return LikelihoodEvaluation(
        deviance=float(deviance),
        log_likelihood=float(-0.5 * deviance),
        mahalanobis=mahalanobis,
        log_determinant=log_determinant,
        n_observations=n_observations,
        residual=residual,
        prediction=mean,
        slope=derivative,
        whitened_residual=whitened,
        covariance=effective_covariance,
        base_covariance_id=base_covariance_id,
        covariance_id=covariance_identifier(effective_covariance),
        covariance_mode=covariance_mode,
        cholesky_jitter=factorization.jitter,
        cholesky_attempts=factorization.attempts,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        condition_number=condition_number,
    )

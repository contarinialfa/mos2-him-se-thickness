"""Auditable SRIM interpolation and endpoint-policy sensitivities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from .deposition import IonizationProfile


@dataclass(frozen=True)
class ResamplingDiagnostics:
    """Metadata for one resampled source profile."""

    interpolation: str
    tail_policy: str
    step_nm: float
    original_maximum_depth_nm: float
    resampled_maximum_depth_nm: float
    clipped_negative_values: int
    minimum_unclipped_value: float


def resample_profile(
    profile: IonizationProfile,
    *,
    interpolation: str,
    tail_policy: str = "zero",
    step_nm: float = 0.05,
    endpoint_extension_nm: float = 1200.0,
) -> tuple[IonizationProfile, ResamplingDiagnostics]:
    """Return a fine-grid profile for interpolation/tail sensitivity checks.

    ``interpolation`` may be ``"linear"``, ``"pchip"``, or ``"cubic"``.
    The cubic option reproduces the legacy convention: the synthetic origin
    node is removed, values shallower than the first SRIM bin use that bin's
    value, and values beyond the final bin use either zero or the final value.
    Negative cubic overshoot is clipped and counted because negative electronic
    deposition is unphysical.

    ``tail_policy="endpoint"`` is retained only to quantify the legacy
    endpoint-clamp sensitivity. It does not assert a physical deposition tail.
    """

    if interpolation not in {"linear", "pchip", "cubic"}:
        raise ValueError("interpolation must be 'linear', 'pchip', or 'cubic'.")
    if tail_policy not in {"zero", "endpoint"}:
        raise ValueError("tail_policy must be 'zero' or 'endpoint'.")
    if not np.isfinite(step_nm) or step_nm <= 0.0:
        raise ValueError("step_nm must be finite and positive.")

    original_maximum = profile.maximum_depth_nm
    if tail_policy == "endpoint":
        if (
            not np.isfinite(endpoint_extension_nm)
            or endpoint_extension_nm <= original_maximum
        ):
            raise ValueError(
                "endpoint_extension_nm must exceed the original profile depth."
            )
        maximum = float(endpoint_extension_nm)
    else:
        maximum = original_maximum

    cell_count = int(np.ceil(maximum / step_nm))
    grid = np.linspace(0.0, maximum, cell_count + 1)
    right_value = (
        float(profile.deposition_ev_per_nm[-1])
        if tail_policy == "endpoint"
        else 0.0
    )

    if interpolation == "linear":
        values = np.interp(
            grid,
            profile.z_nm,
            profile.deposition_ev_per_nm,
            left=profile.deposition_ev_per_nm[0],
            right=right_value,
        )
    elif interpolation == "pchip":
        interpolator = PchipInterpolator(
            profile.z_nm,
            profile.deposition_ev_per_nm,
            extrapolate=False,
        )
        values = np.asarray(interpolator(np.minimum(grid, original_maximum)))
        beyond = grid > original_maximum
        values[beyond] = right_value
    else:
        # The legacy script interpolated only the tabulated SRIM bins and
        # clamped to the first/last values outside them.
        z = profile.z_nm
        deposition = profile.deposition_ev_per_nm
        if z[0] == 0.0:
            z = z[1:]
            deposition = deposition[1:]
        interpolator = interp1d(
            z,
            deposition,
            kind="cubic",
            bounds_error=False,
            fill_value=(float(deposition[0]), right_value),
            assume_sorted=True,
        )
        values = np.asarray(interpolator(grid), dtype=float)

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Interpolated profile contains non-finite values.")
    minimum_unclipped = float(np.min(values))
    clipped = int(np.count_nonzero(values < 0.0))
    values = np.maximum(values, 0.0)
    resampled = IonizationProfile(grid, values)
    return resampled, ResamplingDiagnostics(
        interpolation=interpolation,
        tail_policy=tail_policy,
        step_nm=float(step_nm),
        original_maximum_depth_nm=original_maximum,
        resampled_maximum_depth_nm=maximum,
        clipped_negative_values=clipped,
        minimum_unclipped_value=minimum_unclipped,
    )

"""Ionization profiles and exact exponential escape integrals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IonizationProfile:
    """Nonnegative piecewise-linear deposition with explicit endpoint policy.

    The first node must be at zero and represents the documented constant
    extension from the first SRIM bin to the origin.  Between nodes the profile
    is linear.  Beyond the final node the deposition is exactly zero; front
    escape therefore remains constant and back escape decays exponentially.
    """

    z_nm: np.ndarray
    deposition_ev_per_nm: np.ndarray

    def __post_init__(self) -> None:
        z = np.asarray(self.z_nm, dtype=float)
        deposition = np.asarray(self.deposition_ev_per_nm, dtype=float)
        if z.ndim != 1 or deposition.ndim != 1 or len(z) != len(deposition):
            raise ValueError("z_nm and deposition_ev_per_nm must align as vectors.")
        if len(z) < 2:
            raise ValueError("IonizationProfile requires at least two nodes.")
        if z[0] != 0.0 or np.any(np.diff(z) <= 0.0):
            raise ValueError("z_nm must start at zero and increase strictly.")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(deposition)):
            raise ValueError("IonizationProfile contains non-finite values.")
        if np.any(deposition < 0.0):
            raise ValueError("IonizationProfile must be nonnegative.")
        object.__setattr__(self, "z_nm", z.copy())
        object.__setattr__(self, "deposition_ev_per_nm", deposition.copy())

    @property
    def maximum_depth_nm(self) -> float:
        """Last tabulated depth; deposition is zero beyond it."""

        return float(self.z_nm[-1])

    def evaluate(self, depth_nm: np.ndarray | float) -> np.ndarray:
        """Evaluate the constant-origin, linear-interior, zero-tail profile."""

        query = np.asarray(depth_nm, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("Depth queries must be finite.")
        if np.any(query < 0.0):
            raise ValueError("Depth queries must be nonnegative.")
        return np.interp(
            query,
            self.z_nm,
            self.deposition_ev_per_nm,
            left=self.deposition_ev_per_nm[0],
            right=0.0,
        )

    def escape_integrals(
        self, upper_depth_nm: np.ndarray | float, escape_length_nm: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate exact front and back escape integrals.

        Each profile cell is integrated analytically as a linear function
        multiplied by an exponential.  The implementation accepts a nonuniform
        depth grid and remains stable when depth is much larger than the escape
        length.
        """

        upper = np.asarray(upper_depth_nm, dtype=float)
        if not np.all(np.isfinite(upper)) or np.any(upper < 0.0):
            raise ValueError("Upper depths must be finite and nonnegative.")
        lam = float(escape_length_nm)
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("escape_length_nm must be finite and positive.")

        z = self.z_nm
        profile = self.deposition_ev_per_nm
        widths = np.diff(z)
        slopes = np.diff(profile) / widths
        scaled_width = widths / lam
        decay = np.exp(-scaled_width)
        growth = -np.expm1(-scaled_width)

        front_cells = np.exp(-z[:-1] / lam) * (
            profile[:-1] * lam * growth
            + slopes * (lam * lam * growth - lam * widths * decay)
        )
        front_nodes = np.concatenate(([0.0], np.cumsum(front_cells)))

        back_cells = lam * (
            profile[:-1] * growth
            + slopes * (widths - lam * growth)
        )
        back_nodes = np.zeros_like(z)
        for index, (cell, cell_decay) in enumerate(zip(back_cells, decay)):
            back_nodes[index + 1] = back_nodes[index] * cell_decay + cell

        inside_depth = np.minimum(upper, z[-1])
        index = np.clip(
            np.searchsorted(z, inside_depth, side="right") - 1,
            0,
            len(z) - 2,
        )
        partial_width = inside_depth - z[index]
        d_left = profile[index]
        partial_slope = slopes[index]
        partial_decay = np.exp(-partial_width / lam)
        partial_growth = -np.expm1(-partial_width / lam)

        front = front_nodes[index] + np.exp(-z[index] / lam) * (
            d_left * lam * partial_growth
            + partial_slope
            * (
                lam * lam * partial_growth
                - lam * partial_width * partial_decay
            )
        )
        back = back_nodes[index] * partial_decay + lam * (
            d_left * partial_growth
            + partial_slope * (partial_width - lam * partial_growth)
        )

        beyond = upper > z[-1]
        if np.any(beyond):
            front = np.where(beyond, front_nodes[-1], front)
            tail_distance = np.maximum(upper - z[-1], 0.0)
            back = np.where(
                beyond,
                back_nodes[-1] * np.exp(-tail_distance / lam),
                back,
            )
        return np.asarray(front), np.asarray(back)

    def measured_components(
        self,
        thickness_nm: np.ndarray | float,
        scale: float,
        lambda_front_nm: float,
        lambda_back_nm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return front/back components with all lengths on the EELS axis.

        The dimensionally explicit measured-coordinate source is
        ``S(z) = D(z/s) / s``.  Changing variables to the raw SRIM coordinate
        gives ``u=t/s`` and ``lambda_raw=lambda_measured/s`` with no additional
        multiplicative factor.  Using ``D(z/s)`` instead would multiply both
        components by ``s`` and therefore change only the free overall
        amplitude ``C_o``; fitted escape lengths remain measured-axis nm.
        """

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

        raw_upper = thickness / scale_value
        front, _ = self.escape_integrals(
            raw_upper, lambda_front / scale_value
        )
        _, back = self.escape_integrals(
            raw_upper, lambda_back / scale_value
        )
        return front, back

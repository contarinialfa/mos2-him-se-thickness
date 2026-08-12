"""Validated readers for experimental observations and SRIM ionization data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ANGSTROM_PER_NM = 10.0
_KEY_THICKNESS_DECIMALS = 9


def _validate_spd(matrix: np.ndarray, *, name: str) -> None:
    """Require a finite, symmetric, strictly positive-definite matrix."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-13):
        raise ValueError(f"{name} is not symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    tolerance = (
        max(float(np.max(np.abs(eigenvalues))), 1.0)
        * matrix.shape[0]
        * np.finfo(float).eps
        * 100.0
    )
    if float(eigenvalues[0]) <= tolerance:
        raise ValueError(
            f"{name} is numerically singular or not positive definite "
            f"(minimum eigenvalue {eigenvalues[0]:.6g}, "
            f"tolerance {tolerance:.6g})."
        )
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:
        minimum = float(np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))))
        raise ValueError(
            f"{name} is not positive definite (minimum eigenvalue "
            f"{minimum:.6g})."
        ) from exc


@dataclass(frozen=True)
class ObservationData:
    """Experimental observations with covariance and explicit row metadata.

    Attributes
    ----------
    x
        Measured EELS thicknesses in nm.
    y
        Carbon-normalized intensity estimates.
    y_cov
        Covariance of ``y`` in the same row order.  Posterior-draw blocks are
        used when available; otherwise the corresponding rows are diagonal.
    x_sigma
        Pointwise thickness standard deviations in nm.
    dataset, group_id, row_id
        Stable metadata identifying each observation and preserving alignment
        through covariance subsetting and leave-one-flake-out analyses.
    """

    x: np.ndarray
    y: np.ndarray
    y_cov: np.ndarray
    x_sigma: np.ndarray
    dataset: np.ndarray
    group_id: np.ndarray
    row_id: np.ndarray

    def __post_init__(self) -> None:
        n_rows = len(self.x)
        vectors = {
            "x": self.x,
            "y": self.y,
            "x_sigma": self.x_sigma,
            "dataset": self.dataset,
            "group_id": self.group_id,
            "row_id": self.row_id,
        }
        for name, values in vectors.items():
            if np.asarray(values).ndim != 1 or len(values) != n_rows:
                raise ValueError(
                    f"{name} must be one-dimensional with {n_rows} rows."
                )
        if n_rows == 0:
            raise ValueError("ObservationData cannot be empty.")
        if self.y_cov.shape != (n_rows, n_rows):
            raise ValueError(
                f"y_cov must have shape {(n_rows, n_rows)}, "
                f"got {self.y_cov.shape}."
            )
        if not np.all(np.isfinite(self.x)) or np.any(self.x < 0.0):
            raise ValueError("x must contain finite, nonnegative thicknesses.")
        if not np.all(np.isfinite(self.y)):
            raise ValueError("y contains non-finite values.")
        if not np.all(np.isfinite(self.x_sigma)) or np.any(self.x_sigma < 0.0):
            raise ValueError("x_sigma must be finite and nonnegative.")
        if len(np.unique(self.row_id.astype(str))) != n_rows:
            raise ValueError("row_id values must be unique.")
        _validate_spd(self.y_cov, name="y_cov")

    def subset(self, mask: np.ndarray) -> ObservationData:
        """Return rows and their covariance principal submatrix.

        ``mask`` may be a Boolean mask of the same length as the data or a
        one-dimensional integer index array.  Metadata remains in the selected
        row order.
        """

        selector = np.asarray(mask)
        if selector.ndim != 1:
            raise ValueError("subset mask must be one-dimensional.")
        if selector.dtype == np.bool_:
            if len(selector) != len(self.x):
                raise ValueError(
                    f"Boolean mask has length {len(selector)}; "
                    f"expected {len(self.x)}."
                )
            index = np.flatnonzero(selector)
        elif np.issubdtype(selector.dtype, np.integer):
            index = selector.astype(int, copy=False)
            if np.any(index < 0) or np.any(index >= len(self.x)):
                raise IndexError("subset indices are outside the data range.")
        else:
            raise TypeError("subset mask must contain Booleans or integer indices.")
        if len(index) == 0:
            raise ValueError("subset cannot be empty.")
        return ObservationData(
            x=self.x[index].copy(),
            y=self.y[index].copy(),
            y_cov=self.y_cov[np.ix_(index, index)].copy(),
            x_sigma=self.x_sigma[index].copy(),
            dataset=self.dataset[index].copy(),
            group_id=self.group_id[index].copy(),
            row_id=self.row_id[index].copy(),
        )


def _observation_key(dataset: str, group_id: int, thickness_nm: float) -> tuple:
    """Canonical key used to align summary rows and posterior arrays."""

    return (
        str(dataset).strip(),
        int(group_id),
        round(float(thickness_nm), _KEY_THICKNESS_DECIMALS),
    )


def _posterior_payload(path: Path) -> dict[str, object]:
    """Read and validate one posterior-ratio NPZ file."""

    required = {"ratio_samples", "thickness_nm", "group_ids", "dataset_label"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ arrays: {sorted(missing)}")
        samples = np.asarray(archive["ratio_samples"], dtype=float)
        thickness = np.asarray(archive["thickness_nm"], dtype=float).reshape(-1)
        group_ids = np.asarray(archive["group_ids"]).reshape(-1)
        labels = np.asarray(archive["dataset_label"]).reshape(-1)

    if labels.size != 1:
        raise ValueError(f"{path}: dataset_label must contain exactly one value.")
    dataset = str(labels[0]).strip()
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(f"{path}: ratio_samples must be a 2-D draw matrix.")
    if samples.shape[1] != len(group_ids) or len(group_ids) != len(thickness):
        raise ValueError(
            f"{path}: sample columns, group_ids, and thickness_nm do not align."
        )
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(thickness)):
        raise ValueError(f"{path}: posterior arrays contain non-finite values.")

    keys = [
        _observation_key(dataset, int(group_id), float(value))
        for group_id, value in zip(group_ids, thickness)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: posterior observation keys are not unique.")
    return {"path": path, "dataset": dataset, "keys": keys, "samples": samples}


def _summary_errors(frame: pd.DataFrame) -> np.ndarray:
    errors = 0.5 * (
        frame["normalized_err_lower_68"].to_numpy(dtype=float)
        + frame["normalized_err_upper_68"].to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
        raise ValueError("Summary contains non-finite or non-positive errors.")
    return errors


def load_observations(
    summary_path: Path | str,
    posterior_dir: Path | str | None = None,
    pointwise_fraction: float = 0.10,
    covariance_mode: str = "full",
) -> ObservationData:
    """Load aligned observations with posterior block covariance when available.

    Parameters
    ----------
    summary_path
        Combined Bayesian summary CSV.
    posterior_dir
        Optional directory searched recursively for
        ``*_posterior_ratio_samples.npz`` files created by
        :mod:`scripts.bayesian_normalize`.  In ``"full"`` mode, a complete
        per-flake key match replaces that flake's summary medians and diagonal
        errors with posterior means and sample covariance.  Flakes without an
        available NPZ fall back to the summary diagonal.
    pointwise_fraction
        Point-to-point thickness uncertainty as a fraction of measured
        thickness.
    covariance_mode
        ``"full"`` for posterior blocks when available or ``"diagonal"`` to
        use only symmetrized summary errors.
    """

    if covariance_mode not in {"full", "diagonal"}:
        raise ValueError("covariance_mode must be 'full' or 'diagonal'.")
    if not np.isfinite(pointwise_fraction) or pointwise_fraction < 0.0:
        raise ValueError("pointwise_fraction must be finite and nonnegative.")

    path = Path(summary_path)
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "group_id",
        "thickness_nm",
        "normalized_median",
        "normalized_err_lower_68",
        "normalized_err_upper_68",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no observations.")

    dataset = frame["dataset"].astype(str).str.strip().to_numpy()
    group_id = frame["group_id"].to_numpy(dtype=int)
    x = frame["thickness_nm"].to_numpy(dtype=float)
    keys = [
        _observation_key(label, group, thickness)
        for label, group, thickness in zip(dataset, group_id, x)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Summary observation keys are not unique.")

    # Copy because recent pandas versions may expose a read-only NumPy view;
    # posterior-backed rows are deliberately replaced by posterior means.
    y = frame["normalized_median"].to_numpy(dtype=float).copy()
    errors = _summary_errors(frame)
    y_cov = np.diag(errors**2)

    if covariance_mode == "full" and posterior_dir is not None:
        posterior_root = Path(posterior_dir)
        if not posterior_root.is_dir():
            raise FileNotFoundError(
                f"Posterior directory does not exist: {posterior_root}"
            )
        payloads = [
            _posterior_payload(candidate)
            for candidate in sorted(
                posterior_root.rglob("*_posterior_ratio_samples.npz")
            )
        ]
        for label in dict.fromkeys(dataset):
            row_index = np.flatnonzero(dataset == label)
            expected = {keys[index] for index in row_index}
            same_label = [
                payload for payload in payloads if payload["dataset"] == label
            ]
            exact = [
                payload for payload in same_label
                if set(payload["keys"]) == expected
            ]
            if len(exact) > 1:
                paths = ", ".join(str(payload["path"]) for payload in exact)
                raise ValueError(
                    f"Multiple posterior files exactly match {label}: {paths}"
                )
            if not exact:
                if same_label:
                    paths = ", ".join(str(payload["path"]) for payload in same_label)
                    raise ValueError(
                        f"Posterior keys for {label} do not match the summary "
                        f"rows. Candidate files: {paths}"
                    )
                continue

            payload = exact[0]
            sample_matrix = np.asarray(payload["samples"], dtype=float)
            posterior_index = {
                key: index for index, key in enumerate(payload["keys"])
            }
            ordered_columns = [posterior_index[keys[index]] for index in row_index]
            ordered_samples = sample_matrix[:, ordered_columns]
            block = np.atleast_2d(
                np.cov(ordered_samples, rowvar=False, ddof=1)
            )
            _validate_spd(block, name=f"posterior covariance for {label}")
            y[row_index] = np.mean(ordered_samples, axis=0)
            y_cov[np.ix_(row_index, row_index)] = block

    if not np.all(np.isfinite(y)):
        raise ValueError("Observation means contain non-finite values.")
    _validate_spd(y_cov, name="assembled y covariance")

    row_id = np.asarray(
        [
            f"{label}|group={group}|thickness_nm={thickness:.12g}"
            for label, group, thickness in zip(dataset, group_id, x)
        ],
        dtype=str,
    )
    return ObservationData(
        x=x,
        y=y,
        y_cov=y_cov,
        x_sigma=pointwise_fraction * x,
        dataset=dataset,
        group_id=group_id,
        row_id=row_id,
    )


def load_srim(
    path: Path | str, include_recoils: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Load an SRIM ``IONIZ.txt`` profile in nm and eV/nm.

    SRIM tabulates depth in Angstrom and ion/recoil ionization in
    eV/(Angstrom ion).  The returned deposition is ion ionization alone unless
    ``include_recoils`` is true, in which case both columns are summed.  A
    constant-origin node is prepended at ``z=0``; extrapolation beyond the last
    node is deliberately left to :class:`se_model.deposition.IonizationProfile`.
    """

    source = Path(path)
    depth_nm: list[float] = []
    deposition_ev_per_nm: list[float] = []
    for line in source.read_text(encoding="latin-1").splitlines():
        tokens = line.split()
        if len(tokens) != 3:
            continue
        try:
            depth_angstrom, ionization, recoil_ionization = (
                float(token) for token in tokens
            )
        except ValueError:
            continue
        value = ionization + recoil_ionization if include_recoils else ionization
        depth_nm.append(depth_angstrom / ANGSTROM_PER_NM)
        deposition_ev_per_nm.append(value * ANGSTROM_PER_NM)

    if len(depth_nm) < 2:
        raise ValueError(f"{source} did not yield a usable ionization profile.")
    z = np.asarray(depth_nm, dtype=float)
    deposition = np.asarray(deposition_ev_per_nm, dtype=float)
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(deposition)):
        raise ValueError(f"{source} contains non-finite SRIM values.")
    if np.any(z < 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError(f"{source} depth grid must be nonnegative and increasing.")
    if np.any(deposition < 0.0):
        raise ValueError(f"{source} contains negative ionization values.")
    if z[0] > 0.0:
        z = np.concatenate(([0.0], z))
        deposition = np.concatenate(([deposition[0]], deposition))
    elif z[0] != 0.0:
        raise ValueError(f"{source} has an invalid first depth {z[0]}.")
    return z, deposition

"""Propagate archived intensity-posterior draws through the three S0 fits.

This is a deliberately separate sensitivity analysis.  It is not the primary
Gaussian working likelihood, a nonparametric bootstrap, or a new Bayesian
posterior for the paper-model parameters.

For each replicate, one *joint* ratio vector is selected from each of the
three independently sampled flake posteriors.  Selecting a complete row keeps
the shared-carbon-reference dependence among thickness groups within a flake;
the three row selections are generated from independent deterministic random
streams.  The resulting 15-element latent intensity vector is fitted with the
same full-covariance, parameter-dependent EIV estimator used by the corrected
production analysis.

The archived intensity covariance ``V_y`` is held fixed in every replicate,
but the estimator's total covariance remains parameter dependent:
``Omega(theta) = V_y + diag[(m'(theta) sigma_x)^2]``.  No second random
intensity-noise vector is sampled.  Across-replicate uncertainty is calculated
only from the fitted point estimates.  Per-fit Hessian standard errors and
covariance matrices are intentionally neither exported nor added to the
between-draw spread; doing so would count the same intensity uncertainty twice.

Outputs are descriptive propagation diagnostics.  Quantiles must not be
reported as formal confidence/credible intervals, and replicate frequencies
must not be reported as p-values or mechanism probabilities.  AICc and its
within-replicate differences are descriptive scores from the corrected
Gaussian EIV working likelihood, not formal evidence probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.model_sensitivity import _normalization_quality  # noqa: E402
from se_model.data import ObservationData, load_observations, load_srim  # noqa: E402
from se_model.deposition import IonizationProfile  # noqa: E402
from se_model.fit import FitResult, fit_model  # noqa: E402
from se_model.likelihood import covariance_identifier  # noqa: E402
from se_model.models import (  # noqa: E402
    ModelPrediction,
    ModelSpec,
    paper_model_specs,
    predict_model,
)

DEFAULT_POSTERIOR_DIR = (
    REPOSITORY_ROOT / "data" / "reference_posteriors" / "transmission"
)
DEFAULT_SUMMARY = DEFAULT_POSTERIOR_DIR / "combined_bayesian_summary.csv"
DEFAULT_LONG_IONIZ = (
    REPOSITORY_ROOT / "data" / "SRIM 30 keV - Monolayer Long" / "IONIZ.txt"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "posterior_draw_sensitivity"
DEFAULT_SMOKE_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs" / "posterior_draw_sensitivity_smoke"
)

EXPECTED_DATASETS = ("Dataset 1", "Dataset 2", "Dataset 3")
KEY_THICKNESS_DECIMALS = 9

# The model specifications own the reviewed S0 anchors.  Referencing them
# directly avoids a second stale copy in this sensitivity script.
REVIEWED_ANCHORS: dict[str, tuple[float, ...]] = {
    spec.key: tuple(float(value) for value in spec.initial)
    for spec in paper_model_specs()
}

SUMMARY_METRICS = (
    "deviance",
    "mahalanobis",
    "mahalanobis_per_nominal_dof",
    "aicc",
    "delta_aicc_within_replicate",
    "C_o",
    "C_b",
    "scale",
    "lambda_front_nm",
    "lambda_back_nm",
    "lambda_shared_nm",
    "C_b_unit_area_equivalent",
    "peak_thickness_nm",
    "peak_intensity",
    "model_assigned_back_fraction_at_peak",
)

METHOD_BOUNDARY = {
    "name": "posterior-draw propagation sensitivity",
    "posterior_draw_role": (
        "One complete joint ratio draw is selected independently for each "
        "flake and concatenated into one latent 15-point intensity vector."
    ),
    "intensity_covariance_role": (
        "The archived intensity covariance V_y is fixed. The corrected "
        "estimator uses Omega(theta)=V_y+diag((m'(theta)*sigma_x)^2), so the "
        "EIV contribution remains parameter dependent. No additional "
        "intensity-noise draw is generated."
    ),
    "thickness_role": (
        "Measured thicknesses remain fixed at their reported values while the "
        "declared pointwise thickness uncertainty remains in the corrected "
        "parameter-dependent EIV weighting. This sensitivity resamples only "
        "the intensity-normalization posterior."
    ),
    "aggregation": (
        "Only the between-draw distribution of fitted point estimates is "
        "summarized. Per-fit Hessian standard errors/covariances are not "
        "pooled, added, or exported."
    ),
    "score_role": (
        "AICc and within-replicate delta AICc are descriptive scores from the "
        "corrected Gaussian EIV working likelihood. They are not formal model "
        "probabilities or mechanism evidence probabilities."
    ),
    "claim_limit": (
        "This is not a bootstrap, a parameter posterior, a coverage "
        "calculation, a p-value procedure, or a mechanism-selection analysis."
    ),
}


@dataclass(frozen=True)
class PosteriorBlock:
    """One exactly aligned within-flake posterior-ratio draw matrix."""

    dataset: str
    path: Path
    row_indices: np.ndarray
    row_keys: tuple[tuple[str, int, float], ...]
    samples: np.ndarray

    @property
    def n_draws(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.samples.shape[1])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(vector: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(vector, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"posterior-draw-latent-vector-v1\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _prepare_output_directory(requested: Path) -> Path:
    """Create a fresh staging directory; existing results are immutable."""

    if requested.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {requested}"
        )
    requested.parent.mkdir(parents=True, exist_ok=True)
    staging = requested.with_name(f".{requested.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    return staging


def _finalize_output_directory(staging: Path, requested: Path) -> None:
    if requested.exists():
        raise FileExistsError(f"Cannot publish over existing output: {requested}")
    staging.replace(requested)


def _observation_key(
    dataset: str,
    group_id: int,
    thickness_nm: float,
) -> tuple[str, int, float]:
    return (
        str(dataset).strip(),
        int(group_id),
        round(float(thickness_nm), KEY_THICKNESS_DECIMALS),
    )


def _integer_ids(values: Any, *, label: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{label} contains non-finite values.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{label} must contain integer-valued identifiers.")
    return rounded.astype(np.int64)


def _read_posterior_candidate(path: Path) -> dict[str, Any]:
    required = {"ratio_samples", "thickness_nm", "group_ids", "dataset_label"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ arrays: {sorted(missing)}")
        samples = np.asarray(archive["ratio_samples"], dtype=float)
        thickness = np.asarray(archive["thickness_nm"], dtype=float).reshape(-1)
        group_ids = _integer_ids(
            archive["group_ids"],
            label=f"{path}: group_ids",
        )
        labels = np.asarray(archive["dataset_label"]).astype(str).reshape(-1)

    if labels.size != 1:
        raise ValueError(f"{path}: dataset_label must contain exactly one value.")
    dataset = str(labels[0]).strip()
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(f"{path}: ratio_samples must be a two-dimensional matrix.")
    if samples.shape[1] != group_ids.size or group_ids.size != thickness.size:
        raise ValueError(f"{path}: posterior columns and row metadata do not align.")
    if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
        raise ValueError(f"{path}: ratio samples must be finite and positive.")
    if not np.all(np.isfinite(thickness)):
        raise ValueError(f"{path}: thickness metadata contain non-finite values.")

    keys = tuple(
        _observation_key(dataset, int(group_id), float(value))
        for group_id, value in zip(group_ids, thickness, strict=True)
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: posterior observation keys are not unique.")
    return {
        "dataset": dataset,
        "path": path.resolve(),
        "samples": samples,
        "keys": keys,
    }


def _load_posterior_blocks(
    summary_path: Path,
    posterior_dir: Path,
    *,
    expected_datasets: Sequence[str] = EXPECTED_DATASETS,
) -> tuple[PosteriorBlock, ...]:
    """Require exactly one complete NPZ block for every expected dataset."""

    frame = pd.read_csv(summary_path)
    required = {"dataset", "group_id", "thickness_nm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{summary_path} contains no observations.")

    dataset = frame["dataset"].astype(str).str.strip().to_numpy()
    group_ids = _integer_ids(
        pd.to_numeric(frame["group_id"], errors="coerce").to_numpy(),
        label=f"{summary_path}: group_id",
    )
    thickness = frame["thickness_nm"].to_numpy(dtype=float)
    summary_keys = tuple(
        _observation_key(label, group, value)
        for label, group, value in zip(
            dataset,
            group_ids,
            thickness,
            strict=True,
        )
    )
    if len(summary_keys) != len(set(summary_keys)):
        raise ValueError("Summary observation keys are not unique.")

    observed_order = tuple(dict.fromkeys(dataset.astype(str)))
    expected_order = tuple(str(label) for label in expected_datasets)
    if observed_order != expected_order:
        raise ValueError(
            "Summary datasets or their order differ from the required archive: "
            f"observed={observed_order}, expected={expected_order}."
        )

    candidates = [
        _read_posterior_candidate(path)
        for path in sorted(posterior_dir.glob("*_posterior_ratio_samples.npz"))
    ]
    if len(candidates) != len(expected_order):
        raise ValueError(
            f"Expected exactly {len(expected_order)} posterior-ratio NPZ blocks; "
            f"found {len(candidates)}."
        )

    by_dataset: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        label = str(candidate["dataset"])
        if label in by_dataset:
            raise ValueError(f"Multiple posterior-ratio blocks claim {label}.")
        by_dataset[label] = candidate
    if set(by_dataset) != set(expected_order):
        raise ValueError(
            "Posterior-ratio dataset labels differ from the required archive: "
            f"observed={sorted(by_dataset)}, expected={sorted(expected_order)}."
        )

    blocks: list[PosteriorBlock] = []
    covered_rows: list[int] = []
    for label in expected_order:
        row_indices = np.flatnonzero(dataset == label)
        required_keys = tuple(summary_keys[index] for index in row_indices)
        candidate = by_dataset[label]
        candidate_keys = tuple(candidate["keys"])
        if set(candidate_keys) != set(required_keys):
            missing_keys = sorted(set(required_keys).difference(candidate_keys))
            extra_keys = sorted(set(candidate_keys).difference(required_keys))
            raise ValueError(
                f"{label}: posterior keys do not exactly match summary rows; "
                f"missing={missing_keys}, extra={extra_keys}."
            )
        candidate_index = {key: index for index, key in enumerate(candidate_keys)}
        ordered_columns = [candidate_index[key] for key in required_keys]
        ordered_samples = np.asarray(
            candidate["samples"][:, ordered_columns],
            dtype=float,
        )
        blocks.append(
            PosteriorBlock(
                dataset=label,
                path=Path(candidate["path"]),
                row_indices=row_indices,
                row_keys=required_keys,
                samples=ordered_samples,
            )
        )
        covered_rows.extend(row_indices.tolist())

    if sorted(covered_rows) != list(range(len(frame))):
        raise RuntimeError(
            "Posterior blocks do not cover every summary row exactly once."
        )
    return tuple(blocks)


def _validate_joint_archive(
    base: ObservationData,
    blocks: Sequence[PosteriorBlock],
    posterior_dir: Path,
) -> dict[str, Any]:
    """Bind the keyed joint archive, individual draws, and fitted weights."""

    joint_path = posterior_dir / "joint_normalization_posterior.npz"
    required = {
        "posterior_mean",
        "covariance",
        "dataset",
        "group_id",
        "thickness_nm",
        "draws_per_block",
    }
    with np.load(joint_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"{joint_path} is missing joint arrays: {sorted(missing)}"
            )
        joint_mean = np.asarray(archive["posterior_mean"], dtype=float)
        joint_covariance = np.asarray(archive["covariance"], dtype=float)
        joint_dataset = np.asarray(archive["dataset"]).astype(str).reshape(-1)
        joint_group_ids = _integer_ids(
            archive["group_id"],
            label=f"{joint_path}: group_id",
        )
        joint_thickness = np.asarray(
            archive["thickness_nm"],
            dtype=float,
        ).reshape(-1)
        draws_per_block = _integer_ids(
            archive["draws_per_block"],
            label=f"{joint_path}: draws_per_block",
        )

    n_rows = len(base.x)
    if joint_mean.shape != (n_rows,):
        raise ValueError(
            f"{joint_path}: posterior_mean has shape {joint_mean.shape}; "
            f"expected {(n_rows,)}."
        )
    if joint_covariance.shape != (n_rows, n_rows):
        raise ValueError(
            f"{joint_path}: covariance has shape {joint_covariance.shape}; "
            f"expected {(n_rows, n_rows)}."
        )
    metadata_lengths = {
        len(joint_dataset),
        len(joint_group_ids),
        len(joint_thickness),
        len(draws_per_block),
    }
    if metadata_lengths != {n_rows}:
        raise ValueError(f"{joint_path}: joint row metadata do not align.")
    if not (
        np.all(np.isfinite(joint_mean))
        and np.all(np.isfinite(joint_covariance))
        and np.all(np.isfinite(joint_thickness))
    ):
        raise ValueError(f"{joint_path}: joint arrays contain non-finite values.")

    expected_keys = tuple(
        _observation_key(dataset, int(group_id), float(thickness))
        for dataset, group_id, thickness in zip(
            base.dataset,
            base.group_id,
            base.x,
            strict=True,
        )
    )
    joint_keys = tuple(
        _observation_key(dataset, int(group_id), float(thickness))
        for dataset, group_id, thickness in zip(
            joint_dataset,
            joint_group_ids,
            joint_thickness,
            strict=True,
        )
    )
    if joint_keys != expected_keys:
        raise ValueError(
            "Joint-normalization row keys/order differ from the observations."
        )

    draw_mean = np.empty(n_rows, dtype=float)
    draw_covariance = np.zeros((n_rows, n_rows), dtype=float)
    expected_draw_counts = np.empty(n_rows, dtype=np.int64)
    for block in blocks:
        draw_mean[block.row_indices] = np.mean(block.samples, axis=0)
        block_covariance = np.atleast_2d(
            np.cov(block.samples, rowvar=False, ddof=1)
        )
        draw_covariance[np.ix_(block.row_indices, block.row_indices)] = (
            block_covariance
        )
        expected_draw_counts[block.row_indices] = block.n_draws

    tolerances = {"rtol": 2e-12, "atol": 2e-14}
    if not np.allclose(joint_mean, draw_mean, **tolerances):
        raise ValueError(
            "Joint posterior means do not reproduce individual ratio draws."
        )
    if not np.allclose(joint_covariance, draw_covariance, **tolerances):
        raise ValueError(
            "Joint covariance does not reproduce the block draw covariances "
            "with zero cross-flake covariance."
        )
    if not np.array_equal(draws_per_block, expected_draw_counts):
        raise ValueError(
            "Joint draws_per_block does not match individual ratio archives."
        )
    if not np.allclose(base.y, joint_mean, **tolerances):
        raise ValueError(
            "Observation means used by the fit differ from the joint archive."
        )
    if not np.allclose(base.y_cov, joint_covariance, **tolerances):
        raise ValueError(
            "Weighting covariance used by the fit differs from the joint archive."
        )

    return {
        "joint_path": joint_path,
        "joint_sha256": _sha256(joint_path),
        "row_count": n_rows,
        "row_keys_exact": True,
        "means_match_individual_draws": True,
        "block_covariances_match_individual_draws": True,
        "cross_flake_covariance_zero": True,
        "draw_counts_match": True,
        "fit_means_match_joint_archive": True,
        "fit_covariance_matches_joint_archive": True,
        "covariance_id": covariance_identifier(joint_covariance),
    }


def _draw_indices(
    blocks: Sequence[PosteriorBlock],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Select rows using one independent deterministic stream per flake."""

    if replicates < 1:
        raise ValueError("replicates must be positive.")
    if not blocks:
        raise ValueError("At least one posterior block is required.")
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(len(blocks))
    selected = np.empty((replicates, len(blocks)), dtype=np.int64)
    for column, (block, child) in enumerate(zip(blocks, child_sequences, strict=True)):
        generator = np.random.default_rng(child)
        selected[:, column] = generator.integers(
            0,
            block.n_draws,
            size=replicates,
        )
    return selected


def _latent_vector(
    base: ObservationData,
    blocks: Sequence[PosteriorBlock],
    selected_indices: Sequence[int],
) -> np.ndarray:
    """Assemble one fully posterior-backed vector in observation row order."""

    if len(blocks) != len(selected_indices):
        raise ValueError("One selected draw index is required for each block.")
    latent = np.full(base.y.shape, np.nan, dtype=float)
    for block, selected in zip(blocks, selected_indices, strict=True):
        index = int(selected)
        if not 0 <= index < block.n_draws:
            raise IndexError(
                f"{block.dataset}: draw index {index} is outside [0, {block.n_draws})."
            )
        latent[block.row_indices] = block.samples[index]
    if not np.all(np.isfinite(latent)):
        raise RuntimeError(
            "Latent vector is not completely covered by posterior draws."
        )
    return latent


def _replace_intensity(base: ObservationData, latent: np.ndarray) -> ObservationData:
    """Keep the archived covariance as weights while replacing only y."""

    return ObservationData(
        x=base.x.copy(),
        y=np.asarray(latent, dtype=float).copy(),
        y_cov=base.y_cov.copy(),
        x_sigma=base.x_sigma.copy(),
        dataset=base.dataset.copy(),
        group_id=base.group_id.copy(),
        row_id=base.row_id.copy(),
    )


def _fit_seed(seed: int, replicate_id: int, model_key: str) -> int:
    payload = f"{int(seed)}\0{int(replicate_id)}\0{model_key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _anchored_specs() -> tuple[ModelSpec, ...]:
    specs = paper_model_specs()
    for spec in specs:
        anchor = REVIEWED_ANCHORS[spec.key]
        if anchor != tuple(spec.initial):
            raise RuntimeError(f"Reviewed anchor drifted for {spec.key}.")
        if len(anchor) != len(spec.parameter_names):
            raise RuntimeError(f"Reviewed anchor does not match {spec.key}.")
    return specs


def _escape_lengths(
    spec: ModelSpec,
    result: FitResult,
) -> tuple[float, float, float | None]:
    if spec.escape_mode == "fixed_symmetric":
        return spec.lambda_front_nm, spec.lambda_front_nm, None
    if spec.escape_mode == "free_symmetric":
        shared = result.params["lambda_shared_nm"]
        return shared, shared, shared
    return spec.lambda_front_nm, result.params["lambda_back_nm"], None


def _fit_row(
    *,
    replicate_id: int,
    latent_id: str,
    fit_seed: int,
    spec: ModelSpec,
    result: FitResult,
    profile: IonizationProfile,
    peak_grid: np.ndarray,
) -> dict[str, Any]:
    prediction = predict_model(
        spec,
        profile,
        peak_grid,
        result.params,
        return_components=True,
    )
    if not isinstance(prediction, ModelPrediction):
        raise TypeError("Component prediction did not return ModelPrediction.")
    peak_index = int(np.argmax(prediction.total))
    peak_total = float(prediction.total[peak_index])
    peak_back = float(prediction.back[peak_index])
    lambda_front, lambda_back, lambda_shared = _escape_lengths(spec, result)
    unit_area = result.params["C_b"] * lambda_back / lambda_front
    nominal_dof = result.n_observations - result.n_free_parameters
    free_indices = [
        result.parameter_names.index(name) for name in result.free_parameter_names
    ]
    if free_indices and result.hessian_positive_definite:
        correlation = result.correlation[np.ix_(free_indices, free_indices)]
        off_diagonal = correlation - np.eye(len(free_indices))
        maximum_correlation = float(np.max(np.abs(off_diagonal)))
    else:
        maximum_correlation = np.nan
    accepted_objectives = sorted(
        record.objective
        for record in result.starts
        if record.accepted and record.objective is not None
    )
    near_best = (
        sum(value - accepted_objectives[0] <= 1e-6 for value in accepted_objectives)
        if accepted_objectives
        else 0
    )

    row: dict[str, Any] = {
        "replicate_id": replicate_id,
        "latent_vector_id": latent_id,
        "model_key": spec.key,
        "model_label": spec.label,
        "fit_seed": fit_seed,
        "n_observations": result.n_observations,
        "n_parameters": result.n_free_parameters,
        "nominal_dof": nominal_dof,
        "deviance": result.deviance,
        "log_likelihood": result.log_likelihood,
        "mahalanobis": result.likelihood.mahalanobis,
        "mahalanobis_per_nominal_dof": (
            result.likelihood.mahalanobis / nominal_dof if nominal_dof > 0 else np.nan
        ),
        "aic": result.aic,
        "aicc": result.aicc,
        "bic": result.bic,
        "delta_aicc_within_replicate": np.nan,
        "C_o": result.params["C_o"],
        "C_b": result.params["C_b"],
        "scale": result.params["scale"],
        "lambda_front_nm": lambda_front,
        "lambda_back_nm": lambda_back,
        "lambda_shared_nm": lambda_shared,
        "C_b_unit_area_equivalent": unit_area,
        "peak_thickness_nm": float(peak_grid[peak_index]),
        "peak_intensity": peak_total,
        "peak_at_grid_boundary": peak_index in {0, len(peak_grid) - 1},
        "model_assigned_back_fraction_at_peak": (
            peak_back / peak_total if peak_total > 0.0 else np.nan
        ),
        "optimizer_success": result.success,
        "at_lower_bound": ";".join(result.at_lower_bound),
        "at_upper_bound": ";".join(result.at_upper_bound),
        "accepted_starts": sum(record.accepted for record in result.starts),
        "attempted_starts": result.n_starts,
        "near_best_starts": near_best,
        "best_start_index": result.best_start_index,
        "hessian_positive_definite_diagnostic_only": (result.hessian_positive_definite),
        "hessian_rank_diagnostic_only": result.hessian_rank,
        "hessian_condition_diagnostic_only": result.hessian_condition_number,
        "maximum_parameter_correlation_diagnostic_only": maximum_correlation,
        "base_covariance_id": result.likelihood.base_covariance_id,
        "effective_covariance_id": result.likelihood.covariance_id,
        "cholesky_jitter": result.likelihood.cholesky_jitter,
    }
    return row


def _add_within_replicate_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Never trust a pre-existing delta column: callers may pass a partially
    # assembled frame, stale values from a prior calculation, or failed
    # replicates.  A delta is defined only for one complete, finite three-model
    # comparison within the same posterior draw.
    result["delta_aicc_within_replicate"] = np.nan
    if result.empty or not {"replicate_id", "model_key", "aicc"}.issubset(result):
        return result
    expected_models = set(REVIEWED_ANCHORS)
    for replicate_id, indices in result.groupby("replicate_id").groups.items():
        subset = result.loc[indices]
        if set(subset["model_key"]) != expected_models or len(subset) != 3:
            continue
        if not np.all(np.isfinite(subset["aicc"])):
            continue
        best = float(subset["aicc"].min())
        result.loc[indices, "delta_aicc_within_replicate"] = subset["aicc"] - best
    return result


def _summarize_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize only between-draw point estimates and descriptive fit metrics."""

    columns = (
        "model_key",
        "metric",
        "n_point_estimates",
        "mean",
        "between_draw_sd",
        "minimum",
        "q02_5",
        "q16",
        "median",
        "q84",
        "q97_5",
        "maximum",
        "aggregation",
    )
    if frame.empty or "model_key" not in frame:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for model_key, subset in frame.groupby("model_key", sort=False):
        for metric in SUMMARY_METRICS:
            if metric not in subset:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            quantiles = np.quantile(values, [0.025, 0.16, 0.50, 0.84, 0.975])
            rows.append(
                {
                    "model_key": model_key,
                    "metric": metric,
                    "n_point_estimates": int(values.size),
                    "mean": float(np.mean(values)),
                    "between_draw_sd": (
                        float(np.std(values, ddof=1)) if values.size > 1 else np.nan
                    ),
                    "minimum": float(np.min(values)),
                    "q02_5": float(quantiles[0]),
                    "q16": float(quantiles[1]),
                    "median": float(quantiles[2]),
                    "q84": float(quantiles[3]),
                    "q97_5": float(quantiles[4]),
                    "maximum": float(np.max(values)),
                    "aggregation": ("between_posterior_draw_point_estimates_only"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _integrity_issues(
    frame: pd.DataFrame,
    *,
    n_starts: int,
) -> list[dict[str, Any]]:
    """Classify optimization errors separately from diagnostic warnings."""

    if frame.empty:
        return []
    issues: list[dict[str, Any]] = []
    nesting_violations: list[str] = []
    for replicate_id, subset in frame.groupby("replicate_id"):
        by_model = {
            row.model_key: row
            for row in subset.itertuples(index=False)
        }
        if set(by_model) != set(REVIEWED_ANCHORS):
            continue
        fixed = float(by_model["symmetric_fixed"].deviance)
        for flexible_key in ("symmetric_free", "asymmetric"):
            flexible = float(by_model[flexible_key].deviance)
            if flexible > fixed + 1e-5:
                nesting_violations.append(
                    f"replicate={replicate_id},model={flexible_key},"
                    f"delta_deviance={flexible - fixed:.9g}"
                )
    if nesting_violations:
        issues.append(
            {
                "severity": "error",
                "code": "nested_model_optimization_violation",
                "count": len(nesting_violations),
                "cases": nesting_violations,
                "message": (
                    "A flexible model finished above its nested fixed-model "
                    "deviance, indicating a missed optimizer minimum."
                ),
            }
        )

    warning_definitions = (
        (
            "parameter_bound",
            frame["at_lower_bound"].fillna("").ne("")
            | frame["at_upper_bound"].fillna("").ne(""),
            "Point estimates reached one or more declared parameter bounds.",
        ),
        (
            "hessian_non_positive_definite_diagnostic_only",
            ~frame["hessian_positive_definite_diagnostic_only"].astype(bool),
            (
                "The point estimate completed, but its Hessian diagnostic was "
                "not positive definite; no Hessian uncertainty is aggregated."
            ),
        ),
        (
            "peak_at_grid_boundary",
            frame["peak_at_grid_boundary"].astype(bool),
            "A reported peak occurred at the edge of the 0--400 nm search grid.",
        ),
    )
    for code, mask, message in warning_definitions:
        affected = frame.loc[mask, ["replicate_id", "model_key"]]
        if affected.empty:
            continue
        issues.append(
            {
                "severity": "warning",
                "code": code,
                "count": len(affected),
                "cases": [
                    f"replicate={row.replicate_id},model={row.model_key}"
                    for row in affected.itertuples(index=False)
                ],
                "message": message,
            }
        )

    if n_starts > 1:
        unreproduced = frame[frame["near_best_starts"] < 2]
        if not unreproduced.empty:
            issues.append(
                {
                    "severity": "warning",
                    "code": "best_minimum_not_reproduced",
                    "count": len(unreproduced),
                    "cases": [
                        f"replicate={row.replicate_id},model={row.model_key}"
                        for row in unreproduced.itertuples(index=False)
                    ],
                    "message": (
                        "Only one accepted start reached within 1e-6 deviance "
                        "of the reported best minimum."
                    ),
                }
            )
    return issues


def _require_summary_manifest_match(
    summary_path: Path,
    normalization_quality: Mapping[str, Any],
) -> None:
    manifest = normalization_quality.get("manifest", {})
    records = manifest.get("artifacts", [])
    match = [
        record
        for record in records
        if record.get("path") == "combined_bayesian_summary.csv"
    ]
    if len(match) != 1:
        raise RuntimeError(
            "Normalization manifest does not uniquely bind "
            "combined_bayesian_summary.csv."
        )
    record = match[0]
    if summary_path.stat().st_size != int(record["bytes"]):
        raise RuntimeError("Summary size differs from the bound normalization input.")
    if _sha256(summary_path) != str(record["sha256"]):
        raise RuntimeError("Summary hash differs from the bound normalization input.")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--posterior-dir",
        type=Path,
        default=DEFAULT_POSTERIOR_DIR,
    )
    parser.add_argument("--long-ioniz", type=Path, default=DEFAULT_LONG_IONIZ)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--n-starts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--pointwise-fraction", type=float, default=0.10)
    parser.add_argument("--peak-step-nm", type=float, default=0.10)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run three replicates with one reviewed-anchor start unless "
            "--replicates or --n-starts is supplied. Outputs remain clearly "
            "marked as smoke results."
        ),
    )
    return parser


def _analysis_quality(*, quick: bool, replicates: int, n_starts: int) -> str:
    if quick:
        return "smoke"
    if replicates >= 200 and n_starts >= 3:
        return "production_sensitivity"
    return "custom_sensitivity"


def main(argv: Sequence[str] | None = None) -> int:
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    replicates = (
        int(args.replicates)
        if args.replicates is not None
        else (3 if args.quick else 200)
    )
    n_starts = (
        int(args.n_starts) if args.n_starts is not None else (1 if args.quick else 3)
    )
    if replicates < 1:
        raise ValueError("--replicates must be positive.")
    if n_starts < 1:
        raise ValueError("--n-starts must be positive.")
    analysis_quality = _analysis_quality(
        quick=bool(args.quick),
        replicates=replicates,
        n_starts=n_starts,
    )
    if not np.isfinite(args.pointwise_fraction) or args.pointwise_fraction < 0.0:
        raise ValueError("--pointwise-fraction must be finite and nonnegative.")
    if not np.isfinite(args.peak_step_nm) or args.peak_step_nm <= 0.0:
        raise ValueError("--peak-step-nm must be finite and positive.")

    summary_path = args.summary.resolve()
    posterior_dir = args.posterior_dir.resolve()
    long_ioniz_path = args.long_ioniz.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not posterior_dir.is_dir():
        raise FileNotFoundError(f"Missing posterior directory: {posterior_dir}")
    if not long_ioniz_path.is_file():
        raise FileNotFoundError(f"Missing S0 SRIM profile: {long_ioniz_path}")

    default_output = DEFAULT_SMOKE_OUTPUT_DIR if args.quick else DEFAULT_OUTPUT_DIR
    requested_output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output.resolve()
    )

    # Even smoke runs use the exact production posterior archive.  Quick mode
    # reduces only the downstream replicate count and optimization starts.
    normalization_quality = _normalization_quality(
        posterior_dir,
        require_production=True,
    )
    _require_summary_manifest_match(summary_path, normalization_quality)
    blocks = _load_posterior_blocks(summary_path, posterior_dir)
    base = load_observations(
        summary_path,
        posterior_dir=posterior_dir,
        pointwise_fraction=float(args.pointwise_fraction),
        covariance_mode="full",
    )
    if tuple(dict.fromkeys(base.dataset.astype(str))) != EXPECTED_DATASETS:
        raise RuntimeError(
            "Loaded observation order differs from the archive contract."
        )
    joint_archive_validation = _validate_joint_archive(
        base,
        blocks,
        posterior_dir,
    )

    source = IonizationProfile(*load_srim(long_ioniz_path, include_recoils=False))
    specifications = _anchored_specs()
    block_hashes = {block.path: _sha256(block.path) for block in blocks}
    selected = _draw_indices(
        blocks,
        replicates=replicates,
        seed=int(args.seed),
    )
    maximum = max(400.0, 1.5 * float(np.max(base.x)))
    peak_grid = np.arange(
        0.0,
        maximum + 0.5 * float(args.peak_step_nm),
        float(args.peak_step_nm),
    )
    staging = _prepare_output_directory(requested_output)

    draw_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_step = max(1, replicates // 20)
    for replicate_offset in range(replicates):
        replicate_id = replicate_offset + 1
        latent = _latent_vector(base, blocks, selected[replicate_offset])
        latent_id = _array_sha256(latent)
        for block, draw_index in zip(
            blocks,
            selected[replicate_offset],
            strict=True,
        ):
            draw_rows.append(
                {
                    "replicate_id": replicate_id,
                    "latent_vector_id": latent_id,
                    "dataset": block.dataset,
                    "posterior_file": str(block.path),
                    "posterior_file_sha256": block_hashes[block.path],
                    "draw_index_zero_based": int(draw_index),
                    "n_available_draws": block.n_draws,
                    "n_joint_components": block.n_rows,
                }
            )

        replicate_data = _replace_intensity(base, latent)
        for spec in specifications:
            seed = _fit_seed(int(args.seed), replicate_id, spec.key)
            try:
                result = fit_model(
                    replicate_data,
                    source,
                    spec,
                    seed=seed,
                    n_starts=n_starts,
                )
                fit_rows.append(
                    _fit_row(
                        replicate_id=replicate_id,
                        latent_id=latent_id,
                        fit_seed=seed,
                        spec=spec,
                        result=result,
                        profile=source,
                        peak_grid=peak_grid,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "replicate_id": replicate_id,
                        "latent_vector_id": latent_id,
                        "model_key": spec.key,
                        "fit_seed": seed,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        if (
            replicate_id == 1
            or replicate_id == replicates
            or replicate_id % progress_step == 0
        ):
            print(
                f"Posterior-draw replicate {replicate_id}/{replicates}",
                flush=True,
            )

    draws_frame = pd.DataFrame(draw_rows)
    raw_fits_frame = pd.DataFrame(fit_rows)
    if raw_fits_frame.empty:
        raw_fits_frame = pd.DataFrame(
            columns=[
                "replicate_id",
                "latent_vector_id",
                "model_key",
                "model_label",
                "fit_seed",
                "aicc",
                "at_lower_bound",
                "at_upper_bound",
                "peak_at_grid_boundary",
                "hessian_positive_definite_diagnostic_only",
            ]
        )
    fits_frame = _add_within_replicate_deltas(raw_fits_frame)
    summary_frame = _summarize_replicates(fits_frame)
    draws_frame.insert(0, "analysis_quality", analysis_quality)
    fits_frame.insert(0, "analysis_quality", analysis_quality)
    summary_frame.insert(0, "analysis_quality", analysis_quality)
    draws_frame.to_csv(staging / "replicate_draws.csv", index=False)
    fits_frame.to_csv(staging / "replicate_fits.csv", index=False)
    summary_frame.to_csv(staging / "summary.csv", index=False)

    planned_fits = replicates * len(specifications)
    bound_rows = fits_frame[
        fits_frame["at_lower_bound"].fillna("").ne("")
        | fits_frame["at_upper_bound"].fillna("").ne("")
    ]
    hessian_invalid = fits_frame[
        ~fits_frame["hessian_positive_definite_diagnostic_only"].astype(bool)
    ]
    complete_replicates = int(
        (fits_frame.groupby("replicate_id")["model_key"].nunique() == 3).sum()
    )
    integrity_issues = _integrity_issues(fits_frame, n_starts=n_starts)
    integrity_errors = [
        issue for issue in integrity_issues if issue["severity"] == "error"
    ]
    integrity_warnings = [
        issue for issue in integrity_issues if issue["severity"] == "warning"
    ]
    run_success = (
        not failures
        and not integrity_errors
        and len(fits_frame) == planned_fits
        and complete_replicates == replicates
    )
    if not run_success:
        outcome = "failed"
    elif integrity_warnings:
        outcome = "success_with_warnings"
    else:
        outcome = "success"

    summary_payload = {
        "schema_version": 1,
        "analysis_quality": analysis_quality,
        "method_boundary": METHOD_BOUNDARY,
        "replicates": replicates,
        "models": [spec.key for spec in specifications],
        "summaries": summary_frame.to_dict(orient="records"),
        "diagnostics": {
            "fits_at_parameter_bounds": len(bound_rows),
            "non_positive_definite_hessians_diagnostic_only": len(hessian_invalid),
            "failed_fits": len(failures),
            "integrity_issues": integrity_issues,
        },
    }
    replicate_payload = {
        "schema_version": 1,
        "analysis_quality": analysis_quality,
        "method_boundary": METHOD_BOUNDARY,
        "draw_selections": draws_frame.to_dict(orient="records"),
        "fit_point_estimates": fits_frame.to_dict(orient="records"),
        "failures": failures,
    }
    _write_json(staging / "summary.json", summary_payload)
    _write_json(staging / "replicate_results.json", replicate_payload)

    input_paths = [
        summary_path,
        long_ioniz_path,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts" / "model_sensitivity.py",
        REPOSITORY_ROOT / "src" / "se_model" / "data.py",
        REPOSITORY_ROOT / "src" / "se_model" / "deposition.py",
        REPOSITORY_ROOT / "src" / "se_model" / "fit.py",
        REPOSITORY_ROOT / "src" / "se_model" / "likelihood.py",
        REPOSITORY_ROOT / "src" / "se_model" / "models.py",
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "requirements-lock.txt",
        posterior_dir / "artifact_manifest.json",
        posterior_dir / "sampling_quality.json",
        posterior_dir / "joint_normalization_metadata.json",
        posterior_dir / "joint_normalization_posterior.npz",
        posterior_dir / "run_provenance.json",
        *(block.path for block in blocks),
    ]
    input_paths = sorted(
        {
            str(path.resolve()): path.resolve()
            for path in input_paths
            if path.is_file()
        }.values(),
        key=str,
    )
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_quality": analysis_quality,
        "run_success": run_success,
        "outcome": outcome,
        "method_boundary": METHOD_BOUNDARY,
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            *command_arguments,
        ],
        "repository_root": str(REPOSITORY_ROOT),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git(["status", "--porcelain"]),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "software_versions": _software_versions(),
        "arguments": vars(args),
        "resolved": {
            "summary": summary_path,
            "posterior_dir": posterior_dir,
            "long_ioniz": long_ioniz_path,
            "output_dir": requested_output,
            "replicates": replicates,
            "n_starts": n_starts,
            "seed": int(args.seed),
            "pointwise_fraction": float(args.pointwise_fraction),
            "peak_step_nm": float(args.peak_step_nm),
            "peak_grid_maximum_nm": float(peak_grid[-1]),
        },
        "random_selection": {
            "algorithm": (
                "numpy SeedSequence(master_seed).spawn(3), then one "
                "default_rng stream per flake with replacement"
            ),
            "master_seed": int(args.seed),
            "draw_indices_file": "replicate_draws.csv",
        },
        "reviewed_anchors": REVIEWED_ANCHORS,
        "normalization_quality": normalization_quality,
        "joint_archive_validation": joint_archive_validation,
        "base_weighting_covariance_id": covariance_identifier(base.y_cov),
        "posterior_blocks": [
            {
                "dataset": block.dataset,
                "path": block.path,
                "sha256": block_hashes[block.path],
                "n_draws": block.n_draws,
                "n_rows": block.n_rows,
                "row_indices": block.row_indices,
                "row_keys": block.row_keys,
            }
            for block in blocks
        ],
        "input_files": [
            {
                "path": path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
    }
    status = {
        "schema_version": 1,
        "analysis_quality": analysis_quality,
        "success": run_success,
        "outcome": outcome,
        "planned_replicates": replicates,
        "complete_replicates": complete_replicates,
        "planned_fits": planned_fits,
        "completed_fits": len(fits_frame),
        "failed_fits": len(failures),
        "fits_at_parameter_bounds": len(bound_rows),
        "non_positive_definite_hessians_diagnostic_only": len(hessian_invalid),
        "integrity_errors": len(integrity_errors),
        "integrity_warnings": len(integrity_warnings),
        "integrity_issues": integrity_issues,
        "normalization_quality_passed": normalization_quality["passed"],
        "failures": failures,
        "interpretation": (
            "Descriptive posterior-draw propagation of point estimates only; "
            "not a bootstrap, parameter posterior, p-value, or mechanism test."
        ),
    }
    _write_json(staging / "provenance.json", provenance)
    _write_json(staging / "status.json", status)
    _finalize_output_directory(staging, requested_output)

    print(
        f"Completed {len(fits_frame)}/{planned_fits} fits; outputs: {requested_output}",
        flush=True,
    )
    if not run_success:
        raise SystemExit(
            "Posterior-draw sensitivity was incomplete; inspect status.json."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

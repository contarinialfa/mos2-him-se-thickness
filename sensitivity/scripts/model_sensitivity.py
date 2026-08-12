"""Run the paper-model fit and prespecified statistical sensitivity campaign.

This script compares descriptive fit quality only.  It deliberately makes no
claim that the best statistical curve identifies a microscopic mechanism.

The primary likelihood is the parameter-dependent full-covariance Gaussian
EIV likelihood implemented in :mod:`se_model.likelihood`.  Every comparison
uses that same objective, including its covariance log-determinant.
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
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from se_model.data import (  # noqa: E402
    ObservationData,
    load_observations,
    load_srim,
)
from se_model.deposition import IonizationProfile  # noqa: E402
from se_model.finite import (  # noqa: E402
    finite_source_metadata,
    load_finite_source,
)
from se_model.fit import FitResult, fit_model, profile_parameter  # noqa: E402
from se_model.interpolation import resample_profile  # noqa: E402
from se_model.models import (  # noqa: E402
    ModelSpec,
    paper_model_specs,
    predict_model,
)

DEFAULT_SUMMARY = (
    REPOSITORY_ROOT
    / "data"
    / "reference_posteriors"
    / "transmission"
    / "combined_bayesian_summary.csv"
)
DEFAULT_POSTERIOR_DIR = (
    REPOSITORY_ROOT / "data" / "reference_posteriors" / "transmission"
)
DEFAULT_LONG_IONIZ = (
    REPOSITORY_ROOT / "data" / "SRIM 30 keV - Monolayer Long" / "IONIZ.txt"
)
DEFAULT_SWEEP_DIR = REPOSITORY_ROOT / "data" / "SRIM-Thickness-Series"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "model_sensitivity"
DEFAULT_SMOKE_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs" / "model_sensitivity_smoke"
)

SOURCE_LABELS = {
    "S0": "Long target, ions",
    "S1": "Finite thickness, ions",
    "S2": "Long target, ions + recoils",
    "S3": "Finite thickness, ions + recoils",
    "I1": "Long target, PCHIP interpolation, zero tail",
    "I2": "Legacy cubic interpolation and endpoint clamp",
}
MODEL_STYLES = {
    "symmetric_fixed": {
        "color": "#686868",
        "linestyle": "--",
        "marker": "s",
    },
    "symmetric_free": {
        "color": "#cc0808",
        "linestyle": "-.",
        "marker": "^",
    },
    "asymmetric": {
        "color": "#2e0acc",
        "linestyle": "-",
        "marker": "o",
    },
}


@dataclass(frozen=True)
class CampaignCase:
    """One unique fit in the prespecified sensitivity campaign."""

    source_key: str
    model_key: str
    covariance_mode: str
    eels_scale: float
    pointwise_fraction: float
    lambda_front_nm: float
    fixed_scale: float | None = None
    excluded_dataset: str | None = None
    analysis_tags: tuple[str, ...] = ()

    @property
    def comparison_key(self) -> str:
        exclusion = self.excluded_dataset or "none"
        fixed_scale = (
            "free" if self.fixed_scale is None else f"{self.fixed_scale:.6g}"
        )
        return (
            f"{self.source_key}|cov={self.covariance_mode}|"
            f"g={self.eels_scale:.6g}|point={self.pointwise_fraction:.6g}|"
            f"lambda_front={self.lambda_front_nm:.6g}|"
            f"fixed_scale={fixed_scale}|exclude={exclusion}"
        )

    @property
    def case_id(self) -> str:
        exclusion = self.excluded_dataset or "none"
        fixed_scale = (
            "free" if self.fixed_scale is None else f"{self.fixed_scale:.3f}"
        )
        return (
            f"{self.source_key}__{self.model_key}__{self.covariance_mode}"
            f"__g{self.eels_scale:.3f}__p{self.pointwise_fraction:.3f}"
            f"__lf{self.lambda_front_nm:.3f}__s-{fixed_scale}"
            f"__exclude-{exclusion}"
        ).replace(" ", "_")

    @property
    def is_baseline(self) -> bool:
        return (
            self.source_key == "S0"
            and self.covariance_mode == "full"
            and self.eels_scale == 1.0
            and self.pointwise_fraction == 0.10
            and self.lambda_front_nm == 2.0
            and self.fixed_scale is None
            and self.excluded_dataset is None
        )


@dataclass(frozen=True)
class CompletedFit:
    """Fit result and the exact transformed observations used for it."""

    case: CampaignCase
    spec: ModelSpec
    data: ObservationData
    result: FitResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
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
    packages = ("numpy", "scipy", "pandas", "matplotlib")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _normalization_quality(
    posterior_dir: Path,
    *,
    require_production: bool,
) -> dict[str, Any]:
    """Read and, for production, enforce the upstream sampling contract."""

    quality_path = posterior_dir / "sampling_quality.json"
    metadata_path = posterior_dir / "joint_normalization_metadata.json"
    manifest_path = posterior_dir / "artifact_manifest.json"
    joint_path = posterior_dir / "joint_normalization_posterior.npz"
    required_paths = (quality_path, metadata_path, manifest_path, joint_path)
    if not all(path.is_file() for path in required_paths):
        if require_production:
            raise FileNotFoundError(
                "Production mode requires sampling_quality.json, "
                "joint_normalization_metadata.json, artifact_manifest.json, "
                "and joint_normalization_posterior.npz beside the posterior "
                "draws."
            )
        return {
            "available": False,
            "passed": False,
            "failures": [
                "Sampling-quality metadata or its artifact manifest is unavailable."
            ],
            "quality_path": str(quality_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "joint_path": str(joint_path),
        }

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    datasets = quality.get("datasets", [])
    if len(datasets) != 3:
        failures.append(f"Expected 3 dataset diagnostics, found {len(datasets)}.")
    expected_labels = {"Dataset 1", "Dataset 2", "Dataset 3"}
    quality_labels = {str(record.get("dataset")) for record in datasets}
    if quality_labels != expected_labels:
        failures.append(
            "Sampling diagnostics have unexpected dataset labels: "
            f"{sorted(quality_labels)}."
        )
    for record in datasets:
        label = record.get("dataset", "unknown dataset")
        if record.get("divergences") != 0:
            failures.append(f"{label}: divergences={record.get('divergences')}.")
        maximum_r_hat = record.get("maximum_r_hat")
        if maximum_r_hat is None or float(maximum_r_hat) >= 1.01:
            failures.append(f"{label}: maximum R-hat={maximum_r_hat}.")
        minimum_bulk = record.get("minimum_ess_bulk")
        if minimum_bulk is None or float(minimum_bulk) < 1000:
            failures.append(f"{label}: minimum bulk ESS={minimum_bulk}.")
        minimum_tail = record.get("minimum_ess_tail")
        if minimum_tail is None or float(minimum_tail) < 1000:
            failures.append(f"{label}: minimum tail ESS={minimum_tail}.")

    sampling = metadata.get("sampling", {})
    if int(sampling.get("chains", 0)) < 4:
        failures.append(f"Sampling used only {sampling.get('chains')} chains.")
    if int(sampling.get("draws", 0)) < 2000:
        failures.append(f"Sampling used only {sampling.get('draws')} draws/chain.")
    if float(sampling.get("target_accept", 0.0)) < 0.95:
        failures.append(
            f"Sampling target_accept={sampling.get('target_accept')} is below 0.95."
        )

    manifest_records = manifest.get("artifacts", [])
    if not isinstance(manifest_records, list) or not manifest_records:
        failures.append("The normalization artifact manifest is empty or malformed.")
        manifest_records = []
    listed_names: set[str] = set()
    for record in manifest_records:
        try:
            relative = Path(str(record["path"]))
            expected_bytes = int(record["bytes"])
            expected_sha256 = str(record["sha256"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"Malformed artifact-manifest record: {record!r}.")
            continue
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            failures.append(f"Unsafe artifact-manifest path: {relative}.")
            continue
        artifact_path = posterior_dir / relative
        listed_names.add(relative.name)
        if not artifact_path.is_file():
            failures.append(f"Manifest artifact is missing: {relative.name}.")
            continue
        if artifact_path.stat().st_size != expected_bytes:
            failures.append(f"Manifest size mismatch: {relative.name}.")
        if _sha256(artifact_path) != expected_sha256:
            failures.append(f"Manifest SHA-256 mismatch: {relative.name}.")

    ratio_paths = sorted(posterior_dir.glob("*_posterior_ratio_samples.npz"))
    required_names = {
        quality_path.name,
        metadata_path.name,
        joint_path.name,
        "combined_bayesian_summary.csv",
        *(path.name for path in ratio_paths),
    }
    missing_manifest_names = required_names.difference(listed_names)
    if missing_manifest_names:
        failures.append(
            "Required posterior artifacts are absent from the manifest: "
            f"{sorted(missing_manifest_names)}."
        )

    expected_counts = {"Dataset 1": 6, "Dataset 2": 5, "Dataset 3": 4}
    expected_draws = int(sampling.get("chains", 0)) * int(
        sampling.get("draws", 0)
    )
    try:
        with np.load(joint_path, allow_pickle=False) as joint:
            joint_labels = np.asarray(joint["dataset"]).astype(str)
            posterior_mean = np.asarray(joint["posterior_mean"], dtype=float)
            covariance = np.asarray(joint["covariance"], dtype=float)
        if posterior_mean.shape != (15,) or covariance.shape != (15, 15):
            failures.append(
                "Joint posterior dimensions are not 15 means and a 15x15 covariance."
            )
        joint_label_set = set(joint_labels.tolist())
        if joint_label_set != expected_labels:
            failures.append(
                f"Joint posterior has unexpected labels: {sorted(joint_label_set)}."
            )
        for label, expected_count in expected_counts.items():
            actual_count = int(np.count_nonzero(joint_labels == label))
            if actual_count != expected_count:
                failures.append(
                    f"{label}: joint posterior has {actual_count} rows; "
                    f"expected {expected_count}."
                )
    except (KeyError, OSError, ValueError) as error:
        failures.append(
            "Could not validate joint-normalization dimensions: "
            f"{type(error).__name__}: {error}"
        )

    ratio_labels: set[str] = set()
    ratio_columns = 0
    if len(ratio_paths) != 3:
        failures.append(
            f"Expected 3 posterior-ratio NPZ files, found {len(ratio_paths)}."
        )
    for ratio_path in ratio_paths:
        try:
            with np.load(ratio_path, allow_pickle=False) as posterior:
                ratios = np.asarray(posterior["ratio_samples"], dtype=float)
                labels = np.asarray(posterior["dataset_label"]).astype(str).reshape(-1)
                group_ids = np.asarray(posterior["group_ids"]).reshape(-1)
                thicknesses = np.asarray(posterior["thickness_nm"]).reshape(-1)
            if labels.size != 1:
                failures.append(f"{ratio_path.name}: dataset label is not scalar.")
                continue
            label = str(labels[0])
            ratio_labels.add(label)
            if ratios.ndim != 2:
                failures.append(f"{ratio_path.name}: ratio_samples is not 2D.")
                continue
            ratio_columns += ratios.shape[1]
            if ratios.shape[1] != group_ids.size or ratios.shape[1] != thicknesses.size:
                failures.append(f"{ratio_path.name}: posterior columns do not align.")
            if expected_draws > 0 and ratios.shape[0] != expected_draws:
                failures.append(
                    f"{ratio_path.name}: {ratios.shape[0]} draws; "
                    f"expected {expected_draws}."
                )
            if label in expected_counts and ratios.shape[1] != expected_counts[label]:
                failures.append(
                    f"{ratio_path.name}: {ratios.shape[1]} groups for {label}; "
                    f"expected {expected_counts[label]}."
                )
        except (KeyError, OSError, ValueError) as error:
            failures.append(
                f"Could not validate {ratio_path.name}: "
                f"{type(error).__name__}: {error}"
            )
    if ratio_labels != expected_labels:
        failures.append(
            f"Posterior-ratio files have unexpected labels: {sorted(ratio_labels)}."
        )
    if ratio_columns != 15:
        failures.append(
            f"Posterior-ratio files contain {ratio_columns} columns, not 15."
        )
    if int(metadata.get("rows", -1)) != 15:
        failures.append(f"Normalization metadata rows={metadata.get('rows')}, not 15.")

    payload = {
        "available": True,
        "passed": not failures,
        "failures": failures,
        "quality_path": str(quality_path),
        "quality_sha256": _sha256(quality_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "joint_path": str(joint_path),
        "joint_sha256": _sha256(joint_path),
        "quality": quality,
        "metadata": metadata,
        "manifest": manifest,
    }
    if require_production and failures:
        raise RuntimeError(
            "Upstream normalization is not production quality:\n- "
            + "\n- ".join(failures)
        )
    return payload


def _prepare_output_directory(requested: Path) -> Path:
    """Create a fresh sibling staging directory for one immutable run."""

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
    """Atomically publish a completed or explicitly failed run directory."""

    if requested.exists():
        raise FileExistsError(f"Cannot publish over existing output: {requested}")
    staging.replace(requested)


def _require_full_covariance(data: ObservationData) -> None:
    """Verify that every flake carries a non-diagonal posterior block."""

    labels = tuple(dict.fromkeys(data.dataset.astype(str)))
    for label in labels:
        indices = np.flatnonzero(data.dataset.astype(str) == label)
        block = data.y_cov[np.ix_(indices, indices)]
        off_diagonal = block - np.diag(np.diag(block))
        if not np.any(np.abs(off_diagonal) > 0.0):
            raise ValueError(
                f"{label} has no off-diagonal posterior covariance. "
                "Supply a posterior directory containing an exact NPZ match "
                "for every flake."
            )

    for first_index, first_label in enumerate(labels):
        first = np.flatnonzero(data.dataset.astype(str) == first_label)
        for second_label in labels[first_index + 1 :]:
            second = np.flatnonzero(data.dataset.astype(str) == second_label)
            cross = data.y_cov[np.ix_(first, second)]
            if np.any(cross != 0.0):
                raise ValueError(
                    "Cross-flake posterior covariance is nonzero, but the "
                    "normalization models were fitted independently."
                )


def _transformed_data(
    base: ObservationData,
    *,
    covariance_mode: str,
    eels_scale: float,
    pointwise_fraction: float,
    excluded_dataset: str | None,
) -> ObservationData:
    if covariance_mode == "full":
        covariance = base.y_cov.copy()
    elif covariance_mode == "diagonal":
        covariance = np.diag(np.diag(base.y_cov))
    else:
        raise ValueError(f"Unknown covariance mode {covariance_mode!r}.")

    scaled_x = base.x * eels_scale
    transformed = ObservationData(
        x=scaled_x,
        y=base.y.copy(),
        y_cov=covariance,
        x_sigma=pointwise_fraction * scaled_x,
        dataset=base.dataset.copy(),
        group_id=base.group_id.copy(),
        row_id=base.row_id.copy(),
    )
    if excluded_dataset is None:
        return transformed
    mask = transformed.dataset.astype(str) != excluded_dataset
    return transformed.subset(mask)


def _add_case(
    cases: OrderedDict[tuple[Any, ...], CampaignCase],
    case: CampaignCase,
) -> None:
    key = (
        case.source_key,
        case.model_key,
        case.covariance_mode,
        case.eels_scale,
        case.pointwise_fraction,
        case.lambda_front_nm,
        case.fixed_scale,
        case.excluded_dataset,
    )
    if key not in cases:
        cases[key] = case
        return
    existing = cases[key]
    tags = tuple(dict.fromkeys((*existing.analysis_tags, *case.analysis_tags)))
    cases[key] = replace(existing, analysis_tags=tags)


def build_campaign(dataset_labels: tuple[str, ...]) -> tuple[CampaignCase, ...]:
    """Build and deduplicate the complete prespecified campaign."""

    cases: OrderedDict[tuple[Any, ...], CampaignCase] = OrderedDict()
    model_keys = tuple(spec.key for spec in paper_model_specs())

    for source_key in SOURCE_LABELS:
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    source_key,
                    model_key,
                    "full",
                    1.0,
                    0.10,
                    2.0,
                    analysis_tags=("source_profile",),
                ),
            )

    for lambda_front in (1.5, 2.0, 2.5):
        _add_case(
            cases,
            CampaignCase(
                "S0",
                "asymmetric",
                "full",
                1.0,
                0.10,
                lambda_front,
                analysis_tags=("lambda_front",),
            ),
        )

    for eels_scale in (0.9, 1.0, 1.1):
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    "S0",
                    model_key,
                    "full",
                    eels_scale,
                    0.10,
                    2.0,
                    analysis_tags=("coherent_eels_scale",),
                ),
            )

    for pointwise_fraction in (0.0, 0.05, 0.10, 0.15):
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    "S0",
                    model_key,
                    "full",
                    1.0,
                    pointwise_fraction,
                    2.0,
                    analysis_tags=("pointwise_thickness",),
                ),
            )

    for covariance_mode in ("full", "diagonal"):
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    "S0",
                    model_key,
                    covariance_mode,
                    1.0,
                    0.10,
                    2.0,
                    analysis_tags=("normalization_covariance",),
                ),
            )

    for fixed_scale in (0.9, 1.0, 1.1):
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    "S0",
                    model_key,
                    "full",
                    1.0,
                    0.10,
                    2.0,
                    fixed_scale=fixed_scale,
                    analysis_tags=("fixed_absolute_source_axis_stress",),
                ),
            )

    for excluded_dataset in dataset_labels:
        for model_key in model_keys:
            _add_case(
                cases,
                CampaignCase(
                    "S0",
                    model_key,
                    "full",
                    1.0,
                    0.10,
                    2.0,
                    excluded_dataset=excluded_dataset,
                    analysis_tags=("leave_one_flake_out",),
                ),
            )
    return tuple(cases.values())


def _case_anchor_starts(
    case: CampaignCase,
    spec: ModelSpec,
    completed: list[CompletedFit],
) -> list[OrderedDict[str, float]]:
    """Build deterministic transformed-baseline and nested-model anchors."""

    anchors: list[OrderedDict[str, float]] = []
    baseline = next(
        (
            item
            for item in completed
            if item.case.is_baseline and item.case.model_key == case.model_key
        ),
        None,
    )
    if baseline is not None:
        values = OrderedDict(baseline.result.params)
        values["scale"] *= case.eels_scale
        if case.model_key == "symmetric_free":
            values["lambda_shared_nm"] *= case.eels_scale
        elif case.model_key == "asymmetric":
            values["lambda_back_nm"] *= case.eels_scale
        anchors.append(
            OrderedDict((name, float(values[name])) for name in spec.parameter_names)
        )

    if case.model_key in {"symmetric_free", "asymmetric"}:
        nested = next(
            (
                item
                for item in completed
                if item.case.comparison_key == case.comparison_key
                and item.case.model_key == "symmetric_fixed"
            ),
            None,
        )
        if nested is not None:
            values = OrderedDict(nested.result.params)
            length_name = (
                "lambda_shared_nm"
                if case.model_key == "symmetric_free"
                else "lambda_back_nm"
            )
            values[length_name] = spec.lambda_front_nm
            anchors.append(
                OrderedDict(
                    (name, float(values[name])) for name in spec.parameter_names
                )
            )
    return anchors


def _fit_summary_row(completed: CompletedFit) -> dict[str, Any]:
    case, result = completed.case, completed.result
    uncertainty_valid = bool(
        result.hessian_positive_definite
        and result.hessian_rank == result.n_free_parameters
        and np.all(np.isfinite(result.covariance))
    )
    standard_errors = np.full(len(result.parameter_names), np.nan)
    if uncertainty_valid:
        diagonal = np.diag(result.covariance)
        free_indices = [
            result.parameter_names.index(name)
            for name in result.free_parameter_names
        ]
        if all(diagonal[index] > 0.0 for index in free_indices):
            standard_errors[free_indices] = np.sqrt(diagonal[free_indices])
            for name in result.fixed_params:
                standard_errors[result.parameter_names.index(name)] = 0.0
        else:
            uncertainty_valid = False
            standard_errors[:] = np.nan

    if uncertainty_valid:
        off_diagonal = result.correlation - np.eye(len(result.parameter_names))
        finite_correlations = np.abs(off_diagonal[np.isfinite(off_diagonal)])
    else:
        finite_correlations = np.asarray([], dtype=float)
    nominal_dof = result.n_observations - result.n_free_parameters
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "comparison_key": case.comparison_key,
        "analysis_tags": ";".join(case.analysis_tags),
        "baseline": case.is_baseline,
        "source_key": case.source_key,
        "source_label": SOURCE_LABELS[case.source_key],
        "model_key": case.model_key,
        "model_label": completed.spec.label,
        "covariance_mode": case.covariance_mode,
        "eels_scale": case.eels_scale,
        "pointwise_fraction": case.pointwise_fraction,
        "lambda_front_nm": case.lambda_front_nm,
        "fixed_scale": case.fixed_scale,
        "excluded_dataset": case.excluded_dataset,
        "n_observations": result.n_observations,
        "n_parameters": result.n_free_parameters,
        "nominal_dof": nominal_dof,
        "deviance": result.deviance,
        "log_likelihood": result.log_likelihood,
        "mahalanobis": result.likelihood.mahalanobis,
        "mahalanobis_per_nominal_dof": (
            result.likelihood.mahalanobis / nominal_dof if nominal_dof > 0 else np.nan
        ),
        "log_determinant": result.likelihood.log_determinant,
        "aic": result.aic,
        "aicc": result.aicc,
        "bic": result.bic,
        "covariance_id": result.likelihood.covariance_id,
        "base_covariance_id": result.likelihood.base_covariance_id,
        "likelihood_covariance_mode": result.likelihood.covariance_mode,
        "likelihood_covariance_condition": result.likelihood.condition_number,
        "cholesky_jitter": result.likelihood.cholesky_jitter,
        "max_abs_whitened_residual": float(
            np.max(np.abs(result.likelihood.whitened_residual))
        ),
        "rms_whitened_residual": float(
            np.sqrt(np.mean(result.likelihood.whitened_residual**2))
        ),
        "hessian_rank": result.hessian_rank,
        "hessian_condition": result.hessian_condition_number,
        "hessian_positive_definite": result.hessian_positive_definite,
        "uncertainty_valid": uncertainty_valid,
        "uncertainty_invalid_reason": (
            ""
            if uncertainty_valid
            else "Hessian is rank-deficient/non-positive-definite or covariance "
            "has invalid free-parameter variances."
        ),
        "maximum_absolute_parameter_correlation": (
            float(np.max(finite_correlations)) if finite_correlations.size else np.nan
        ),
        "at_lower_bound": ";".join(result.at_lower_bound),
        "at_upper_bound": ";".join(result.at_upper_bound),
        "accepted_starts": sum(record.accepted for record in result.starts),
        "attempted_starts": result.n_starts,
        "near_best_starts": sum(
            record.accepted
            and record.objective is not None
            and record.objective
            <= result.deviance + 1e-5 * max(1.0, abs(result.deviance))
            for record in result.starts
        ),
    }
    for name, value, error in zip(
        result.parameter_names,
        result.param_vector,
        standard_errors,
        strict=True,
    ):
        row[f"parameter_{name}"] = value
        row[f"standard_error_{name}"] = error
    return row


def _parameter_rows(completed: CompletedFit) -> list[dict[str, Any]]:
    result = completed.result
    uncertainty_valid = bool(
        result.hessian_positive_definite
        and result.hessian_rank == result.n_free_parameters
        and np.all(np.isfinite(result.covariance))
    )
    errors = np.full(len(result.parameter_names), np.nan)
    if uncertainty_valid:
        diagonal = np.diag(result.covariance)
        free_indices = [
            result.parameter_names.index(name)
            for name in result.free_parameter_names
        ]
        if all(diagonal[index] > 0.0 for index in free_indices):
            errors[free_indices] = np.sqrt(diagonal[free_indices])
            for name in result.fixed_params:
                errors[result.parameter_names.index(name)] = 0.0
        else:
            uncertainty_valid = False
            errors[:] = np.nan
    return [
        {
            "case_id": completed.case.case_id,
            "model_key": completed.case.model_key,
            "parameter": name,
            "value": value,
            "local_standard_error": error,
            "uncertainty_valid": uncertainty_valid,
            "fixed": name in result.fixed_params,
            "at_lower_bound": name in result.at_lower_bound,
            "at_upper_bound": name in result.at_upper_bound,
        }
        for name, value, error in zip(
            result.parameter_names,
            result.param_vector,
            errors,
            strict=True,
        )
    ]


def _observation_rows(completed: CompletedFit) -> list[dict[str, Any]]:
    data = completed.data
    evaluation = completed.result.likelihood
    marginal_sigma = np.sqrt(np.diag(evaluation.covariance))
    standardized = evaluation.residual / marginal_sigma
    rows = []
    for index in range(len(data.x)):
        rows.append(
            {
                "case_id": completed.case.case_id,
                "row_id": str(data.row_id[index]),
                "dataset": str(data.dataset[index]),
                "group_id": int(data.group_id[index]),
                "thickness_nm": float(data.x[index]),
                "thickness_sigma_nm": float(data.x_sigma[index]),
                "observed": float(data.y[index]),
                "predicted": float(evaluation.prediction[index]),
                "residual": float(evaluation.residual[index]),
                "marginal_sigma": float(marginal_sigma[index]),
                "marginal_standardized_residual": float(standardized[index]),
                "whitened_residual": float(evaluation.whitened_residual[index]),
            }
        )
    return rows


def _comparison_deltas(
    frame: pd.DataFrame,
    invalid_case_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Add contrasts only for diagnostically valid, complete comparisons."""

    frame = frame.copy()
    invalid = invalid_case_ids or set()
    frame["comparison_complete"] = False
    frame["delta_aicc_within_comparison"] = np.nan
    frame["delta_deviance_within_comparison"] = np.nan
    for _, index in frame.groupby("comparison_key").groups.items():
        subset = frame.loc[index]
        if any(case_id in invalid for case_id in subset["case_id"]):
            continue
        if set(subset["model_key"]) != {
            "symmetric_fixed",
            "symmetric_free",
            "asymmetric",
        }:
            continue
        frame.loc[index, "comparison_complete"] = True
        frame.loc[index, "delta_aicc_within_comparison"] = (
            subset["aicc"] - subset["aicc"].min()
        )
        frame.loc[index, "delta_deviance_within_comparison"] = (
            subset["deviance"] - subset["deviance"].min()
        )

    baseline = frame[
        (frame["baseline"])
        & (frame["source_key"] == "S0")
        & (frame["excluded_dataset"].isna())
        & (~frame["case_id"].isin(invalid))
    ].set_index("model_key")
    frame["delta_aicc_vs_same_model_baseline"] = np.nan
    frame["delta_deviance_vs_same_model_baseline"] = np.nan
    for index, row in frame.iterrows():
        if row["case_id"] in invalid:
            continue
        if row["model_key"] not in baseline.index:
            continue
        reference = baseline.loc[row["model_key"]]
        frame.at[index, "delta_aicc_vs_same_model_baseline"] = (
            row["aicc"] - reference["aicc"]
        )
        frame.at[index, "delta_deviance_vs_same_model_baseline"] = (
            row["deviance"] - reference["deviance"]
        )
    return frame


def _integrity_issues(
    completed: list[CompletedFit],
    campaign: tuple[CampaignCase, ...],
    *,
    production: bool,
) -> list[dict[str, Any]]:
    """Return structural, optimizer, and uncertainty-integrity violations."""

    issues: list[dict[str, Any]] = []
    by_id = {item.case.case_id: item for item in completed}
    planned_by_comparison: dict[str, list[CampaignCase]] = {}
    for case in campaign:
        planned_by_comparison.setdefault(case.comparison_key, []).append(case)

    for comparison_key, planned in planned_by_comparison.items():
        planned_models = {case.model_key for case in planned}
        if planned_models != {
            "symmetric_fixed",
            "symmetric_free",
            "asymmetric",
        }:
            continue
        available = {
            case.model_key: by_id[case.case_id]
            for case in planned
            if case.case_id in by_id
        }
        if set(available) != planned_models:
            issues.append(
                {
                    "severity": "error",
                    "code": "incomplete_model_comparison",
                    "comparison_key": comparison_key,
                    "case_ids": [case.case_id for case in planned],
                    "message": "A planned three-model comparison is incomplete.",
                }
            )
            continue
        fixed_deviance = available["symmetric_fixed"].result.deviance
        for extension in ("symmetric_free", "asymmetric"):
            extension_deviance = available[extension].result.deviance
            tolerance = 1e-5 * max(1.0, abs(fixed_deviance))
            if extension_deviance > fixed_deviance + tolerance:
                issues.append(
                    {
                        "severity": "error",
                        "code": "nesting_violation",
                        "comparison_key": comparison_key,
                        "case_ids": [
                            available["symmetric_fixed"].case.case_id,
                            available[extension].case.case_id,
                        ],
                        "message": (
                            f"{extension} deviance {extension_deviance:.9g} exceeds "
                            f"its nested fixed model {fixed_deviance:.9g}."
                        ),
                    }
                )

    for item in completed:
        result = item.result
        case_issues = []
        if result.at_any_bound:
            case_issues.append(f"parameters at bounds: {result.at_any_bound}")
        if not result.hessian_positive_definite:
            case_issues.append("Hessian is not positive definite")
        if result.hessian_rank != result.n_free_parameters:
            case_issues.append(
                f"Hessian rank {result.hessian_rank} != "
                f"{result.n_free_parameters} free parameters"
            )
        if case_issues:
            diagnostic_stress_case = bool(
                item.case.fixed_scale is not None
                or item.case.pointwise_fraction == 0.0
                or item.case.excluded_dataset is not None
            )
            issues.append(
                {
                    "severity": (
                        "error"
                        if production and not diagnostic_stress_case
                        else "warning"
                    ),
                    "code": "fit_diagnostics_invalid",
                    "case_ids": [item.case.case_id],
                    "message": "; ".join(case_issues) + ".",
                }
            )

    # Free-symmetric fits are exactly equivariant under a coherent thickness
    # rescaling: scale and lambda both multiply by g, while deviance, C_o, and
    # C_b remain unchanged. A violation is an optimizer/numerical failure.
    coherent = [
        item
        for item in completed
        if item.case.source_key == "S0"
        and item.case.model_key == "symmetric_free"
        and item.case.covariance_mode == "full"
        and item.case.pointwise_fraction == 0.10
        and item.case.lambda_front_nm == 2.0
        and item.case.fixed_scale is None
        and item.case.excluded_dataset is None
        and "coherent_eels_scale" in item.case.analysis_tags
    ]
    reference = next(
        (item for item in coherent if np.isclose(item.case.eels_scale, 1.0)),
        None,
    )
    if reference is not None:
        for item in coherent:
            g = item.case.eels_scale
            checks = {
                "deviance": (
                    item.result.deviance,
                    reference.result.deviance,
                ),
                "C_o": (
                    item.result.params["C_o"],
                    reference.result.params["C_o"],
                ),
                "C_b": (
                    item.result.params["C_b"],
                    reference.result.params["C_b"],
                ),
                "scale": (
                    item.result.params["scale"],
                    reference.result.params["scale"] * g,
                ),
                "lambda_shared_nm": (
                    item.result.params["lambda_shared_nm"],
                    reference.result.params["lambda_shared_nm"] * g,
                ),
            }
            failed = [
                name
                for name, (actual, expected) in checks.items()
                if not np.isclose(actual, expected, rtol=2e-4, atol=2e-6)
            ]
            if failed:
                issues.append(
                    {
                        "severity": "error",
                        "code": "coherent_scale_invariance_violation",
                        "case_ids": [
                            reference.case.case_id,
                            item.case.case_id,
                        ],
                        "message": (
                            "Free-symmetric coherent-scale invariance failed "
                            f"for {failed} at g={g:g}."
                        ),
                    }
                )
    return issues


def _derived_fit_row(completed: CompletedFit, source: Any) -> dict[str, Any]:
    """Compute convention-explicit component and peak summaries."""

    result = completed.result
    parameters = result.params
    if completed.spec.escape_mode == "free_symmetric":
        lambda_front = parameters["lambda_shared_nm"]
        lambda_back = parameters["lambda_shared_nm"]
    elif completed.spec.escape_mode == "fixed_symmetric":
        lambda_front = completed.spec.lambda_front_nm
        lambda_back = completed.spec.lambda_front_nm
    else:
        lambda_front = completed.spec.lambda_front_nm
        lambda_back = parameters["lambda_back_nm"]

    maximum = max(250.0, 1.15 * float(np.max(completed.data.x)))
    grid = np.linspace(0.0, maximum, 2501)
    prediction = predict_model(
        completed.spec,
        source,
        grid,
        parameters,
        return_components=True,
    )
    peak_index = int(np.argmax(prediction.total))
    peak_total = float(prediction.total[peak_index])
    peak_back = float(prediction.back[peak_index])
    area_equivalent = float(parameters["C_b"] * lambda_back / lambda_front)
    area_standard_error = np.nan
    result_covariance = np.asarray(result.covariance, dtype=float)
    if (
        result.hessian_positive_definite
        and result.hessian_rank == result.n_free_parameters
        and np.all(np.isfinite(result_covariance))
    ):
        gradient = np.zeros(len(result.parameter_names), dtype=float)
        c_b_index = result.parameter_names.index("C_b")
        gradient[c_b_index] = lambda_back / lambda_front
        if completed.spec.escape_mode == "asymmetric":
            lambda_index = result.parameter_names.index("lambda_back_nm")
            gradient[lambda_index] = parameters["C_b"] / lambda_front
        variance = float(gradient @ result_covariance @ gradient)
        if variance >= 0.0 and np.isfinite(variance):
            area_standard_error = float(np.sqrt(variance))
    return {
        "case_id": completed.case.case_id,
        "lambda_front_measured_nm": float(lambda_front),
        "lambda_back_measured_nm": float(lambda_back),
        "lambda_back_raw_srim_nm": float(lambda_back / parameters["scale"]),
        "C_b_kernel_amplitude": float(parameters["C_b"]),
        "C_b_unit_area_equivalent": area_equivalent,
        "C_b_unit_area_local_standard_error": area_standard_error,
        "peak_thickness_nm": float(grid[peak_index]),
        "peak_intensity": peak_total,
        "model_assigned_back_fraction_at_peak": (
            peak_back / peak_total if peak_total > 0.0 else np.nan
        ),
    }


def _prediction_rows(
    completed: CompletedFit,
    source: Any,
    *,
    grid_points: int = 301,
) -> list[dict[str, Any]]:
    maximum = 1.05 * float(np.max(completed.data.x))
    grid = np.linspace(0.0, maximum, grid_points)
    prediction = predict_model(
        completed.spec,
        source,
        grid,
        completed.result.params,
        return_components=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        back_fraction = np.divide(
            prediction.back,
            prediction.total,
            out=np.full_like(prediction.total, np.nan),
            where=prediction.total > 0.0,
        )
    return [
        {
            "case_id": completed.case.case_id,
            "thickness_nm": float(thickness),
            "predicted_total": float(total),
            "predicted_front_component": float(front),
            "predicted_back_component": float(back),
            "model_assigned_back_fraction": float(fraction),
        }
        for thickness, total, front, back, fraction in zip(
            grid,
            prediction.total,
            prediction.front,
            prediction.back,
            back_fraction,
            strict=True,
        )
    ]


def _finite_grid_convergence_rows(
    completed: list[CompletedFit],
    production_sources: dict[str, Any],
    half_step_sources: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare finite-source predictions at the fitted parameters.

    This is a numerical quadrature check, not a second fit.  The comparison
    uses both the observation locations and a dense curve grid; differences
    are normalized to the peak prediction rather than to near-zero values.
    """

    rows: list[dict[str, Any]] = []
    for item in completed:
        if (
            item.case.source_key not in {"S1", "S3"}
            or item.case.covariance_mode != "full"
            or item.case.eels_scale != 1.0
            or item.case.pointwise_fraction != 0.10
            or item.case.lambda_front_nm != 2.0
            or item.case.fixed_scale is not None
            or item.case.excluded_dataset is not None
        ):
            continue
        dense = np.linspace(0.0, 250.0, 251)
        evaluation_grid = np.unique(np.concatenate((item.data.x, dense)))
        production = np.asarray(
            predict_model(
                item.spec,
                production_sources[item.case.source_key],
                evaluation_grid,
                item.result.params,
            ),
            dtype=float,
        )
        half_step = np.asarray(
            predict_model(
                item.spec,
                half_step_sources[item.case.source_key],
                evaluation_grid,
                item.result.params,
            ),
            dtype=float,
        )
        difference = production - half_step
        peak = max(float(np.max(np.abs(half_step))), np.finfo(float).tiny)
        maximum_absolute = float(np.max(np.abs(difference)))
        rows.append(
            {
                "case_id": item.case.case_id,
                "source_key": item.case.source_key,
                "model_key": item.case.model_key,
                "production_step_nm": production_sources[
                    item.case.source_key
                ].integration_step_nm,
                "reference_step_nm": half_step_sources[
                    item.case.source_key
                ].integration_step_nm,
                "evaluation_points": evaluation_grid.size,
                "maximum_absolute_prediction_difference": maximum_absolute,
                "rms_prediction_difference": float(
                    np.sqrt(np.mean(np.square(difference)))
                ),
                "maximum_difference_relative_to_peak": maximum_absolute / peak,
                "passed": maximum_absolute / peak < 5e-5,
            }
        )
    return rows


def _profile_crossing_interval(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> tuple[float | None, float | None]:
    """Linearly interpolate the profile interval surrounding its minimum."""

    valid = sorted(
        (
            (float(row["value"]), float(row["delta_deviance"]))
            for row in rows
            if row["success"] and row["delta_deviance"] is not None
        ),
        key=lambda item: item[0],
    )
    if not valid:
        return None, None
    values = np.asarray([item[0] for item in valid], dtype=float)
    deltas = np.asarray([item[1] for item in valid], dtype=float)
    minimum_index = int(np.argmin(deltas))

    def crossing(indices: range) -> float | None:
        previous_index = minimum_index
        for index in indices:
            if deltas[index] >= threshold and deltas[previous_index] < threshold:
                x0, x1 = values[previous_index], values[index]
                y0, y1 = deltas[previous_index], deltas[index]
                if y1 == y0:
                    return float(x1)
                return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
            previous_index = index
        return None

    lower = crossing(range(minimum_index - 1, -1, -1))
    upper = crossing(range(minimum_index + 1, len(values)))
    return lower, upper


def _baseline_profile_outputs(
    completed: list[CompletedFit],
    source: Any,
    *,
    seed: int,
    n_starts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Profile the two load-bearing effective response lengths."""

    rows: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    parameter_by_model = {
        "symmetric_free": "lambda_shared_nm",
        "asymmetric": "lambda_back_nm",
    }
    for model_key, parameter in parameter_by_model.items():
        item = next(
            entry
            for entry in completed
            if entry.case.is_baseline and entry.case.model_key == model_key
        )
        optimum = float(item.result.params[parameter])
        grid = np.unique(
            np.concatenate(
                (
                    np.geomspace(max(0.2, optimum * 0.45), optimum * 1.8, 31),
                    np.asarray([optimum]),
                )
            )
        )
        profile = profile_parameter(
            item.data,
            source,
            item.spec,
            parameter,
            grid,
            reference_fit=item.result,
            seed=seed,
            n_starts=n_starts,
        )
        model_rows = [
            {
                "case_id": item.case.case_id,
                "model_key": model_key,
                "parameter": parameter,
                "value": point.value,
                "success": point.success,
                "deviance": point.deviance,
                "delta_deviance": point.delta_deviance,
                "message": point.message,
            }
            for point in profile.points
        ]
        rows.extend(model_rows)
        interval_68 = _profile_crossing_interval(model_rows, threshold=1.0)
        interval_95 = _profile_crossing_interval(
            model_rows,
            threshold=3.841458820694124,
        )
        intervals.append(
            {
                "case_id": item.case.case_id,
                "model_key": model_key,
                "parameter": parameter,
                "optimum": optimum,
                "working_likelihood_interval_68_lower": interval_68[0],
                "working_likelihood_interval_68_upper": interval_68[1],
                "working_likelihood_interval_95_lower": interval_95[0],
                "working_likelihood_interval_95_upper": interval_95[1],
                "interpretation": (
                    "Conditional profile interval under the Gaussian EIV "
                    "working likelihood; excludes source and calibration "
                    "systematics."
                ),
            }
        )
    return rows, intervals


def _parameter_shift_rows(completed: list[CompletedFit]) -> list[dict[str, Any]]:
    """Compare each parameter with the corresponding S0 model baseline."""

    baseline = {
        item.case.model_key: item
        for item in completed
        if item.case.is_baseline
    }
    rows: list[dict[str, Any]] = []
    for item in completed:
        reference = baseline.get(item.case.model_key)
        if reference is None:
            continue
        reference_diagonal = np.diag(reference.result.covariance)
        reference_uncertainty_valid = bool(
            reference.result.hessian_positive_definite
            and reference.result.hessian_rank
            == reference.result.n_free_parameters
            and np.all(np.isfinite(reference_diagonal))
        )
        for name in item.result.parameter_names:
            if name not in reference.result.params:
                continue
            value = item.result.params[name]
            reference_value = reference.result.params[name]
            parameter_index = reference.result.parameter_names.index(name)
            reference_sigma = (
                float(np.sqrt(reference_diagonal[parameter_index]))
                if reference_uncertainty_valid
                and reference_diagonal[parameter_index] > 0.0
                else np.nan
            )
            shift = value - reference_value
            rows.append(
                {
                    "case_id": item.case.case_id,
                    "baseline_case_id": reference.case.case_id,
                    "model_key": item.case.model_key,
                    "parameter": name,
                    "value": value,
                    "baseline_value": reference_value,
                    "absolute_shift": shift,
                    "baseline_local_standard_error": reference_sigma,
                    "shift_in_baseline_sigma": (
                        shift / reference_sigma
                        if np.isfinite(reference_sigma) and reference_sigma > 0.0
                        else np.nan
                    ),
                }
            )
    return rows


def _plot_baseline(
    completed: list[CompletedFit],
    source: IonizationProfile,
    output_dir: Path,
    *,
    dpi: int,
    analysis_quality: str,
) -> None:
    baseline = {
        item.case.model_key: item for item in completed if item.case.is_baseline
    }
    expected = {spec.key for spec in paper_model_specs()}
    if set(baseline) != expected:
        missing = sorted(expected.difference(baseline))
        raise RuntimeError(f"Cannot plot baseline comparison; missing {missing}.")

    first = next(iter(baseline.values()))
    data = first.data
    curve_x = np.linspace(0.0, 1.05 * np.max(data.x), 500)

    figure, (fit_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(6.6, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.2), "hspace": 0.08},
    )
    fit_axis.errorbar(
        data.x,
        data.y,
        xerr=data.x_sigma,
        yerr=np.sqrt(np.diag(data.y_cov)),
        fmt="o",
        color="black",
        markerfacecolor="white",
        markersize=4.0,
        elinewidth=0.8,
        capsize=1.5,
        label="Data: posterior mean and marginal SD",
        zorder=4,
    )

    for spec in paper_model_specs():
        item = baseline[spec.key]
        style = MODEL_STYLES[spec.key]
        curve = np.asarray(
            predict_model(spec, source, curve_x, item.result.params),
            dtype=float,
        )
        label = (
            f"{spec.label}: AICc={item.result.aicc:.2f}, "
            f"$M^2$={item.result.likelihood.mahalanobis:.2f}"
        )
        fit_axis.plot(
            curve_x,
            curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            label=label,
        )
        marginal_sigma = np.sqrt(np.diag(item.result.likelihood.covariance))
        standardized = item.result.likelihood.residual / marginal_sigma
        residual_axis.plot(
            item.data.x,
            standardized,
            linestyle="none",
            marker=style["marker"],
            markersize=4.0,
            color=style["color"],
            label=spec.label,
        )

    fit_axis.set_ylabel("Normalized collected intensity")
    fit_axis.set_ylim(bottom=0.0)
    fit_axis.legend(fontsize=7.5, loc="best")
    fit_axis.set_title(
        "S0 model comparison: full-covariance Gaussian EIV\n"
        "Fit statistics are descriptive; no mechanism-selection claim"
    )
    if analysis_quality == "smoke":
        fit_axis.text(
            0.5,
            0.5,
            "SMOKE TEST — NOT FOR SCIENTIFIC USE",
            transform=fit_axis.transAxes,
            ha="center",
            va="center",
            rotation=24,
            fontsize=14,
            color="#aa0000",
            alpha=0.24,
            weight="bold",
            zorder=10,
        )
    residual_axis.axhline(0.0, color="#777777", linewidth=0.8)
    residual_axis.set_xlabel("EELS thickness (nm)")
    residual_axis.set_ylabel("Marginal\nresidual")
    residual_axis.grid(axis="y", alpha=0.2)
    figure.align_ylabels()

    pdf_path = output_dir / "model_comparison.pdf"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(
        output_dir / "model_comparison.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_source_profiles(
    sources: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int,
    analysis_quality: str,
) -> None:
    """Render the source terms and finite/long correction used in fits."""

    depth = np.linspace(0.0, 300.0, 1201)
    figure, (profile_axis, ratio_axis) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.0),
    )
    profile_axis.plot(
        depth,
        sources["S0"].evaluate(depth),
        label="Ions",
        color="#1f77b4",
    )
    profile_axis.plot(
        depth,
        sources["S2"].evaluate(depth),
        label="Ions + recoil ionization",
        color="#d62728",
    )
    profile_axis.set_xlabel("Raw SRIM depth (nm)")
    profile_axis.set_ylabel("Electronic ionization (eV/nm/ion)")
    profile_axis.set_yscale("log")
    profile_axis.set_ylim(bottom=1e-3)
    profile_axis.legend(fontsize=8)

    for thickness in (10.0, 50.0, 100.0, 200.0):
        local_depth = np.linspace(0.0, thickness, 401)
        ratio_axis.plot(
            local_depth / thickness,
            sources["S1"].ratio_raw(local_depth, thickness),
            label=f"{thickness:g} nm",
        )
    ratio_axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ratio_axis.set_xlabel("Fractional depth, z/t")
    ratio_axis.set_ylabel("Finite/long ion-source ratio")
    ratio_axis.legend(fontsize=8)
    ratio_axis.set_ylim(0.85, 1.01)
    figure.suptitle("SRIM electronic-ionization source sensitivities")
    if analysis_quality == "smoke":
        figure.text(
            0.5,
            0.5,
            "SMOKE TEST — NOT FOR SCIENTIFIC USE",
            ha="center",
            va="center",
            rotation=24,
            fontsize=15,
            color="#aa0000",
            alpha=0.24,
            weight="bold",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "source_profiles.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "source_profiles.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--posterior-dir",
        type=Path,
        default=DEFAULT_POSTERIOR_DIR,
        help=(
            "Directory containing all three *_posterior_ratio_samples.npz "
            "files. The default is the intended archived-data location."
        ),
    )
    parser.add_argument("--long-ioniz", type=Path, default=DEFAULT_LONG_IONIZ)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Fresh output directory. Defaults to outputs/model_sensitivity "
            "for production and outputs/model_sensitivity_smoke for --quick."
        ),
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help=(
            "Starts per campaign fit (production default: 12; smoke default: 3)."
        ),
    )
    parser.add_argument(
        "--baseline-verification-starts",
        type=int,
        default=None,
        help=(
            "Starts for each of the three S0 baseline fits (production "
            "default: 100; smoke default: 3)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--finite-step-nm",
        type=float,
        default=None,
        help=(
            "Finite-source quadrature step (production default: 0.05 nm; "
            "smoke default: 0.5 nm). Production also verifies against half "
            "this step."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke mode: one optimization start and a coarser finite-source "
            "integration grid. The full campaign definition is retained."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    analysis_quality = "smoke" if args.quick else "production"
    summary_path = args.summary.resolve()
    posterior_dir = args.posterior_dir.resolve()
    long_ioniz_path = args.long_ioniz.resolve()
    sweep_dir = args.sweep_dir.resolve()
    default_output = (
        DEFAULT_SMOKE_OUTPUT_DIR if args.quick else DEFAULT_OUTPUT_DIR
    )
    requested_output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output.resolve()
    )

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not posterior_dir.is_dir():
        raise FileNotFoundError(
            f"Missing posterior directory: {posterior_dir}\n"
            "Run the production Bayesian normalization or pass "
            "--posterior-dir explicitly."
        )
    if not long_ioniz_path.is_file():
        raise FileNotFoundError(f"Missing long SRIM profile: {long_ioniz_path}")
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Missing finite-thickness sweep: {sweep_dir}")
    normalization_quality = _normalization_quality(
        posterior_dir,
        require_production=not args.quick,
    )

    n_starts = args.n_starts if args.n_starts is not None else (3 if args.quick else 12)
    if n_starts < 1:
        raise ValueError("--n-starts must be positive.")
    baseline_verification_starts = (
        args.baseline_verification_starts
        if args.baseline_verification_starts is not None
        else (3 if args.quick else 100)
    )
    if baseline_verification_starts < 1:
        raise ValueError("--baseline-verification-starts must be positive.")
    finite_step = (
        args.finite_step_nm
        if args.finite_step_nm is not None
        else (0.5 if args.quick else 0.05)
    )
    if finite_step <= 0.0:
        raise ValueError("--finite-step-nm must be positive.")
    output_dir = _prepare_output_directory(requested_output)

    base_data = load_observations(
        summary_path,
        posterior_dir=posterior_dir,
        pointwise_fraction=0.10,
        covariance_mode="full",
    )
    _require_full_covariance(base_data)

    ion_z, ion_deposition = load_srim(long_ioniz_path, include_recoils=False)
    recoil_z, recoil_deposition = load_srim(
        long_ioniz_path,
        include_recoils=True,
    )
    source_s0 = IonizationProfile(ion_z, ion_deposition)
    source_s2 = IonizationProfile(recoil_z, recoil_deposition)
    finite_kwargs = {
        "repository_root": REPOSITORY_ROOT,
        "long_ioniz_path": long_ioniz_path,
        "sweep_directory": sweep_dir,
        "integration_step_nm": finite_step,
        "minimum_cells": 32 if args.quick else 64,
        "maximum_cells": 1024 if args.quick else 16384,
    }
    source_s1 = load_finite_source(include_recoils=False, **finite_kwargs)
    source_s3 = load_finite_source(include_recoils=True, **finite_kwargs)
    interpolation_step = 1.0 if args.quick else 0.10
    source_i1, diagnostics_i1 = resample_profile(
        source_s0,
        interpolation="pchip",
        tail_policy="zero",
        step_nm=interpolation_step,
    )
    source_i2, diagnostics_i2 = resample_profile(
        source_s0,
        interpolation="cubic",
        tail_policy="endpoint",
        step_nm=interpolation_step,
        endpoint_extension_nm=2000.0,
    )
    sources = {
        "S0": source_s0,
        "S1": source_s1,
        "S2": source_s2,
        "S3": source_s3,
        "I1": source_i1,
        "I2": source_i2,
    }

    dataset_labels = tuple(dict.fromkeys(base_data.dataset.astype(str)))
    campaign = build_campaign(dataset_labels)
    completed: list[CompletedFit] = []
    failures: list[dict[str, Any]] = []

    for index, case in enumerate(campaign, start=1):
        print(f"[{index:02d}/{len(campaign)}] {case.case_id}", flush=True)
        specs = {spec.key: spec for spec in paper_model_specs(case.lambda_front_nm)}
        spec = specs[case.model_key]
        fit_data = _transformed_data(
            base_data,
            covariance_mode=case.covariance_mode,
            eels_scale=case.eels_scale,
            pointwise_fraction=case.pointwise_fraction,
            excluded_dataset=case.excluded_dataset,
        )
        try:
            case_starts = (
                max(n_starts, baseline_verification_starts)
                if case.is_baseline
                else n_starts
            )
            result = fit_model(
                fit_data,
                sources[case.source_key],
                spec,
                seed=args.seed,
                n_starts=case_starts,
                additional_starts=_case_anchor_starts(case, spec, completed),
                fixed_params=(
                    None
                    if case.fixed_scale is None
                    else {"scale": case.fixed_scale}
                ),
            )
        except Exception as error:
            failures.append(
                {
                    "case": asdict(case),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            continue
        completed.append(
            CompletedFit(
                case=case,
                spec=spec,
                data=fit_data,
                result=result,
            )
        )

    profile_rows: list[dict[str, Any]] = []
    profile_intervals: list[dict[str, Any]] = []
    profile_failure: dict[str, Any] | None = None
    if not args.quick and all(
        any(item.case.is_baseline and item.case.model_key == key for item in completed)
        for key in ("symmetric_free", "asymmetric")
    ):
        try:
            profile_rows, profile_intervals = _baseline_profile_outputs(
                completed,
                source_s0,
                seed=args.seed,
                n_starts=max(4, min(n_starts, 8)),
            )
        except Exception as error:
            profile_failure = {
                "severity": "error",
                "code": "baseline_profile_failure",
                "case_ids": [],
                "message": f"{type(error).__name__}: {error}",
            }

    integrity_issues = _integrity_issues(
        completed,
        campaign,
        production=not args.quick,
    )
    if profile_failure is not None:
        integrity_issues.append(profile_failure)
    for source_key, source in (("S1", source_s1), ("S3", source_s3)):
        selected = next(
            fit
            for fit in source.diagnostics.kernel_fits
            if fit.name == source.diagnostics.selected_kernel
        )
        if selected.reduced_chi2 > 2.0:
            integrity_issues.append(
                {
                    "severity": "warning",
                    "code": "finite_source_approximation",
                    "case_ids": [],
                    "source_key": source_key,
                    "message": (
                        f"{source_key} finite/long correction has "
                        f"chi-square/dof={selected.reduced_chi2:.3f}; use it "
                        "only as an approximate source sensitivity and do not "
                        "interpret its local fit errors as complete uncertainty."
                    ),
                }
            )

    convergence_rows: list[dict[str, Any]] = []
    if not args.quick:
        reference_kwargs = {
            **finite_kwargs,
            "integration_step_nm": finite_step / 2.0,
        }
        reference_sources = {
            "S1": load_finite_source(
                include_recoils=False,
                **reference_kwargs,
            ),
            "S3": load_finite_source(
                include_recoils=True,
                **reference_kwargs,
            ),
        }
        convergence_rows = _finite_grid_convergence_rows(
            completed,
            sources,
            reference_sources,
        )
        failed_convergence = [
            row for row in convergence_rows if not bool(row["passed"])
        ]
        if failed_convergence:
            integrity_issues.append(
                {
                    "severity": "error",
                    "code": "finite_grid_nonconvergence",
                    "case_ids": [row["case_id"] for row in failed_convergence],
                    "message": (
                        "Finite-source prediction changed by at least 5e-5 "
                        "of its peak when the integration step was halved."
                    ),
                }
            )

    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "finite_grid_convergence.csv",
        index=False,
    )
    pd.DataFrame(profile_rows).to_csv(
        output_dir / "profile_likelihood.csv",
        index=False,
    )
    (output_dir / "profile_intervals.json").write_text(
        json.dumps(_jsonable(profile_intervals), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    error_case_ids = {
        case_id
        for issue in integrity_issues
        if issue["severity"] == "error"
        for case_id in issue.get("case_ids", [])
    }
    warning_case_ids = {
        case_id
        for issue in integrity_issues
        if issue["severity"] == "warning"
        for case_id in issue.get("case_ids", [])
    }
    invalid_case_ids = error_case_ids | warning_case_ids
    failed_case_ids = {
        CampaignCase(**failure["case"]).case_id for failure in failures
    }

    def case_integrity_status(case_id: str) -> str:
        if case_id in error_case_ids:
            return "invalid"
        if case_id in warning_case_ids:
            return "diagnostic_limit"
        return "valid"

    fit_summary = pd.DataFrame(
        [_fit_summary_row(item) for item in completed]
    )
    if not fit_summary.empty:
        fit_summary = _comparison_deltas(fit_summary, invalid_case_ids)
        fit_summary.insert(0, "analysis_quality", analysis_quality)
        fit_summary.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in fit_summary["case_id"]
            ],
        )
    fit_summary.to_csv(output_dir / "fit_summary.csv", index=False)
    parameter_frame = pd.DataFrame(
        [row for item in completed for row in _parameter_rows(item)]
    )
    if not parameter_frame.empty:
        parameter_frame.insert(0, "analysis_quality", analysis_quality)
        parameter_frame.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in parameter_frame["case_id"]
            ],
        )
    parameter_frame.to_csv(output_dir / "parameter_summary.csv", index=False)
    observation_frame = pd.DataFrame(
        [row for item in completed for row in _observation_rows(item)]
    )
    if not observation_frame.empty:
        observation_frame.insert(0, "analysis_quality", analysis_quality)
        observation_frame.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in observation_frame["case_id"]
            ],
        )
    observation_frame.to_csv(
        output_dir / "observation_diagnostics.csv",
        index=False,
    )

    derived_frame = pd.DataFrame(
        [
            _derived_fit_row(item, sources[item.case.source_key])
            for item in completed
        ]
    )
    if not derived_frame.empty:
        derived_frame.insert(0, "analysis_quality", analysis_quality)
        derived_frame.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in derived_frame["case_id"]
            ],
        )
    derived_frame.to_csv(output_dir / "derived_fit_summary.csv", index=False)
    prediction_frame = pd.DataFrame(
        [
            row
            for item in completed
            for row in _prediction_rows(
                item,
                sources[item.case.source_key],
                grid_points=101 if args.quick else 301,
            )
        ]
    )
    if not prediction_frame.empty:
        prediction_frame.insert(0, "analysis_quality", analysis_quality)
        prediction_frame.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in prediction_frame["case_id"]
            ],
        )
    prediction_frame.to_csv(output_dir / "prediction_curves.csv", index=False)
    shift_frame = pd.DataFrame(_parameter_shift_rows(completed))
    if not shift_frame.empty:
        shift_frame.insert(0, "analysis_quality", analysis_quality)
        shift_frame.insert(
            1,
            "integrity_status",
            [
                case_integrity_status(case_id)
                for case_id in shift_frame["case_id"]
            ],
        )
    shift_frame.to_csv(output_dir / "parameter_shifts.csv", index=False)

    campaign_rows = []
    for case in campaign:
        row = {
            "analysis_quality": analysis_quality,
            "case_id": case.case_id,
            "comparison_key": case.comparison_key,
            **asdict(case),
        }
        if case.case_id in failed_case_ids:
            row["execution_status"] = "failed"
        elif case.case_id in error_case_ids:
            row["execution_status"] = "integrity_invalid"
        elif case.case_id in warning_case_ids:
            row["execution_status"] = "diagnostic_limit"
        elif any(item.case.case_id == case.case_id for item in completed):
            row["execution_status"] = "valid"
        else:
            row["execution_status"] = "missing"
        campaign_rows.append(row)
    pd.DataFrame(campaign_rows).to_csv(
        output_dir / "campaign_cases.csv",
        index=False,
    )

    try:
        _plot_baseline(
            completed,
            source_s0,
            output_dir,
            dpi=args.dpi,
            analysis_quality=analysis_quality,
        )
    except Exception as error:
        integrity_issues.append(
            {
                "severity": "error",
                "code": "baseline_figure_failure",
                "case_ids": [],
                "message": f"{type(error).__name__}: {error}",
            }
        )
    try:
        _plot_source_profiles(
            sources,
            output_dir,
            dpi=args.dpi,
            analysis_quality=analysis_quality,
        )
    except Exception as error:
        integrity_issues.append(
            {
                "severity": "error",
                "code": "source_figure_failure",
                "case_ids": [],
                "message": f"{type(error).__name__}: {error}",
            }
        )

    input_paths = [
        summary_path,
        long_ioniz_path,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "requirements-lock.txt",
        posterior_dir / "sampling_quality.json",
        posterior_dir / "joint_normalization_metadata.json",
        posterior_dir / "artifact_manifest.json",
        posterior_dir / "joint_normalization_posterior.npz",
        posterior_dir / "run_provenance.json",
    ]
    input_paths.extend(sorted((REPOSITORY_ROOT / "src" / "se_model").glob("*.py")))
    posterior_paths = sorted(posterior_dir.rglob("*_posterior_ratio_samples.npz"))
    sweep_paths = sorted(sweep_dir.glob("*nm/IONIZ.txt"))
    input_paths.extend(posterior_paths)
    input_paths.extend(sweep_paths)
    input_paths = list(
        {
            str(path.resolve()): path.resolve()
            for path in input_paths
            if path.is_file()
        }.values()
    )
    input_paths.sort(key=str)
    error_issues = [
        issue for issue in integrity_issues if issue["severity"] == "error"
    ]
    warning_issues = [
        issue for issue in integrity_issues if issue["severity"] == "warning"
    ]
    run_success = not failures and not error_issues and len(completed) == len(campaign)
    outcome = (
        "failed"
        if not run_success
        else ("success_with_warnings" if warning_issues else "success")
    )
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "analysis_quality": analysis_quality,
        "run_success": run_success,
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
            "sweep_dir": sweep_dir,
            "output_dir": requested_output,
            "n_starts": n_starts,
            "baseline_verification_starts": baseline_verification_starts,
            "finite_step_nm": finite_step,
            "finite_reference_step_nm": (
                None if args.quick else finite_step / 2.0
            ),
        },
        "normalization_quality": normalization_quality,
        "input_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "base_observation_covariance": {
            "shape": list(base_data.y_cov.shape),
            "datasets": list(dataset_labels),
            "off_diagonal_elements": int(
                np.count_nonzero(base_data.y_cov - np.diag(np.diag(base_data.y_cov)))
            ),
        },
        "sources": {
            "S0": {
                "label": SOURCE_LABELS["S0"],
                "include_recoils": False,
                "long_ioniz_path": str(long_ioniz_path),
            },
            "S1": {
                "label": SOURCE_LABELS["S1"],
                **dict(finite_source_metadata(source_s1)),
            },
            "S2": {
                "label": SOURCE_LABELS["S2"],
                "include_recoils": True,
                "long_ioniz_path": str(long_ioniz_path),
            },
            "S3": {
                "label": SOURCE_LABELS["S3"],
                **dict(finite_source_metadata(source_s3)),
            },
            "I1": {
                "label": SOURCE_LABELS["I1"],
                **asdict(diagnostics_i1),
            },
            "I2": {
                "label": SOURCE_LABELS["I2"],
                **asdict(diagnostics_i2),
            },
        },
        "campaign": {
            "planned_fits": len(campaign),
            "successful_fits": len(completed),
            "diagnostically_valid_fits": sum(
                item.case.case_id not in invalid_case_ids for item in completed
            ),
            "diagnostic_limit_fits": len(warning_case_ids),
            "integrity_invalid_fits": len(error_case_ids),
            "failed_fits": len(failures),
            "integrity_issue_count": len(integrity_issues),
            "outcome": outcome,
            "finite_grid_convergence": convergence_rows,
            "working_likelihood_profile_intervals": profile_intervals,
            "likelihood": (
                "parameter-dependent full Gaussian EIV with covariance log-determinant"
            ),
            "interpretation": (
                "Descriptive statistical sensitivity only; no microscopic "
                "mechanism-selection claim."
            ),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(_jsonable(provenance), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    results_payload = {
        "provenance_file": "provenance.json",
        "analysis_quality": analysis_quality,
        "run_success": run_success,
        "outcome": outcome,
        "finite_grid_convergence": convergence_rows,
        "working_likelihood_profile_intervals": profile_intervals,
        "fits": [
            {
                "case": asdict(item.case),
                "result": item.result.to_dict(
                    include_likelihood_arrays=True,
                    include_start_records=True,
                ),
            }
            for item in completed
        ],
        "failures": failures,
        "integrity_issues": integrity_issues,
    }
    (output_dir / "fit_results.json").write_text(
        json.dumps(_jsonable(results_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    status_payload = {
        "schema_version": 2,
        "analysis_quality": analysis_quality,
        "success": run_success,
        "outcome": outcome,
        "planned_fits": len(campaign),
        "completed_fits": len(completed),
        "diagnostically_valid_fits": sum(
            item.case.case_id not in invalid_case_ids for item in completed
        ),
        "diagnostic_limit_fits": len(warning_case_ids),
        "integrity_invalid_fits": len(error_case_ids),
        "failed_fits": len(failures),
        "integrity_errors": len(error_issues),
        "integrity_warnings": len(warning_issues),
        "normalization_quality_passed": normalization_quality["passed"],
        "failures": failures,
        "integrity_issues": integrity_issues,
    }
    (output_dir / "status.json").write_text(
        json.dumps(_jsonable(status_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _finalize_output_directory(output_dir, requested_output)

    print(
        f"Completed {len(completed)}/{len(campaign)} fits; "
        f"outcome={outcome}; "
        f"outputs: {requested_output}",
        flush=True,
    )
    if not run_success:
        raise SystemExit(
            "Campaign failed its completeness/integrity gate; inspect status.json."
        )


if __name__ == "__main__":
    main()

"""Normalize grouped SEM intensities to a shared carbon reference.

For each image, the model estimates one latent carbon intensity and one latent
MoS2 intensity for every thickness group. ROI observations use Student-t
likelihoods so that an unusual ROI has less leverage than it would under a
Gaussian model. The reported normalized intensity is the posterior ratio
``mu_mos2 / mu_carbon``.

Dataset paths and carbon-reference groups are defined in
``config/datasets.json``. Outputs are written below ``outputs/bayesian`` unless
another directory is supplied.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "datasets.json"


@dataclass(frozen=True)
class SamplingSettings:
    """PyMC sampling controls shared by all datasets."""

    draws: int
    tune: int
    chains: int
    cores: int
    target_accept: float
    random_seed: int


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels while preserving the underlying values."""
    frame = frame.copy()
    frame.columns = [
        str(column).strip().lower().replace(" ", "_") for column in frame.columns
    ]
    return frame.dropna(axis=1, how="all")


def read_roi_table(path: Path) -> pd.DataFrame:
    """Read a comma-, tab-, or whitespace-separated ROI table."""
    return clean_columns(pd.read_csv(path, sep=None, engine="python"))


def posterior_interval(samples: np.ndarray) -> np.ndarray:
    """Return the 2.5, 16, 50, 84, and 97.5 percentiles."""
    return np.percentile(samples, [2.5, 16.0, 50.0, 84.0, 97.5])


def safe_name(label: str) -> str:
    """Convert a display label to a filesystem-friendly stem."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    return normalized.strip("_")


def load_datasets(config_path: Path, condition: str) -> list[dict[str, Any]]:
    """Load and resolve one condition from the repository dataset config."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if condition not in payload:
        raise KeyError(
            f"Condition {condition!r} is absent from {config_path}. "
            f"Available conditions: {sorted(payload)}"
        )

    datasets = []
    for record in payload[condition]:
        resolved = dict(record)
        resolved["roi_csv"] = REPOSITORY_ROOT / record["roi_csv"]
        datasets.append(resolved)
    return datasets


def prepare_observations(
    roi_path: Path,
    carbon_group_id: int,
    intensity_column: str,
    group_column: str,
    thickness_column: str,
) -> dict[str, Any]:
    """Validate an ROI table and return arrays used by the model."""
    frame = read_roi_table(roi_path)
    required = {intensity_column, group_column, thickness_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{roi_path} is missing columns {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[group_column, intensity_column]).copy()
    frame[group_column] = frame[group_column].astype(int)

    carbon = frame[frame[group_column] == carbon_group_id].copy()
    mos2 = frame[frame[group_column] != carbon_group_id].copy()
    if carbon.empty:
        raise ValueError(f"{roi_path} has no carbon ROIs in group {carbon_group_id}.")
    if mos2.empty:
        raise ValueError(f"{roi_path} has no non-carbon ROI groups.")

    group_ids = sorted(mos2[group_column].unique())
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    observation_group_index = mos2[group_column].map(group_index).to_numpy(dtype=int)
    grouped = mos2.groupby(group_column)

    return {
        "frame": frame,
        "carbon": carbon,
        "mos2": mos2,
        "group_ids": group_ids,
        "group_index": observation_group_index,
        "carbon_values": carbon[intensity_column].to_numpy(dtype=float),
        "mos2_values": mos2[intensity_column].to_numpy(dtype=float),
        "thicknesses": (grouped[thickness_column].mean().reindex(group_ids).to_numpy()),
        "roi_counts": (grouped[intensity_column].count().reindex(group_ids).to_numpy()),
        "raw_means": (grouped[intensity_column].mean().reindex(group_ids).to_numpy()),
        "raw_stds": (
            grouped[intensity_column].std(ddof=1).reindex(group_ids).to_numpy()
        ),
    }


def build_and_sample_model(
    observations: dict[str, Any],
    settings: SamplingSettings,
    progressbar: bool,
) -> az.InferenceData:
    """Construct and sample the robust shared-reference model."""
    frame = observations["frame"]
    all_values = frame["mean_0_255"].to_numpy(dtype=float)
    data_sd = float(np.std(all_values, ddof=1))
    if not np.isfinite(data_sd) or data_sd <= 0.0:
        data_sd = 10.0

    carbon_values = observations["carbon_values"]
    mos2_values = observations["mos2_values"]
    raw_means = observations["raw_means"]
    group_index = observations["group_index"]
    group_count = len(observations["group_ids"])

    with pm.Model():
        mu_carbon = pm.Normal(
            "mu_carbon",
            mu=float(np.mean(carbon_values)),
            sigma=5.0 * data_sd,
        )
        mu_mos2 = pm.Normal(
            "mu_mos2",
            mu=raw_means,
            sigma=5.0 * data_sd,
            shape=group_count,
        )

        # These terms describe variation among selected ROIs, not pixel noise.
        sigma_carbon_roi = pm.HalfNormal(
            "sigma_carbon_roi",
            sigma=data_sd,
        )
        sigma_mos2_roi = pm.HalfNormal(
            "sigma_mos2_roi",
            sigma=data_sd,
        )
        nu = pm.Deterministic(
            "nu",
            pm.Exponential("nu_minus_two", lam=1.0 / 30.0) + 2.0,
        )

        pm.StudentT(
            "carbon_obs",
            nu=nu,
            mu=mu_carbon,
            sigma=sigma_carbon_roi,
            observed=carbon_values,
        )
        pm.StudentT(
            "mos2_obs",
            nu=nu,
            mu=mu_mos2[group_index],
            sigma=sigma_mos2_roi,
            observed=mos2_values,
        )

        # The shared denominator automatically propagates carbon-reference
        # uncertainty and correlation to every thickness group in the image.
        pm.Deterministic("normalized_intensity", mu_mos2 / mu_carbon)

        return pm.sample(
            draws=settings.draws,
            tune=settings.tune,
            chains=settings.chains,
            cores=settings.cores,
            # Keep the BLAS library's own thread setting. This avoids a
            # threadpoolctl crash seen with some Windows scientific stacks.
            blas_cores=None,
            target_accept=settings.target_accept,
            random_seed=settings.random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
        )


def build_summary(
    trace: az.InferenceData,
    observations: dict[str, Any],
    dataset_label: str,
    source_file: str,
    carbon_group_id: int,
) -> pd.DataFrame:
    """Convert posterior samples to the tabular result used downstream."""
    group_ids = observations["group_ids"]
    group_count = len(group_ids)
    ratio_samples = trace.posterior["normalized_intensity"].values.reshape(
        -1,
        group_count,
    )
    mos2_samples = trace.posterior["mu_mos2"].values.reshape(-1, group_count)
    carbon_samples = trace.posterior["mu_carbon"].values.reshape(-1)
    carbon_interval = posterior_interval(carbon_samples)

    rows = []
    for index, group_id in enumerate(group_ids):
        ratio_interval = posterior_interval(ratio_samples[:, index])
        mean_interval = posterior_interval(mos2_samples[:, index])
        rows.append(
            {
                "dataset": dataset_label,
                "source_file": source_file,
                "group_id": int(group_id),
                "carbon_group_id": int(carbon_group_id),
                "thickness_nm": observations["thicknesses"][index],
                "n_roi": int(observations["roi_counts"][index]),
                "raw_roi_mean_0_255": observations["raw_means"][index],
                "raw_roi_std_0_255": observations["raw_stds"][index],
                "bayes_mu_mos2_median_0_255": mean_interval[2],
                "bayes_mu_mos2_ci16_0_255": mean_interval[1],
                "bayes_mu_mos2_ci84_0_255": mean_interval[3],
                "bayes_mu_mos2_ci2p5_0_255": mean_interval[0],
                "bayes_mu_mos2_ci97p5_0_255": mean_interval[4],
                "carbon_mu_median_0_255": carbon_interval[2],
                "carbon_mu_ci16_0_255": carbon_interval[1],
                "carbon_mu_ci84_0_255": carbon_interval[3],
                "carbon_mu_ci2p5_0_255": carbon_interval[0],
                "carbon_mu_ci97p5_0_255": carbon_interval[4],
                "normalized_median": ratio_interval[2],
                "normalized_ci16": ratio_interval[1],
                "normalized_ci84": ratio_interval[3],
                "normalized_ci2p5": ratio_interval[0],
                "normalized_ci97p5": ratio_interval[4],
                "normalized_err_lower_68": (ratio_interval[2] - ratio_interval[1]),
                "normalized_err_upper_68": (ratio_interval[3] - ratio_interval[2]),
                "normalized_err_lower_95": (ratio_interval[2] - ratio_interval[0]),
                "normalized_err_upper_95": (ratio_interval[4] - ratio_interval[2]),
            }
        )

    return pd.DataFrame(rows).sort_values("thickness_nm").reset_index(drop=True)


def save_dataset_plot(
    summary: pd.DataFrame,
    observations: dict[str, Any],
    dataset_label: str,
    path: Path,
) -> None:
    """Plot posterior group estimates with the individual ROI means."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    axes.errorbar(
        summary["thickness_nm"],
        summary["normalized_median"],
        yerr=[
            summary["normalized_err_lower_68"],
            summary["normalized_err_upper_68"],
        ],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.5,
        label="Posterior median ± 68% credible interval",
    )

    carbon_median = float(summary["carbon_mu_median_0_255"].iloc[0])
    roi_frame = observations["mos2"].copy()
    roi_frame["normalized_raw_roi"] = roi_frame["mean_0_255"] / carbon_median
    axes.scatter(
        roi_frame["thickness"],
        roi_frame["normalized_raw_roi"],
        marker="x",
        alpha=0.35,
        label="Individual ROI mean / carbon median",
    )

    for row in summary.itertuples(index=False):
        axes.annotate(
            f"G{int(row.group_id)}",
            (row.thickness_nm, row.normalized_median),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    axes.set_xlabel("EELS thickness (nm)")
    axes.set_ylabel("Normalized SE grayscale intensity")
    axes.set_title(f"{dataset_label}: Bayesian-normalized SE intensity")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_combined_plot(summary: pd.DataFrame, path: Path) -> None:
    """Plot all datasets belonging to one experimental condition."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.2, 5.0))
    for dataset_label, subset in summary.groupby("dataset"):
        subset = subset.sort_values("thickness_nm")
        axes.errorbar(
            subset["thickness_nm"],
            subset["normalized_median"],
            yerr=[
                subset["normalized_err_lower_68"],
                subset["normalized_err_upper_68"],
            ],
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.5,
            label=f"{dataset_label}: median ± 68% CI",
        )
        short_label = dataset_label.replace("Dataset ", "D")
        for row in subset.itertuples(index=False):
            axes.annotate(
                f"{short_label}-G{int(row.group_id)}",
                (row.thickness_nm, row.normalized_median),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    axes.set_xlabel("EELS thickness (nm)")
    axes.set_ylabel("Normalized SE grayscale intensity")
    axes.set_title("Bayesian-normalized SE intensity vs. EELS thickness")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_condition(
    condition: str,
    config_path: Path,
    output_root: Path,
    settings: SamplingSettings,
    progressbar: bool,
    save_plots: bool,
) -> Path:
    """Run every configured dataset for one experimental condition."""
    datasets = load_datasets(config_path, condition)
    condition_dir = output_root / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for record in datasets:
        roi_path = Path(record["roi_csv"])
        label = str(record["label"])
        carbon_group_id = int(record["carbon_group_id"])
        print(f"\nProcessing {label}: {record['name']}")
        print(f"ROI table: {roi_path}")

        observations = prepare_observations(
            roi_path,
            carbon_group_id=carbon_group_id,
            intensity_column="mean_0_255",
            group_column="group_id",
            thickness_column="thickness",
        )
        low_count_groups = [
            int(group_id)
            for group_id, count in zip(
                observations["group_ids"],
                observations["roi_counts"],
            )
            if count < 2
        ]
        if low_count_groups:
            print(
                "Warning: one-ROI groups depend strongly on the shared "
                f"ROI-scatter estimate: {low_count_groups}"
            )

        trace = build_and_sample_model(
            observations,
            settings=settings,
            progressbar=progressbar,
        )
        summary = build_summary(
            trace,
            observations,
            dataset_label=label,
            source_file=roi_path.name,
            carbon_group_id=carbon_group_id,
        )
        summaries.append(summary)

        stem = safe_name(str(record["name"]))
        summary.to_csv(condition_dir / f"{stem}_summary.csv", index=False)
        az.summary(
            trace,
            var_names=[
                "mu_carbon",
                "mu_mos2",
                "sigma_carbon_roi",
                "sigma_mos2_roi",
                "nu",
                "normalized_intensity",
            ],
        ).to_csv(condition_dir / f"{stem}_diagnostics.csv")
        trace.to_netcdf(condition_dir / f"{stem}_trace.nc")

        ratio_samples = trace.posterior["normalized_intensity"].values.reshape(
            -1,
            len(observations["group_ids"]),
        )
        np.savez_compressed(
            condition_dir / f"{stem}_posterior_ratio_samples.npz",
            ratio_samples=ratio_samples,
            thickness_nm=observations["thicknesses"],
            group_ids=np.asarray(observations["group_ids"]),
            dataset_label=np.asarray([label]),
            source_file=np.asarray([roi_path.name]),
        )
        if save_plots:
            save_dataset_plot(
                summary,
                observations,
                dataset_label=label,
                path=condition_dir / f"{stem}_plot.png",
            )

    combined = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(["dataset", "thickness_nm"])
        .reset_index(drop=True)
    )
    combined_path = condition_dir / "combined_bayesian_summary.csv"
    combined.to_csv(combined_path, index=False)
    if save_plots:
        save_combined_plot(
            combined,
            condition_dir / "combined_bayesian_plot.png",
        )
    print(f"Saved combined summary: {combined_path}")
    return combined_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("transmission", "backside_blocked", "all"),
        default="all",
        help="Experimental condition to analyze (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Dataset configuration (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "bayesian",
        help="Root output directory.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Use 1 for reliable execution from Windows/Spyder.",
    )
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide PyMC progress bars.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write tables and traces without generating diagnostic plots.",
    )
    return parser


def main() -> None:
    """Run the requested condition or both configured conditions."""
    args = build_parser().parse_args()
    settings = SamplingSettings(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
    )
    conditions = (
        ("transmission", "backside_blocked")
        if args.condition == "all"
        else (args.condition,)
    )
    for condition in conditions:
        run_condition(
            condition,
            config_path=args.config.resolve(),
            output_root=args.output_dir.resolve(),
            settings=settings,
            progressbar=not args.quiet,
            save_plots=not args.skip_plots,
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()

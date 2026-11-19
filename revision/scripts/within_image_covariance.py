"""Within-image correlation matrix of the carbon-normalized SE ratios.

Ratios measured in the same image share a latent carbon-reference
denominator and are therefore correlated. ``build_correlation`` computes
these correlations from the saved posterior ratio samples of the per-image
Bayesian normalization
(``analysis/bayesian_outputs/byasian2/<image>_se_intensity_per_roi_posterior_R_samples.npz``)
and assembles a block-diagonal correlation matrix ordered like the
transmission summary table; cross-image correlations are zero.

Used for the within-image off-diagonal correlation summary in SI Sec. SVII
and imported by ``revision/scripts/refit_supplement.py`` for the generalized
least-squares covariance refit in SI Sec. SVIII.

Usage (from the repository root):
    python revision/scripts/within_image_covariance.py [--out FOLDER]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POSTERIOR_DIR = ROOT / "analysis" / "bayesian_outputs" / "byasian2"
DATA_CSV = ROOT / "data" / "reference_summaries" / "transmission_bayesian_summary.csv"
DEFAULT_OUT = ROOT / "revision" / "outputs" / "within_image_correlations"

POSTERIOR_FILES = {
    "Dataset 1": POSTERIOR_DIR / "4_se_intensity_per_roi_posterior_R_samples.npz",
    "Dataset 2": POSTERIOR_DIR / "12_se_intensity_per_roi_posterior_R_samples.npz",
    "Dataset 3": POSTERIOR_DIR / "13_se_intensity_per_roi_posterior_R_samples.npz",
}


def build_correlation(data: pd.DataFrame):
    """Return the block-diagonal correlation matrix and the per-pair rows.

    ``data`` is the transmission summary with a 0..n-1 index and a
    ``dataset`` column ('Dataset 1', 'Dataset 2', 'Dataset 3'); each of its
    thicknesses is matched to the corresponding column of the posterior
    samples of its image.
    """
    n = len(data)
    correlation = np.eye(n)
    rows = []

    for dataset, posterior_path in POSTERIOR_FILES.items():
        saved = np.load(posterior_path)
        samples = np.asarray(saved["R_samples"], float)
        thicknesses = np.asarray(saved["thickness_nm"], float)
        rho_native = np.corrcoef(samples, rowvar=False)

        global_indices = data.index[data["dataset"] == dataset].to_numpy()
        global_thicknesses = data.loc[global_indices, "thickness_nm"].to_numpy(float)
        posterior_indices = []
        for thickness in global_thicknesses:
            match = np.flatnonzero(np.isclose(thicknesses, thickness, atol=1e-8))
            if len(match) != 1:
                raise RuntimeError(
                    f"Could not uniquely match {dataset}, t={thickness} nm "
                    f"in {posterior_path.name}"
                )
            posterior_indices.append(int(match[0]))

        rho = rho_native[np.ix_(posterior_indices, posterior_indices)]
        correlation[np.ix_(global_indices, global_indices)] = rho

        for i, global_i in enumerate(global_indices):
            for j, global_j in enumerate(global_indices):
                rows.append(
                    {
                        "dataset": dataset,
                        "thickness_i_nm": float(data.loc[global_i, "thickness_nm"]),
                        "thickness_j_nm": float(data.loc[global_j, "thickness_nm"]),
                        "correlation": float(rho[i, j]),
                    }
                )

    return correlation, rows


def correlation_summary(data: pd.DataFrame, correlation: np.ndarray) -> dict:
    """Minimum, median, and maximum of the within-image off-diagonal values."""
    off_diagonal = correlation[~np.eye(len(correlation), dtype=bool)]
    within_dataset_mask = np.zeros_like(correlation, dtype=bool)
    for dataset in data["dataset"].unique():
        idx = np.flatnonzero(data["dataset"].to_numpy() == dataset)
        within_dataset_mask[np.ix_(idx, idx)] = True
    within_dataset_mask &= ~np.eye(len(correlation), dtype=bool)
    within = correlation[within_dataset_mask]
    return {
        "within_dataset_min": float(np.min(within)),
        "within_dataset_median": float(np.median(within)),
        "within_dataset_max": float(np.max(within)),
        "all_off_diagonal_median_including_cross_dataset_zeros": float(
            np.median(off_diagonal)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output folder (default: %(default)s)")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_CSV).sort_values("thickness_nm").reset_index(drop=True)
    correlation, rows = build_correlation(data)
    summary = correlation_summary(data, correlation)

    pd.DataFrame(rows).to_csv(out / "ratio_correlations.csv", index=False, lineterminator="\n")
    (out / "correlation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("within-image correlations: %.2f to %.2f, median %.2f"
          % (summary["within_dataset_min"], summary["within_dataset_max"],
             summary["within_dataset_median"]))


if __name__ == "__main__":
    main()

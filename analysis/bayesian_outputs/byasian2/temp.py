import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

# Transmission-only datasets used in Fig. 2(a)
# Exclude backside-blocked and copper-tape enhancement datasets.
sample_files = [
    "4_se_intensity_per_roi_posterior_R_samples.npz",
    "12_se_intensity_per_roi_posterior_R_samples.npz",
    "13_se_intensity_per_roi_posterior_R_samples.npz",
]

# Optional: label the dataset that contains the low-thickness rising branch
# Based on your table, Dataset 3 / 13_se... has 14.1, 23.4, 35.6, 54.1 nm.
flake3_file = "13_se_intensity_per_roi_posterior_R_samples.npz"

# Number of Monte Carlo samples to use after combining independent runs
n_mc = 100_000

random_seed = 42

output_csv = "posterior_rising_branch_probability.csv"


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_npz_samples(path):
    data = np.load(path, allow_pickle=True)

    R = data["R_samples"]              # shape: n_samples x n_groups
    t = data["thickness_nm"]           # shape: n_groups
    group_ids = data["group_ids"]      # shape: n_groups

    if "dataset_label" in data:
        label = str(data["dataset_label"][0])
    else:
        label = Path(path).stem

    if "source_file" in data:
        source_file = str(data["source_file"][0])
    else:
        source_file = Path(path).name

    return {
        "path": str(path),
        "label": label,
        "source_file": source_file,
        "R": R,
        "t": t,
        "group_ids": group_ids,
    }


def combine_independent_posteriors(datasets, n_mc=100_000, random_seed=42):
    """
    Combines independent posterior samples from separate PyMC runs.

    Each Monte Carlo row draws one posterior sample from each dataset,
    then concatenates all groups into one pooled Fig. 2(a) sample.
    """
    rng = np.random.default_rng(random_seed)

    R_blocks = []
    t_blocks = []
    group_blocks = []
    dataset_blocks = []
    source_blocks = []

    for d in datasets:
        R = d["R"]
        n_samples, n_groups = R.shape

        idx = rng.integers(0, n_samples, size=n_mc)

        R_blocks.append(R[idx, :])
        t_blocks.append(d["t"])
        group_blocks.append(d["group_ids"])
        dataset_blocks.extend([d["label"]] * n_groups)
        source_blocks.extend([d["source_file"]] * n_groups)

    R_all = np.concatenate(R_blocks, axis=1)
    t_all = np.concatenate(t_blocks)
    group_all = np.concatenate(group_blocks)

    metadata = pd.DataFrame({
        "pooled_index": np.arange(len(t_all)),
        "dataset": dataset_blocks,
        "source_file": source_blocks,
        "group_id": group_all,
        "thickness_nm": t_all,
        "posterior_median_R": np.median(R_all, axis=0),
        "posterior_ci16_R": np.percentile(R_all, 16, axis=0),
        "posterior_ci84_R": np.percentile(R_all, 84, axis=0),
    })

    return R_all, t_all, metadata


def posterior_exceedance(R_all, t_all, metadata):
    """
    Computes P(R_peak > R_thinnest).

    The peak group is fixed a priori as the group with the highest
    posterior median, not selected separately for each posterior draw.
    """
    medians = np.median(R_all, axis=0)

    g_thin = int(np.argmin(t_all))
    g_peak = int(np.argmax(medians))

    p = float(np.mean(R_all[:, g_peak] > R_all[:, g_thin]))

    result = {
        "p_peak_gt_thinnest": p,
        "g_thin_index": g_thin,
        "g_peak_index": g_peak,
        "thin_thickness_nm": float(t_all[g_thin]),
        "peak_thickness_nm": float(t_all[g_peak]),
        "thin_median_R": float(medians[g_thin]),
        "peak_median_R": float(medians[g_peak]),
    }

    return result


def format_probability(p, n_mc):
    """
    Conservative reporting convention.
    If all Monte Carlo samples satisfy the inequality,
    do not print P = 1.000. Print a lower bound.
    """
    if p >= 1.0:
        bound = 1.0 - 1.0 / n_mc
        return f"> {bound:.5f}"
    else:
        return f"{p:.5f}"


# ============================================================
# MAIN ANALYSIS
# ============================================================

datasets = []

for f in sample_files:
    path = Path(f)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {f}\n"
            "You need to rerun the Bayesian script after adding the "
            "posterior sample saving block."
        )

    datasets.append(load_npz_samples(path))

# Combine all Fig. 2(a) transmission datasets
R_all, t_all, metadata = combine_independent_posteriors(
    datasets,
    n_mc=n_mc,
    random_seed=random_seed,
)

metadata.to_csv("posterior_rising_branch_point_metadata.csv", index=False)

result = posterior_exceedance(R_all, t_all, metadata)

p_text = format_probability(result["p_peak_gt_thinnest"], n_mc)

print("\n============================================================")
print("Pooled transmission datasets: rising-branch test")
print("============================================================")
print(f"Thinnest group: {result['thin_thickness_nm']:.1f} nm")
print(f"Peak group:     {result['peak_thickness_nm']:.1f} nm")
print(f"R_thin median:  {result['thin_median_R']:.4f}")
print(f"R_peak median:  {result['peak_median_R']:.4f}")
print(f"P(R_peak > R_thinnest) = {p_text}")

# Save result
out = pd.DataFrame([{
    "test": "pooled_transmission_peak_gt_thinnest",
    "p_peak_gt_thinnest": result["p_peak_gt_thinnest"],
    "p_report_text": p_text,
    "thin_thickness_nm": result["thin_thickness_nm"],
    "peak_thickness_nm": result["peak_thickness_nm"],
    "thin_median_R": result["thin_median_R"],
    "peak_median_R": result["peak_median_R"],
    "n_mc": n_mc,
}])

# ============================================================
# Optional: same test within Flake 3 / low-thickness dataset
# ============================================================

flake3_path = Path(flake3_file)

if flake3_path.exists():
    f3 = load_npz_samples(flake3_path)

    R_f3 = f3["R"]
    t_f3 = f3["t"]

    med_f3 = np.median(R_f3, axis=0)

    g_thin_f3 = int(np.argmin(t_f3))
    g_peak_f3 = int(np.argmax(med_f3))

    p_f3 = float(np.mean(R_f3[:, g_peak_f3] > R_f3[:, g_thin_f3]))
    p_f3_text = format_probability(p_f3, R_f3.shape[0])

    print("\n============================================================")
    print("Flake 3 only: rising-branch robustness check")
    print("============================================================")
    print(f"Thinnest group: {t_f3[g_thin_f3]:.1f} nm")
    print(f"Peak group:     {t_f3[g_peak_f3]:.1f} nm")
    print(f"R_thin median:  {med_f3[g_thin_f3]:.4f}")
    print(f"R_peak median:  {med_f3[g_peak_f3]:.4f}")
    print(f"P(R_peak > R_thinnest) = {p_f3_text}")

    out2 = pd.DataFrame([{
        "test": "flake3_only_peak_gt_thinnest",
        "p_peak_gt_thinnest": p_f3,
        "p_report_text": p_f3_text,
        "thin_thickness_nm": float(t_f3[g_thin_f3]),
        "peak_thickness_nm": float(t_f3[g_peak_f3]),
        "thin_median_R": float(med_f3[g_thin_f3]),
        "peak_median_R": float(med_f3[g_peak_f3]),
        "n_mc": R_f3.shape[0],
    }])

    out = pd.concat([out, out2], ignore_index=True)

out.to_csv(output_csv, index=False)

print(f"\nSaved result: {output_csv}")
print("Saved point metadata: posterior_rising_branch_point_metadata.csv")
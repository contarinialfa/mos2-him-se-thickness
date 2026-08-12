import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

# Transmission-only datasets used in Fig. 2(a)
# Exclude backside-blocked and copper-tape datasets.
sample_files = [
    "4_se_intensity_per_roi_posterior_R_samples.npz",
    "12_se_intensity_per_roi_posterior_R_samples.npz",
    "13_se_intensity_per_roi_posterior_R_samples.npz",
]

# Number of Monte Carlo rows to use when combining independent PyMC runs
n_mc = 100_000
random_seed = 42

out_summary_csv = "posterior_peak_position_summary.csv"
out_prob_csv = "posterior_peak_position_group_probabilities.csv"
out_samples_csv = "posterior_peak_position_samples.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_npz_samples(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    R = data["R_samples"]          # shape: n_samples x n_groups
    t = data["thickness_nm"]       # shape: n_groups
    group_ids = data["group_ids"]  # shape: n_groups

    if "dataset_label" in data:
        label = str(data["dataset_label"][0])
    else:
        label = path.stem

    if "source_file" in data:
        source_file = str(data["source_file"][0])
    else:
        source_file = path.name

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
    Combine independent posterior samples from separate PyMC runs.

    For each Monte Carlo row, randomly draw one posterior sample from each
    dataset, then concatenate all thickness groups into one Fig. 2(a) sample.
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

    R_all = np.concatenate(R_blocks, axis=1)   # n_mc x total_groups
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


def discrete_peak_distribution(R_all, t_all, metadata):
    """
    Primary peak-position estimate.

    For each posterior draw, find the group with maximum R.
    The peak-position posterior is therefore discrete over measured
    EELS thickness groups.
    """
    g_star = np.argmax(R_all, axis=1)
    t_peak = t_all[g_star]

    t_med = np.median(t_peak)
    t_16, t_84 = np.percentile(t_peak, [16, 84])
    t_025, t_975 = np.percentile(t_peak, [2.5, 97.5])

    vals, counts = np.unique(t_peak, return_counts=True)
    probs = counts / t_peak.size

    prob_df = pd.DataFrame({
        "peak_thickness_nm": vals,
        "posterior_probability": probs,
    }).sort_values("peak_thickness_nm")

    # Also attach most likely source/group info for each thickness value
    # If duplicate thicknesses exist, this lists all matching groups.
    source_info = []

    for v in vals:
        matches = metadata[np.isclose(metadata["thickness_nm"], v)]
        source_info.append("; ".join(
            f"{row.dataset}, G{int(row.group_id)}, {row.source_file}"
            for row in matches.itertuples()
        ))

    prob_df["matching_groups"] = source_info

    summary = {
        "peak_t_median_nm": float(t_med),
        "peak_t_ci16_nm": float(t_16),
        "peak_t_ci84_nm": float(t_84),
        "peak_t_ci2p5_nm": float(t_025),
        "peak_t_ci97p5_nm": float(t_975),
        "most_probable_peak_thickness_nm": float(prob_df.loc[
            prob_df["posterior_probability"].idxmax(), "peak_thickness_nm"
        ]),
        "most_probable_peak_probability": float(prob_df["posterior_probability"].max()),
        "n_mc": int(R_all.shape[0]),
    }

    return t_peak, prob_df, summary


def parabolic_peak_distribution(R_all, t_all):
    """
    Optional continuous interpolation estimate.

    This fits a parabola through the argmax group and its two neighboring
    thickness points for each posterior draw.

    Use this only as a secondary estimate. The discrete argmax distribution
    should be the primary reported result because it does not assume smooth
    interpolation between measured groups.
    """
    order = np.argsort(t_all)
    ts = t_all[order]

    tp = np.empty(R_all.shape[0], dtype=float)

    for s in range(R_all.shape[0]):
        r = R_all[s, order]

        i_raw = int(np.argmax(r))

        # If maximum is at endpoint, cannot fit a centered parabola.
        # Fall back to discrete argmax thickness.
        if i_raw == 0 or i_raw == len(ts) - 1:
            tp[s] = ts[i_raw]
            continue

        x3 = ts[i_raw - 1:i_raw + 2]
        y3 = r[i_raw - 1:i_raw + 2]

        try:
            c = np.polyfit(x3, y3, 2)
            a, b, _ = c

            # If parabola opens upward or is almost flat, fallback.
            if a >= 0 or np.isclose(a, 0):
                tp[s] = ts[i_raw]
                continue

            vertex = -b / (2 * a)

            # Accept vertex only if it lies between neighboring measured points.
            if x3[0] <= vertex <= x3[-1]:
                tp[s] = vertex
            else:
                tp[s] = ts[i_raw]

        except Exception:
            tp[s] = ts[i_raw]

    summary = {
        "parabolic_peak_t_median_nm": float(np.median(tp)),
        "parabolic_peak_t_ci16_nm": float(np.percentile(tp, 16)),
        "parabolic_peak_t_ci84_nm": float(np.percentile(tp, 84)),
        "parabolic_peak_t_ci2p5_nm": float(np.percentile(tp, 2.5)),
        "parabolic_peak_t_ci97p5_nm": float(np.percentile(tp, 97.5)),
    }

    return tp, summary


# ============================================================
# MAIN
# ============================================================

datasets = []

for f in sample_files:
    path = Path(f)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {f}\n"
            "Make sure you generated *_posterior_R_samples.npz files "
            "from the Bayesian script."
        )

    datasets.append(load_npz_samples(path))

R_all, t_all, metadata = combine_independent_posteriors(
    datasets,
    n_mc=n_mc,
    random_seed=random_seed,
)

# Primary discrete posterior peak position
t_peak, prob_df, summary = discrete_peak_distribution(R_all, t_all, metadata)

# Optional continuous parabolic interpolation
t_peak_parabola, parabola_summary = parabolic_peak_distribution(R_all, t_all)

summary.update(parabola_summary)

# Save peak samples for checking/replotting
samples_df = pd.DataFrame({
    "t_peak_discrete_nm": t_peak,
    "t_peak_parabolic_nm": t_peak_parabola,
})

samples_df.to_csv(out_samples_csv, index=False)
prob_df.to_csv(out_prob_csv, index=False)
pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)

# Print results
print("\n============================================================")
print("Posterior peak-position distribution: discrete argmax")
print("============================================================")
print(f"Median peak position: {summary['peak_t_median_nm']:.1f} nm")
print(
    f"68% credible interval: "
    f"{summary['peak_t_ci16_nm']:.1f} -- {summary['peak_t_ci84_nm']:.1f} nm"
)
print(
    f"95% credible interval: "
    f"{summary['peak_t_ci2p5_nm']:.1f} -- {summary['peak_t_ci97p5_nm']:.1f} nm"
)
print(
    f"Most probable peak group: "
    f"{summary['most_probable_peak_thickness_nm']:.1f} nm "
    f"(P = {summary['most_probable_peak_probability']:.3f})"
)

print("\nPosterior probability by discrete thickness group:")
for row in prob_df.itertuples():
    print(
        f"  P(peak at {row.peak_thickness_nm:.1f} nm) "
        f"= {row.posterior_probability:.4f}    {row.matching_groups}"
    )

print("\n============================================================")
print("Optional parabolic interpolation estimate")
print("============================================================")
print(
    f"Median peak position: "
    f"{summary['parabolic_peak_t_median_nm']:.1f} nm"
)
print(
    f"68% credible interval: "
    f"{summary['parabolic_peak_t_ci16_nm']:.1f} -- "
    f"{summary['parabolic_peak_t_ci84_nm']:.1f} nm"
)
print(
    f"95% credible interval: "
    f"{summary['parabolic_peak_t_ci2p5_nm']:.1f} -- "
    f"{summary['parabolic_peak_t_ci97p5_nm']:.1f} nm"
)

print("\nSaved files:")
print(f"  {out_summary_csv}")
print(f"  {out_prob_csv}")
print(f"  {out_samples_csv}")
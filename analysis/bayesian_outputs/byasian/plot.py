import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az


# ============================================================
# USER SETTINGS
# ============================================================

# Add/remove datasets here.
# If a listed file does not exist, the script will skip it.
datasets = [
    {
        "csv_path": "4_se_intensity_per_roi.csv",
        "carbon_group_id": 7,
        "label": "Dataset 1",
    },
    {
        "csv_path": "12_se_intensity_per_roi.csv",
        "carbon_group_id": 6,
        "label": "Dataset 2",
    },
    {
        "csv_path": "13_se_intensity_per_roi.csv",
        "carbon_group_id": 5,
        "label": "Dataset 3",
    },
]

# Column names from your ROI CSV
intensity_col = "mean_0_255"
group_col = "group_id"
thickness_col = "thickness"

# Bayesian sampling settings
# These are safer for Windows / Spyder / IPython.
draws = 2000
tune = 1000
chains = 2
cores = 1   # IMPORTANT: keep this as 1 on Windows/Spyder
target_accept = 0.95
random_seed = 42

# Output files
combined_summary_path = Path("combined_bayesian_normalized_summary.csv")
combined_plot_path = Path("combined_bayesian_normalized_plot.png")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """Clean column names so they are easier to use."""
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def read_roi_csv(csv_path):
    """
    Read CSV or tab-separated ROI table.

    sep=None with engine='python' lets pandas automatically detect
    commas, tabs, or mixed whitespace-style separators.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = clean_columns(df)
    df = df.dropna(axis=1, how="all")
    return df


def posterior_interval(samples):
    """
    Return 95%, 68%, and median interval values:
    2.5%, 16%, 50%, 84%, 97.5%.
    """
    return np.percentile(samples, [2.5, 16, 50, 84, 97.5])


# ============================================================
# MAIN BAYESIAN ANALYSIS FUNCTION
# ============================================================

def run_bayesian_normalization(
    csv_path,
    carbon_group_id,
    label,
    intensity_col="mean_0_255",
    group_col="group_id",
    thickness_col="thickness",
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print("\n" + "=" * 70)
        print(f"Skipping {label}")
        print("=" * 70)
        print(f"File not found: {csv_path}")
        return None, None

    # --------------------------------------------------------
    # Load and clean data
    # --------------------------------------------------------

    df = read_roi_csv(csv_path)

    required = {group_col, intensity_col, thickness_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path}: missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[group_col] = pd.to_numeric(df[group_col], errors="coerce")
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
    df[thickness_col] = pd.to_numeric(df[thickness_col], errors="coerce")

    df = df.dropna(subset=[group_col, intensity_col]).copy()
    df[group_col] = df[group_col].astype(int)

    carbon_df = df[df[group_col] == carbon_group_id].copy()
    mos2_df = df[df[group_col] != carbon_group_id].copy()

    if carbon_df.empty:
        raise ValueError(
            f"{csv_path}: no carbon reference ROIs found for group_id = {carbon_group_id}"
        )

    if mos2_df.empty:
        raise ValueError(
            f"{csv_path}: no MoS2 groups found after removing carbon group."
        )

    group_ids = sorted(mos2_df[group_col].unique())
    n_groups = len(group_ids)

    group_to_idx = {gid: i for i, gid in enumerate(group_ids)}
    g_idx = mos2_df[group_col].map(group_to_idx).to_numpy(dtype=int)

    y_mos2 = mos2_df[intensity_col].to_numpy(dtype=float)
    y_carbon = carbon_df[intensity_col].to_numpy(dtype=float)

    thickness_by_group = (
        mos2_df.groupby(group_col)[thickness_col]
        .mean()
        .reindex(group_ids)
        .to_numpy()
    )

    n_roi_by_group = (
        mos2_df.groupby(group_col)[intensity_col]
        .count()
        .reindex(group_ids)
        .to_numpy()
    )

    raw_mean_by_group = (
        mos2_df.groupby(group_col)[intensity_col]
        .mean()
        .reindex(group_ids)
        .to_numpy()
    )

    raw_std_by_group = (
        mos2_df.groupby(group_col)[intensity_col]
        .std(ddof=1)
        .reindex(group_ids)
        .to_numpy()
    )

    all_y = df[intensity_col].to_numpy(dtype=float)
    data_sd = float(np.std(all_y, ddof=1))

    if not np.isfinite(data_sd) or data_sd <= 0:
        data_sd = 10.0

    initial_carbon_mean = float(np.mean(y_carbon))
    initial_mos2_means = raw_mean_by_group

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"File: {csv_path}")
    print(f"Carbon group: {carbon_group_id}")
    print(f"Carbon ROI count: {len(carbon_df)}")
    print("MoS2 ROI count by group:")
    print(pd.Series(n_roi_by_group, index=group_ids))

    if np.any(n_roi_by_group < 2):
        print(
            "\nWARNING: Some MoS2 groups have only one ROI. "
            "Their Bayesian intervals will depend strongly on the shared ROI scatter."
        )

    # --------------------------------------------------------
    # Bayesian model
    # --------------------------------------------------------

    with pm.Model() as model:

        # Latent true carbon reference intensity
        mu_carbon = pm.Normal(
            "mu_carbon",
            mu=initial_carbon_mean,
            sigma=5 * data_sd,
        )

        # Latent true mean intensity for each MoS2 thickness group
        mu_mos2 = pm.Normal(
            "mu_mos2",
            mu=initial_mos2_means,
            sigma=5 * data_sd,
            shape=n_groups,
        )

        # ROI-to-ROI scatter.
        # This is not pixel noise; this represents ROI-level variation.
        sigma_carbon_roi = pm.HalfNormal(
            "sigma_carbon_roi",
            sigma=data_sd,
        )

        sigma_mos2_roi = pm.HalfNormal(
            "sigma_mos2_roi",
            sigma=data_sd,
        )

        # Student-t robustness parameter.
        # nu > 2 keeps variance finite.
        nu_minus_two = pm.Exponential(
            "nu_minus_two",
            lam=1 / 30,
        )

        nu = pm.Deterministic(
            "nu",
            nu_minus_two + 2,
        )

        # Carbon ROI observations
        pm.StudentT(
            "carbon_obs",
            nu=nu,
            mu=mu_carbon,
            sigma=sigma_carbon_roi,
            observed=y_carbon,
        )

        # MoS2 ROI observations
        pm.StudentT(
            "mos2_obs",
            nu=nu,
            mu=mu_mos2[g_idx],
            sigma=sigma_mos2_roi,
            observed=y_mos2,
        )

        # Posterior normalized intensity.
        # Carbon reference is shared, so its uncertainty is naturally shared.
        normalized_intensity = pm.Deterministic(
            "normalized_intensity",
            mu_mos2 / mu_carbon,
        )

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,  # IMPORTANT for Windows / Spyder / IPython
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # --------------------------------------------------------
    # Posterior summary
    # --------------------------------------------------------

    ratio_samples = trace.posterior["normalized_intensity"].values.reshape(-1, n_groups)
    mu_mos2_samples = trace.posterior["mu_mos2"].values.reshape(-1, n_groups)
    mu_carbon_samples = trace.posterior["mu_carbon"].values.reshape(-1)

    carbon_ci = posterior_interval(mu_carbon_samples)

    rows = []

    for i, gid in enumerate(group_ids):
        r = ratio_samples[:, i]
        mu = mu_mos2_samples[:, i]

        r_ci = posterior_interval(r)
        mu_ci = posterior_interval(mu)

        rows.append({
            "dataset": label,
            "source_file": csv_path.name,
            "group_id": int(gid),
            "carbon_group_id": int(carbon_group_id),
            "thickness_nm": thickness_by_group[i],
            "n_roi": int(n_roi_by_group[i]),

            "raw_roi_mean_0_255": raw_mean_by_group[i],
            "raw_roi_std_0_255": raw_std_by_group[i],

            "bayes_mu_mos2_median_0_255": mu_ci[2],
            "bayes_mu_mos2_ci16_0_255": mu_ci[1],
            "bayes_mu_mos2_ci84_0_255": mu_ci[3],
            "bayes_mu_mos2_ci2p5_0_255": mu_ci[0],
            "bayes_mu_mos2_ci97p5_0_255": mu_ci[4],

            "carbon_mu_median_0_255": carbon_ci[2],
            "carbon_mu_ci16_0_255": carbon_ci[1],
            "carbon_mu_ci84_0_255": carbon_ci[3],
            "carbon_mu_ci2p5_0_255": carbon_ci[0],
            "carbon_mu_ci97p5_0_255": carbon_ci[4],

            "normalized_median": r_ci[2],
            "normalized_ci16": r_ci[1],
            "normalized_ci84": r_ci[3],
            "normalized_ci2p5": r_ci[0],
            "normalized_ci97p5": r_ci[4],

            "normalized_err_lower_68": r_ci[2] - r_ci[1],
            "normalized_err_upper_68": r_ci[3] - r_ci[2],
            "normalized_err_lower_95": r_ci[2] - r_ci[0],
            "normalized_err_upper_95": r_ci[4] - r_ci[2],
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("thickness_nm").reset_index(drop=True)

    # --------------------------------------------------------
    # Save individual summary and diagnostics
    # --------------------------------------------------------

    summary_path = csv_path.with_name(
        csv_path.stem + "_bayesian_normalized_summary.csv"
    )

    diagnostics_path = csv_path.with_name(
        csv_path.stem + "_bayesian_diagnostics.csv"
    )

    plot_path = csv_path.with_name(
        csv_path.stem + "_bayesian_normalized_plot.png"
    )

    summary_df.to_csv(summary_path, index=False)

    diagnostics = az.summary(
        trace,
        var_names=[
            "mu_carbon",
            "mu_mos2",
            "sigma_carbon_roi",
            "sigma_mos2_roi",
            "nu",
            "normalized_intensity",
        ],
    )

    diagnostics.to_csv(diagnostics_path)

    # --------------------------------------------------------
    # Individual plot
    # --------------------------------------------------------

    plt.figure(figsize=(6.4, 4.4))

    x = summary_df["thickness_nm"].to_numpy()
    y = summary_df["normalized_median"].to_numpy()
    yerr_lower = summary_df["normalized_err_lower_68"].to_numpy()
    yerr_upper = summary_df["normalized_err_upper_68"].to_numpy()

    plt.errorbar(
        x,
        y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=7,
        capsize=4,
        linewidth=1.5,
        label="Bayesian median ± 68% credible interval",
    )

    # Plot individual ROI means normalized by posterior median carbon
    carbon_median = carbon_ci[2]

    mos2_plot = mos2_df.copy()
    mos2_plot["normalized_raw_roi"] = mos2_plot[intensity_col] / carbon_median

    plt.scatter(
        mos2_plot[thickness_col],
        mos2_plot["normalized_raw_roi"],
        marker="x",
        alpha=0.35,
        label="Individual ROI means / carbon median",
    )

    for _, row in summary_df.iterrows():
        if np.isfinite(row["thickness_nm"]) and np.isfinite(row["normalized_median"]):
            plt.annotate(
                f"G{int(row['group_id'])}",
                (row["thickness_nm"], row["normalized_median"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

    plt.xlabel("EELS thickness (nm)", fontsize=12)
    plt.ylabel("Normalized SE grayscale intensity", fontsize=12)
    plt.title(f"{label}: Bayesian-normalized SE intensity", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved diagnostics: {diagnostics_path}")
    print(f"Saved plot: {plot_path}")

    return summary_df, trace


# ============================================================
# COMBINED PLOT FUNCTION
# ============================================================

def make_combined_plot(combined_df, combined_plot_path):
    plt.figure(figsize=(7.2, 5.0))

    for label, sub in combined_df.groupby("dataset"):
        sub = sub.sort_values("thickness_nm")

        x = sub["thickness_nm"].to_numpy()
        y = sub["normalized_median"].to_numpy()
        yerr_lower = sub["normalized_err_lower_68"].to_numpy()
        yerr_upper = sub["normalized_err_upper_68"].to_numpy()

        plt.errorbar(
            x,
            y,
            yerr=[yerr_lower, yerr_upper],
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.5,
            label=f"{label}: median ± 68% CI",
        )

        for _, row in sub.iterrows():
            if np.isfinite(row["thickness_nm"]) and np.isfinite(row["normalized_median"]):
                plt.annotate(
                    f"{label.replace('Dataset ', 'D')}-G{int(row['group_id'])}",
                    (row["thickness_nm"], row["normalized_median"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    plt.xlabel("EELS thickness (nm)", fontsize=12)
    plt.ylabel("Normalized SE grayscale intensity", fontsize=12)
    plt.title("Bayesian-normalized SE intensity vs. EELS thickness", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved combined plot: {combined_plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    all_summaries = []
    all_traces = {}

    for item in datasets:
        summary, trace = run_bayesian_normalization(
            csv_path=item["csv_path"],
            carbon_group_id=item["carbon_group_id"],
            label=item["label"],
            intensity_col=intensity_col,
            group_col=group_col,
            thickness_col=thickness_col,
        )

        if summary is not None:
            all_summaries.append(summary)
            all_traces[item["label"]] = trace

    if len(all_summaries) == 0:
        print("\nNo datasets were processed. Check your file names and paths.")
        return

    combined_df = pd.concat(all_summaries, ignore_index=True)
    combined_df = combined_df.sort_values(
        ["dataset", "thickness_nm"]
    ).reset_index(drop=True)

    combined_df.to_csv(combined_summary_path, index=False)

    print(f"\nSaved combined summary: {combined_summary_path}")

    make_combined_plot(combined_df, combined_plot_path)

    print("\nFinal combined Bayesian result:")
    print(
        combined_df[
            [
                "dataset",
                "source_file",
                "group_id",
                "carbon_group_id",
                "thickness_nm",
                "n_roi",
                "raw_roi_mean_0_255",
                "raw_roi_std_0_255",
                "normalized_median",
                "normalized_ci16",
                "normalized_ci84",
                "normalized_ci2p5",
                "normalized_ci97p5",
            ]
        ]
    )


# ============================================================
# WINDOWS / SPYDER SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()
    main()
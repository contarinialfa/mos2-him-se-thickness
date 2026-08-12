import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
INPUT_CSV = Path("combined_bayesian_normalized_summary.csv")
OUTPUT_PLOT = Path("recreated_combined_plot.png")

def plot_from_summary(csv_path: Path, plot_path: Path):
    """
    Reads the Bayesian summary CSV and recreates the combined errorbar plot.
    """
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the file is in the same directory as this script.")
        return

    # Load the data
    df = pd.read_csv(csv_path)
    
    # Create the figure
    # You can easily adjust the figure size or DPI here
    plt.figure(figsize=(7.2, 5.0))

    # Iterate through each dataset to plot them as distinct series
    for label, sub in df.groupby("dataset"):
        # Ensure data is plotted in order of thickness to prevent crossing lines
        sub = sub.sort_values("thickness_nm")

        # Extract values for plotting
        x = sub["thickness_nm"].to_numpy()
        y = sub["normalized_median"].to_numpy()
        
        # Asymmetric errors calculated from the credible intervals
        yerr_lower = sub["normalized_err_lower_68"].to_numpy()
        yerr_upper = sub["normalized_err_upper_68"].to_numpy()

        # Plot the error bars and central median points
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

        # Add text annotations for the group IDs next to each point
        for _, row in sub.iterrows():
            if np.isfinite(row["thickness_nm"]) and np.isfinite(row["normalized_median"]):
                # Shorten "Dataset X" to "DX" for cleaner labels
                short_label = str(label).replace('Dataset ', 'D')
                
                plt.annotate(
                    f"{short_label}-G{int(row['group_id'])}",
                    (row["thickness_nm"], row["normalized_median"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    # Styling and formatting
    plt.xlabel("EELS thickness (nm)", fontsize=12)
    plt.ylabel("Normalized SE grayscale intensity", fontsize=12)
    plt.title("Bayesian-normalized SE intensity vs. EELS thickness", fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Successfully saved plot to {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_from_summary(INPUT_CSV, OUTPUT_PLOT)
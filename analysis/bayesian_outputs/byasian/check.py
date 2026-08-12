import pandas as pd
from pathlib import Path

files = [
    "3_se_intensity_per_roi.csv",
    "12_se_intensity_per_roi.csv",
    "13_se_intensity_per_roi.csv",
    # add backside-blocked files here
]

rows = []

for f in files:
    path = Path(f)
    if not path.exists():
        print(f"Skipping missing file: {f}")
        continue

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "max_0_255" not in df.columns or "min_0_255" not in df.columns:
        raise ValueError(f"{f} missing min_0_255 or max_0_255 columns")

    max_val = df["max_0_255"].max()
    min_val = df["min_0_255"].min()

    n_upper = (df["max_0_255"] >= 255).sum()
    n_lower = (df["min_0_255"] <= 0).sum()

    rows.append({
        "file": f,
        "n_roi": len(df),
        "minimum_roi_min_0_255": min_val,
        "maximum_roi_max_0_255": max_val,
        "n_rois_touching_0": int(n_lower),
        "n_rois_touching_255": int(n_upper),
    })

qc = pd.DataFrame(rows)
qc.to_csv("roi_saturation_qc_summary.csv", index=False)

print(qc)

if (qc["n_rois_touching_255"].sum() == 0) and (qc["n_rois_touching_0"].sum() == 0):
    print("\nPASS: No analyzed ROI reaches 0 or 255 based on ROI extrema.")
else:
    print("\nWARNING: At least one ROI touches 0 or 255. Inspect histograms or exclude/reprocess those ROIs.")
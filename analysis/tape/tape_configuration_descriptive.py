"""Descriptive audit of the none/one-/two-tape normalized data.

This intentionally does not infer an SE3 fraction.  Each configuration was
normalized to its own carbon reference, so the resulting ratios cannot be
converted into a total SE3 bound without a calibrated interception/yield/
collection model.  The script only reports transparent matched-point changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUTS = {
    "none": HERE / "paper2" / "none_summary.csv",
    "one_tape": HERE / "paper2" / "1_tape_summary.csv",
    "two_tapes": HERE / "paper2" / "2_tape_summary.csv",
}
OUTPUT_CSV = HERE / "tape_configuration_matched_ratios.csv"
OUTPUT_JSON = HERE / "tape_configuration_descriptive_results.json"


def load_configuration(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["configuration"] = label
    frame["thickness_key"] = frame["thickness_nm"].round(3)
    return frame


def describe(values):
    values = np.asarray(values, float)
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main():
    frames = {
        label: load_configuration(path, label)
        for label, path in INPUTS.items()
    }

    base_columns = [
        "dataset",
        "thickness_key",
        "thickness_nm",
        "normalized_median",
        "bayes_mu_mos2_median_0_255",
        "carbon_mu_median_0_255",
    ]
    base = frames["none"][base_columns].rename(
        columns={
            "thickness_nm": "thickness_nm_none",
            "normalized_median": "normalized_none",
            "bayes_mu_mos2_median_0_255": "mos2_raw_none",
            "carbon_mu_median_0_255": "carbon_raw_none",
        }
    )

    matched = base
    for label in ("one_tape", "two_tapes"):
        current = frames[label][base_columns].rename(
            columns={
                "thickness_nm": f"thickness_nm_{label}",
                "normalized_median": f"normalized_{label}",
                "bayes_mu_mos2_median_0_255": f"mos2_raw_{label}",
                "carbon_mu_median_0_255": f"carbon_raw_{label}",
            }
        )
        matched = matched.merge(
            current,
            on=["dataset", "thickness_key"],
            how="inner",
            validate="one_to_one",
        )

    for label in ("one_tape", "two_tapes"):
        matched[f"normalized_ratio_{label}_over_none"] = (
            matched[f"normalized_{label}"] / matched["normalized_none"]
        )
        matched[f"mos2_raw_ratio_{label}_over_none"] = (
            matched[f"mos2_raw_{label}"] / matched["mos2_raw_none"]
        )
        matched[f"carbon_raw_ratio_{label}_over_none"] = (
            matched[f"carbon_raw_{label}"] / matched["carbon_raw_none"]
        )

    matched = matched.sort_values(
        ["dataset", "thickness_key"]
    ).reset_index(drop=True)
    matched.to_csv(OUTPUT_CSV, index=False)

    result = {
        "description": (
            "Descriptive matched-point changes only; not an SE3-fraction "
            "calibration because every configuration has its own carbon "
            "denominator."
        ),
        "matched_points": int(len(matched)),
        "normalized_ratios": {
            label: describe(
                matched[f"normalized_ratio_{label}_over_none"]
            )
            for label in ("one_tape", "two_tapes")
        },
        "raw_mos2_ratios": {
            label: describe(
                matched[f"mos2_raw_ratio_{label}_over_none"]
            )
            for label in ("one_tape", "two_tapes")
        },
        "carbon_reference_ratios_by_dataset": {},
    }

    for dataset, subset in matched.groupby("dataset"):
        result["carbon_reference_ratios_by_dataset"][dataset] = {
            label: float(
                subset[f"carbon_raw_ratio_{label}_over_none"].iloc[0]
            )
            for label in ("one_tape", "two_tapes")
        }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

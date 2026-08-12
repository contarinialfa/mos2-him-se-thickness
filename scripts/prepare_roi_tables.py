"""Build the final analysis tables from the raw ROI exports.

The preparation rules in ``config/roi_preparation.json`` document which
source regions were retained, how groups were renumbered, which EELS
thickness belongs to each group, and where carbon-reference ROIs were
appended. The script also writes a row-level provenance table and compares
each result with the preserved final table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "roi_preparation.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "prepared_roi_measurements"
ROUND_TO_DECIMALS = 9


def load_export(path: Path) -> pd.DataFrame:
    """Load and validate a raw per-ROI export."""
    frame = pd.read_csv(path)
    required = {
        "region_id",
        "group_id",
        "area_px",
        "mean_0_1",
        "median_0_1",
        "std_0_1",
        "min_0_1",
        "max_0_1",
        "mean_0_255",
        "median_0_255",
        "min_0_255",
        "max_0_255",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def select_part(
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one configured set of ROIs and record its provenance."""
    input_path = REPOSITORY_ROOT / rule["input"]
    source = load_export(input_path)

    if "region_ids" in rule:
        requested_ids = [int(value) for value in rule["region_ids"]]
        selected = source[source["region_id"].isin(requested_ids)].copy()
        selected = selected.set_index("region_id").loc[requested_ids].reset_index()
        if len(selected) != len(requested_ids):
            raise ValueError(f"Not every requested region exists in {input_path}.")
    elif "source_group_id" in rule:
        source_group = int(rule["source_group_id"])
        selected = source[source["group_id"] == source_group].copy()
    else:
        raise ValueError(
            f"A preparation rule for {input_path} needs region_ids or source_group_id."
        )

    source_region_ids = selected["region_id"].astype(int).to_numpy()
    source_group_ids = selected["group_id"].astype(int).to_numpy()
    selected["group_id"] = int(rule["new_group_id"])
    thickness = rule.get("thickness_nm")
    selected["thickness"] = np.nan if thickness is None else float(thickness)
    if "area_um2" not in selected.columns:
        selected["area_um2"] = np.nan

    provenance = pd.DataFrame(
        {
            "input": rule["input"],
            "source_region_id": source_region_ids,
            "source_group_id": source_group_ids,
            "new_group_id": int(rule["new_group_id"]),
            "thickness_nm": np.nan if thickness is None else float(thickness),
        }
    )
    return selected, provenance


def build_dataset(
    dataset: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble and renumber all configured parts for one dataset."""
    parts = []
    provenance_parts = []
    for rule in dataset["parts"]:
        part, provenance = select_part(rule)
        parts.append(part)
        provenance_parts.append(provenance)

    combined = pd.concat(parts, ignore_index=True)
    combined["region_id"] = np.arange(1, len(combined) + 1)
    provenance = pd.concat(provenance_parts, ignore_index=True)
    provenance.insert(0, "output_region_id", combined["region_id"])

    float_columns = (
        "mean_0_1",
        "median_0_1",
        "std_0_1",
        "min_0_1",
        "max_0_1",
    )
    combined.loc[:, float_columns] = combined.loc[:, float_columns].round(
        ROUND_TO_DECIMALS
    )
    column_order = [
        "region_id",
        "group_id",
        "area_px",
        "mean_0_1",
        "median_0_1",
        "std_0_1",
        "min_0_1",
        "max_0_1",
        "mean_0_255",
        "median_0_255",
        "min_0_255",
        "max_0_255",
        "thickness",
        "area_um2",
    ]
    return combined[column_order], provenance


def assert_matches_reference(
    generated: pd.DataFrame,
    reference_path: Path,
) -> None:
    """Confirm that a generated table matches the preserved analysis input."""
    reference = pd.read_csv(reference_path)
    pd.testing.assert_frame_equal(
        generated.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=5e-9,
    )


def run_condition(
    condition: str,
    config: dict[str, Any],
    output_root: Path,
    check_only: bool,
) -> None:
    """Build or validate every dataset for one condition."""
    for dataset in config[condition]:
        generated, provenance = build_dataset(dataset)
        reference_path = REPOSITORY_ROOT / dataset["reference"]
        assert_matches_reference(generated, reference_path)
        print(f"PASS: {dataset['name']} matches {reference_path}")

        if check_only:
            continue
        output_path = output_root / dataset["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(output_path, index=False)
        provenance.to_csv(
            output_path.with_name(f"{output_path.stem}_provenance.csv"),
            index=False,
        )
        print(f"Saved prepared table: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("transmission", "backside_blocked", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all rules without writing regenerated tables.",
    )
    return parser


def main() -> None:
    """Load preparation rules and build or validate the requested datasets."""
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    conditions = (
        ("transmission", "backside_blocked")
        if args.condition == "all"
        else (args.condition,)
    )
    for condition in conditions:
        run_condition(
            condition,
            config=config,
            output_root=args.output_dir.resolve(),
            check_only=args.check_only,
        )


if __name__ == "__main__":
    main()

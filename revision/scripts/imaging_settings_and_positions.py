"""Extract acquisition settings and stage positions of the analysed HIM frames.

Summarizes the ORION XML metadata (TIFF tag 65000, read via zeiss_metadata.py) for:
  * main text, Methods: line-scan acquisition settings (2048 x 2048 pixels, 100 us dwell,
    two-line averaging, +500 V ET Faraday-cage grid bias, +100 V nominal sample bias);
  * SI Sec. SI, "Imaging positions": stage translations between flakes (zero beam shift,
    image shift <= 1 um), in-plane separations of 0.10-0.22 mm between the three Fig. 3
    flakes, working-distance range (<= 0.20 mm), and 39 min acquisition span at constant
    detector settings.

Two modes:
  1. default: reads the 16 analysed raw TIFF frames stored in the repository (three Fig. 3
     transmission frames and two backside-blocked control frames in data/raw_images/, plus
     11 analysed frames of the second copper-tape series in data/tape_series/), and writes
     frames.csv and summary.txt. Also verifies consistency with the 33-frame table in
     revision/outputs/analysed_frames_metadata/frames.csv if present.
  2. --from-table CSV: evaluates the same summary from a pre-extracted per-frame table
     (such as revision/outputs/analysed_frames_metadata/frames.csv, which covers all 33
     analysed frames across the transmission, blocked-control, and both copper-tape runs).

Usage (from the repository root):
    python -B revision/scripts/imaging_settings_and_positions.py
    python -B revision/scripts/imaging_settings_and_positions.py \
        --from-table revision/outputs/analysed_frames_metadata/frames.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from zeiss_metadata import read_fields  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.csv"
RECORD_33 = ROOT / "revision" / "outputs" / "analysed_frames_metadata" / "frames.csv"
OUT_FRAMES = ROOT / "revision" / "outputs" / "imaging_settings_and_positions"
OUT_TABLE = ROOT / "revision" / "outputs" / "imaging_settings_and_positions_from_table"

# Analysed frames of the second copper-tape series stored in data/tape_series/ (11 of 17),
# in acquisition order: (path, flake).
TAPE_RUN2_FRAMES = (
    ("data/tape_series/large_tape/1-2tape-90deg.tif", "1"),
    ("data/tape_series/large_tape/2-2tape-90deg.tif", "2"),
    ("data/tape_series/large_tape/3-2tape-90deg.tif", "3"),
    ("data/tape_series/small_tape/1-1tape-90deg.tif", "1"),
    ("data/tape_series/small_tape/3-1tape-90deg.tif", "3"),
    ("data/tape_series/small_tape/2-1tape-90deg.tif", "2"),
    ("data/tape_series/45deg_converter/1-1tape-45deg.tif", "1"),
    ("data/tape_series/backside_blocked/1-1tape-0deg.tif", "1"),
    ("data/tape_series/control/1-0.tif", "1"),
    ("data/tape_series/control/2-0.tif", "2"),
    ("data/tape_series/control/3-0.tif", "3"),
)
DET_DIST_UM = 9000.0  # main text, Methods: ~9 mm from the sample to the centre of the Faraday cage
FIELDS = ("TimeStamp", "StageX", "StageY", "StageZ", "StageTilt", "StageRotate",
          "WorkingDistance", "Fov", "PanX", "PanY", "McXShift", "McYShift", "ScanRotate",
          "ETContrast", "ETBrightness", "ETGridVoltage", "Detector.Scintillator",
          "SampleBiasVoltage", "FloodGunMode", "DwellTime")
SETTINGS_FIELDS = ("ImageWidth", "ImageHeight", "AveragingMode", "NumAverages")
ROLES = ["Fig. 3 transmission", "blocked control", "tape run 1", "tape run 2"]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def num(text: str) -> float:
    return float(text)


def read_repository_frames() -> list[dict]:
    """Read XML fields of the analysed frames stored in the repository (mode 1)."""
    frames: list[dict] = []
    with open(MANIFEST, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = row["path"]
            if not p.startswith("data/raw_images/"):
                continue
            role = "Fig. 3 transmission" if "/transmission/" in p else "blocked control"
            flake = re.search(r"flake_(\d)", p).group(1)
            frames.append({"role": role, "flake": flake, "file": p})
    for p, flake in TAPE_RUN2_FRAMES:
        frames.append({"role": "tape run 2", "flake": flake, "file": p})
    rows = []
    for fr in frames:
        path = ROOT / fr["file"]
        if not path.exists():
            raise SystemExit(f"frame not found: {fr['file']}")
        f = read_fields(path)
        rec = {"role": fr["role"], "flake": fr["flake"], "file": fr["file"]}
        for k in FIELDS + SETTINGS_FIELDS:
            rec[k] = f[k][0] if k in f else ""
            rec[f"{k}_units"] = f[k][1] if k in f else ""
        rows.append(rec)
    return rows


def read_table(path: Path) -> list[dict]:
    """Read per-frame table with columns role, flake, and FIELDS (mode 2)."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [k for k in ("role", "flake") + FIELDS if k not in rows[0]]
    if missing:
        raise SystemExit(f"table {path} lacks columns {missing}")
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-table", type=Path, default=None,
                    help="recompute the summary from this per-frame table instead of reading frames")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out: Path = args.out or (OUT_TABLE if args.from_table else OUT_FRAMES)

    header = [f"imaging_settings_and_positions.py  python {platform.python_version()}  numpy {np.__version__}"]
    if args.from_table is None:
        rows = read_repository_frames()
        have_units = True
        header.append("input: ORION XML (TIFF tag 65000) of the analysed frames held by the repository "
                      "(data/raw_images/, data/tape_series/)")
    else:
        rows = read_table(args.from_table)
        have_units = False
        header.append(f"input: per-frame table {rel(args.from_table)}")
    header.append("")

    out.mkdir(parents=True, exist_ok=True)
    if args.from_table is None:
        cols = ["role", "flake", "file"] + list(FIELDS) + list(SETTINGS_FIELDS)
        with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    lines: list[str] = []
    roles = ROLES
    lines.append(f"Analysed frames: {len(rows)} (" + ", ".join(
        f"{sum(r['role'] == ro for r in rows)} {ro}" for ro in roles) + ")")
    if have_units:
        units = {k: sorted({r[f"{k}_units"] for r in rows}) for k in FIELDS}
        lines.append("Units as recorded: " + "; ".join(f"{k} {units[k]}" for k in FIELDS if k != "TimeStamp"))
    else:
        lines.append("Units as recorded: not stored in the per-frame table (the frame-reading mode prints them)")
    lines.append("")

    # ------------------------------------------------------------ 1. shifts and settings
    def distinct(key, sel):
        return sorted({r[key] for r in rows if sel(r)})

    groups = (("Fig. 3 frames", lambda r: r["role"] == roles[0]),
              ("blocked control", lambda r: r["role"] == roles[1]),
              ("tape runs", lambda r: r["role"].startswith("tape")),
              ("all analysed", lambda r: True))
    for label, sel in groups:
        pan = [abs(num(r[k])) for r in rows if sel(r) for k in ("PanX", "PanY")]
        if not pan:
            lines.append(f"{label}: no frames")
            continue
        lines.append(f"{label}: McXShift {distinct('McXShift', sel)}, McYShift {distinct('McYShift', sel)}, "
                     f"ScanRotate {distinct('ScanRotate', sel)}, max |PanX|,|PanY| = {max(pan):.2f} um, "
                     f"StageZ {distinct('StageZ', sel)}, StageTilt {distinct('StageTilt', sel)}")
    lines.append("")

    # ------------------------------------------------------------ 2. acquisition settings
    lines.append("Acquisition settings (image size; averaging mode, number of averages; dwell; ET grid; "
                 "sample-bias read-back):")
    have_settings = all(k in rows[0] for k in SETTINGS_FIELDS)
    for label, sel in groups:
        sub = [r for r in rows if sel(r)]
        if not sub:
            lines.append(f"  {label}: no frames")
            continue
        if have_settings:
            size = sorted({f"{r['ImageWidth']} x {r['ImageHeight']}" for r in sub})
            avg = f"{distinct('AveragingMode', sel)}, {distinct('NumAverages', sel)}"
        else:
            size, avg = "not in the table", "not in the table"
        bias = [num(r["SampleBiasVoltage"]) for r in sub]
        lines.append(f"  {label} (n={len(sub)}): {size}; {avg}; DwellTime {distinct('DwellTime', sel)}; "
                     f"ETGridVoltage {distinct('ETGridVoltage', sel)}; "
                     f"SampleBiasVoltage {min(bias):g}-{max(bias):g}")
    lines.append("")

    # ------------------------------------------------------------ 3. Fig. 3 geometry
    fig3 = sorted([r for r in rows if r["role"] == roles[0]], key=lambda r: r["flake"])
    lines.append("Fig. 3 frames:")
    for r in fig3:
        lines.append(f"  flake {r['flake']}: {r['TimeStamp']}; StageX {r['StageX']}, StageY {r['StageY']} um; "
                     f"WD {r['WorkingDistance']} mm; Fov {r['Fov']} um; Pan ({r['PanX']}, {r['PanY']}) um; "
                     f"tilt {r['StageTilt']} deg; ETContrast {r['ETContrast']}, ETBrightness {r['ETBrightness']}, "
                     f"ETGrid {r['ETGridVoltage']} V, scint {r['Detector.Scintillator']} V, "
                     f"sample bias {r['SampleBiasVoltage']} V, flood {r['FloodGunMode']}, dwell {r['DwellTime']}")
    t = [datetime.strptime(r["TimeStamp"], "%m/%d/%Y %I:%M:%S %p") for r in fig3]
    lines.append(f"  time span of the three frames: {(max(t) - min(t)).total_seconds() / 60:.1f} min; "
                 f"same calendar day: {len({x.date() for x in t}) == 1}")
    wds = [num(r["WorkingDistance"]) for r in fig3]
    wd_range_um = (max(wds) - min(wds)) * 1000
    lines.append(f"  WD range {min(wds):.3f}-{max(wds):.3f} mm = {wd_range_um:.0f} um = "
                 f"{100 * wd_range_um / DET_DIST_UM:.2f}% of 9 mm")
    lines.append("  pairs: stage travel / in-plane on grid / predicted dY*sin(tilt) / measured dWD (um)")
    inplane = []
    for a, b in combinations(fig3, 2):
        dx = num(b["StageX"]) - num(a["StageX"])
        dy = num(b["StageY"]) - num(a["StageY"])
        tilt = math.radians(0.5 * (num(a["StageTilt"]) + num(b["StageTilt"])))
        travel = math.hypot(dx, dy)
        plane = math.hypot(dx, dy * math.cos(tilt))
        inplane.append(plane)
        pred = dy * math.sin(tilt)
        dwd = (num(b["WorkingDistance"]) - num(a["WorkingDistance"])) * 1000
        lines.append(f"    flake {a['flake']}->{b['flake']}: dX {dx:.2f}, dY {dy:.2f}; travel {travel:.1f}; "
                     f"in-plane {plane:.1f}; predicted {pred:.1f}; measured {dwd:.1f}")
    lines.append(f"  in-plane separations {min(inplane):.1f}-{max(inplane):.1f} um; largest = "
                 f"{100 * max(inplane) / 3050:.1f}% of the 3.05 mm grid diameter")
    elev = math.radians(16.0)
    dz = wd_range_um
    lines.append(f"  largest height change {dz:.0f} um seen from 9 mm at 16 deg elevation: view-angle change "
                 f"{math.degrees(dz * math.cos(elev) / DET_DIST_UM):.2f} deg, distance change from the vertical "
                 f"move alone {dz * math.sin(elev):.0f} um ({100 * dz * math.sin(elev) / DET_DIST_UM:.2f}%; "
                 f"any detector azimuth: detector_distance_bound.py)")
    lines.append("")

    # ------------------------------------------------------------ 4. in-image angles
    fovs = sorted({num(r["Fov"]) for r in rows})
    lines.append("Angle spanned by one field as seen from 9 mm (half-field; half-diagonal):")
    for fov in fovs:
        lines.append(f"  Fov {fov:g} um: +/-{math.degrees(math.atan(fov / 2 / DET_DIST_UM)):.3f} deg; "
                     f"+/-{math.degrees(math.atan(fov / math.sqrt(2) / DET_DIST_UM)):.3f} deg")
    fov_fig3 = sorted({num(r["Fov"]) for r in fig3})
    lines.append(f"  Fig. 3 fields: {fov_fig3} um")

    # ------------------------------------------------------------ cross-check with the 33-frame record
    if args.from_table is None:
        lines.append("")
        if RECORD_33.exists():
            with open(RECORD_33, encoding="utf-8") as fh:
                record = {r["TimeStamp"]: r for r in csv.DictReader(fh)}
            matched, differ = 0, []
            for r in rows:
                rr = record.get(r["TimeStamp"])
                if rr is None:
                    differ.append(f"{r['file']}: no row with this time stamp")
                    continue
                matched += 1
                for k in ("role", "flake") + FIELDS:
                    if rr[k] != r[k]:
                        differ.append(f"{r['file']}: {k} {r[k]!r} vs {rr[k]!r}")
            lines.append(f"Record of all analysed frames {rel(RECORD_33)} ({len(record)} rows): "
                         f"{matched} of the {len(rows)} frames read here matched by time stamp; "
                         f"differing fields: {len(differ)}")
            lines += [f"  {d}" for d in differ]
        else:
            lines.append(f"Record of all analysed frames {rel(RECORD_33)} not found; no cross-check")

    with open(out / "summary.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header + lines) + "\n")
    print("\n".join(header + lines))


if __name__ == "__main__":
    main()

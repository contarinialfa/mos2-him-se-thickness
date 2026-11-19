"""Stage positions of the analysed HIM frames and radial positions of the flakes on the TEM grid.

Computes the stage geometry and grid-centre distances reported in SI Sec. SI ("Imaging positions"):
  - stage translations between flakes (zero beam shift, image shift <= 1 um);
  - radial distance of the three Fig. 3 flakes from the centre of the 3.05 mm TEM grid (<= 0.5 mm);
  - working-distance (WD) variation with tilted-stage Y translations, and pairs_fig3.csv
    (used by detector_distance_bound.py).

Steps:
 1. Read the ORION XML metadata (TIFF tag 65000, zeiss_metadata.py) of every frame in the
    analysed sessions and tabulate beam/image shift, scan rotation, tilt, stage coordinates,
    and detector settings.
 2. Compute stage spreads, pairwise separations, and WD changes. Because the stage is tilted
    (~42.9 deg) while the custom holder keeps the grid normal to the beam, a stage-Y translation
    shifts the grid in-plane by dY*cos(tilt) and vertically by dY*sin(tilt).
 3. Register the analysed flakes onto the stereo-microscope optical location map of the grid
    via a similarity fit between the numbered box centres and the survey-session HIM stage
    coordinates, and measure distances from the circle-fitted grid centre.

Note: Running this script from scratch requires the full HIM survey and optical-map archive
(--archive DIR), which is not stored in this repository. The resulting derived tables
(analysed_fields_on_grid.csv, pairs_fig3.csv, grid_position_check.txt) are provided in
revision/outputs/grid_positions/.

Usage (from the repository root):
    python -B revision/scripts/grid_positions.py --archive DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from zeiss_metadata import read_fields  # noqa: E402

MODEL_MANIFEST = ROOT / "data" / "manifest.csv"
OUT_DEFAULT = ROOT / "revision" / "outputs" / "grid_positions"
ARCHIVE_MANIFEST = "_index/manifest_images.csv"
GRID_D_UM, DET_D_UM, ELEV_DEG = 3050.0, 9000.0, 16.0  # main text, Methods

ANALYSED_SESSIONS = (
    "2025-08-06_back-blocked_top-cover-removed_PAPER-blocked-control",
    "2025-08-14_transmission_PAPER-main-dataset",
    "2025-09-30_tape-run1",
    "2026-07-25_tape-run2",
    "2026-07-27_tape-run2",
)
# Analysed frames of the two copper-tape series: (series, role in the tape analysis, flake,
# file name in the archive).
ANALYSED_TAPE_FRAMES = (
    ("tape_run1", "repeat", "2", "153908_20250930_loc12-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg2coppertape.tif"),
    ("tape_run1", "main large", "3", "155719_20250930_loc13-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg2coppertape.tif"),
    ("tape_run1", "main large", "1", "162917_20250930_loc4_topcoverremoverd_100Vbias_noflood_20umaper_90deg2coppertape.tif"),
    ("tape_run1", "main large", "2", "164626_20250930_loc12-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg2coppertape1.tif"),
    ("tape_run1", "main small", "2", "171828_20250930_loc12-1_topcoverremoverd_100Vbias_noflood_20umaper_90degcoppertape.tif"),
    ("tape_run1", "main small", "3", "173919_20250930_loc13-1_topcoverremoverd_100Vbias_noflood_20umaper_90degcoppertape.tif"),
    ("tape_run1", "main small", "1", "175629_20250930_loc4_topcoverremoverd_100Vbias_noflood_20umaper_90degcoppertape.tif"),
    ("tape_run1", "main none", "2", "182434_20250930_loc12-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg.tif"),
    ("tape_run1", "main none", "3", "184730_20250930_loc13-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg.tif"),
    ("tape_run1", "main none", "1", "190758_20250930_loc4_topcoverremoverd_100Vbias_noflood_20umaper_90deg.tif"),
    ("tape_run1", "repeat", "3", "192500_20250930_loc13-1_topcoverremoverd_100Vbias_noflood_20umaper_90deg1.tif"),
    ("tape_run2", "large", "1", "105240_1-2tape-90deg.tif"),
    ("tape_run2", "large", "2", "111154_2-2tape-90deg.tif"),
    ("tape_run2", "large", "3", "112906_3-2tape-90deg.tif"),
    ("tape_run2", "small", "1", "120339_1-1tape-90deg.tif"),
    ("tape_run2", "small_dup", "2", "123021_2-1tape-90deg.tif"),
    ("tape_run2", "small", "3", "124925_3-1tape-90deg.tif"),
    ("tape_run2", "small", "2", "130840_2-1tape-90deg-1.tif"),
    ("tape_run2", "conv45", "1", "133632_1-1tape-45deg.tif"),
    ("tape_run2", "blockedA", "1", "142729_1-1tape-0deg.tif"),
    ("tape_run2", "blockedA", "2", "144440_2-1tape-0deg.tif"),
    ("tape_run2", "blockedA", "3", "150134_3-1tape-0deg.tif"),
    ("tape_run2", "blockedB", "1", "075307_1-1tape-0deg.tif"),
    ("tape_run2", "blockedB", "2", "081032_2-1tape-0deg.tif"),
    ("tape_run2", "blockedB", "3", "082658_3-1tape-0deg.tif"),
    ("tape_run2", "control", "1", "085426_1-0.tif"),
    ("tape_run2", "control", "2", "091004_2-0.tif"),
    ("tape_run2", "control", "3", "092531_3-0.tif"),
)
# Frames of the mounting in which locations 1-17 were first surveyed (same holder seating:
# the 30-um loc4 field sits within 2 um in all of them). Used to check the optical numbering.
SURVEY_SESSIONS = (
    "2025-07-28_flake-survey",
    "2025-07-29_flake-survey",
    "2025-07-30_flake-survey",
    "2025-07-31_beam-energy-series_20-25-30kV",
    "2025-08-01_beam-energy-and-sample-bias-series",
)
# The WD fit over all frames on this grid leaves out the stage-tilt series recorded on another
# sample on this date.
OTHER_SAMPLE_SESSION_PREFIX = "2025-07-18_"
OPTICAL_MAP = "03_related_non-HIM/optical-stereo_location-maps/MoS2-E10/location.tif"
OPTICAL_FULL_GRID = "03_related_non-HIM/optical-stereo_location-maps/2025-07-11/large 1.tif"
# Box numbers as read by eye from the annotated map (full-resolution pixel positions of the
# box centres, read from a zoomed crop). Only used to give each detected box its number.
BOX_READING = {
    1: (1861, 1387), 2: (1967, 1436), 3: (1818, 1176), 4: (1602, 1783), 5: (1588, 1882),
    6: (1944, 1186), 7: (1721, 1871), 8: (1834, 1871), 9: (1946, 1871), 10: (1381, 1745),
    11: (1381, 1856), 12: (1492, 1856), 13: (1376, 1972), 14: (1494, 1972), 15: (1602, 1974),
    16: (1718, 1975), 17: (1834, 1975),
}
FLAKE_LOC = {1: "loc4", 2: "loc12", 3: "loc13"}  # paper flake -> location number (image archive)
# Field-name check against the ORION user interface: a screenshot of the Columns panel taken
# while the frame below was set up (same FOV 850 um, WD 9.952 mm, stigmation -1.85/1.65).
# Values read by eye from the panel ("Tuning" and "Shift And Tilt" sections):
GUI_FRAME = "01_HIM_raw/1_sample-survey/2025-06-12_flake-survey_25kV/171048_1-2 100um.tif"
GUI_READING = {"Beam Shift X [urad]": 0.00, "Beam Shift Y [urad]": 0.00,
               "Beam Tilt X [urad]": -25.00, "Beam Tilt Y [urad]": 1162.00,
               "Pan X": -0.14, "Pan Y": -0.02}

POSITION_RE = re.compile(r"(shift|pan|deflect|offset|rotat|stage|tilt|location|crossover)", re.I)
AUX_FIELDS = (
    "TimeStamp", "AccelerationVoltage", "Fov", "GFIS.FieldOfView", "ScalingX", "ScalingY",
    "ImageWidth", "ImageHeight", "WorkingDistance", "GFIS.WorkingDistance", "DwellTime",
    "AveragingMode", "NumAverages", "LineRetrace", "FrameRetrace", "SpotNumber", "ApertureSize",
    "GFIS.ApertureIndex", "GunPressure", "DetectorName", "ETGridVoltage", "ETContrast",
    "ETBrightness", "Detector.Scintillator", "SampleBiasVoltage", "FloodGunMode", "ColumnMode",
)

NOT_DEPOSITED = (
    "grid_positions.py needs the HIM image archive (manifest, raw frames of the analysed and survey "
    "sessions, optical location maps of the grid), which is not part of this deposit. The derived "
    "records it wrote are in revision/outputs/grid_positions/ (analysed_fields_on_grid.csv, "
    "pairs_fig3.csv, grid_position_check.txt). To rerun, pass --archive DIR (the archive root) and "
    "--out <new folder>.")


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def fmt(v, nd=1):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


# ----------------------------------------------------------------------- optical helpers
def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(float).mean(axis=2)


def otsu(g: np.ndarray) -> float:
    h, _ = np.histogram(g, bins=256, range=(0, 256))
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(np.argmax(between))


def bar_pitch(g: np.ndarray, axis: str, band: tuple[int, int], p0: float) -> tuple[float, float, int]:
    """Mesh pitch (px) from the bar minima of a band-averaged profile; slope of a line fit."""
    prof = g[band[0]:band[1], :].mean(axis=0) if axis == "x" else g[:, band[0]:band[1]].mean(axis=1)
    sm = ndi.gaussian_filter1d(prof, p0 / 8)
    mins = [i for i in range(1, len(sm) - 1) if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]]
    keep: list[int] = []
    for i in sorted(mins, key=lambda k: sm[k]):
        if all(abs(i - k) > 0.6 * p0 for k in keep):
            keep.append(i)
    keep_a = np.array(sorted(keep))
    keep_a = keep_a[sm[keep_a] < np.percentile(sm, 35)]
    sub = []
    for i in keep_a:
        a, b, c = sm[i - 1], sm[i], sm[i + 1]
        den = a - 2 * b + c
        sub.append(i + (0.5 * (a - c) / den if den != 0 else 0.0))
    sub_a = np.array(sub)
    d = np.diff(sub_a)
    ok = np.abs(d - np.median(d)) < 0.25 * np.median(d)
    idx = np.concatenate([[0], np.cumsum(np.round(d / np.median(d)))])
    good = np.concatenate([[True], ok]) | np.concatenate([ok, [True]])
    slope, icpt = np.polyfit(idx[good], sub_a[good], 1)
    res = sub_a[good] - (slope * idx[good] + icpt)
    return float(slope), float(np.std(res)), int(good.sum())


def kasa(x: np.ndarray, y: np.ndarray):
    m = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(m, x ** 2 + y ** 2, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx ** 2 + cy ** 2)
    return cx, cy, r, np.hypot(x - cx, y - cy) - r


def raycast(frac: np.ndarray, c0, r_min: float, r_max: float, thr_in: float, thr_out: float):
    """Rays from c0: first radius where the dark fraction exceeds thr_in (edge of the meshed
    area, i.e. inner edge of the solid rim), then where it drops below thr_out (outer rim)."""
    h, w = frac.shape
    rr = np.arange(r_min, r_max, 1.0)
    pin, pout = [], []
    for ang in np.radians(np.arange(0, 360, 0.5)):
        xs = c0[0] + rr * np.cos(ang)
        ys = c0[1] + rr * np.sin(ang)
        inside = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
        if not inside.any():
            continue
        v = ndi.map_coordinates(frac, [ys, xs], order=1, mode="constant", cval=np.nan)
        hit = np.where(inside & (v > thr_in))[0]
        if hit.size and inside[:hit[0] + 1].all():
            k = hit[0]
            pin.append((xs[k], ys[k]))
            after = np.where(inside[k:] & (v[k:] < thr_out))[0]
            if after.size and inside[k:k + after[0] + 1].all():
                pout.append((xs[k + after[0]], ys[k + after[0]]))
    return np.array(pin), np.array(pout)


def fit_circle_robust(pts: np.ndarray):
    cx, cy, r, res = kasa(pts[:, 0], pts[:, 1])
    ok = np.abs(res) < 2.5 * np.std(res)
    cx, cy, r, res = kasa(pts[ok, 0], pts[ok, 1])
    return cx, cy, r, float(np.sqrt(np.mean(res ** 2))), int(ok.sum()), len(pts)


def grid_geometry(path: Path, p_guess: float) -> dict:
    g = gray(path)
    h, w = g.shape
    px, sx, nx = bar_pitch(g, "x", (h // 2 - 400, h // 2 + 400), p_guess)
    py, sy, ny = bar_pitch(g, "y", (w // 2 - 400, w // 2 + 400), p_guess)
    pitch = 0.5 * (px + py)
    frac = ndi.uniform_filter((g < otsu(g)).astype(float), size=int(round(1.5 * pitch)))
    c0 = (w / 2, h / 2)
    for _ in range(3):
        pin, pout = raycast(frac, c0, 3 * pitch, 2.0 * max(h, w), 0.85, 0.3)
        cx, cy, r_in, rms_in, n_in, n_in_all = fit_circle_robust(pin)
        c0 = (cx, cy)
    out = {"file": path.name, "width": w, "height": h, "pitch_x_px": px, "pitch_y_px": py,
           "pitch_px": pitch, "pitch_line_fit_sd_px": max(sx, sy), "n_bars": nx + ny,
           "centre_x_px": cx, "centre_y_px": cy, "mesh_edge_r_px": r_in, "mesh_edge_rms_px": rms_in,
           "mesh_edge_points": f"{n_in}/{n_in_all}"}
    if len(pout) > 50:
        ox, oy, r_out, rms_out, n_out, n_out_all = fit_circle_robust(pout)
        out.update({"outer_centre_x_px": ox, "outer_centre_y_px": oy, "outer_r_px": r_out,
                    "outer_rms_px": rms_out, "outer_points": f"{n_out}/{n_out_all}"})
    return out


def red_boxes(path: Path) -> list[dict]:
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    red = (a[..., 0] > 150) & (a[..., 1] < 90) & (a[..., 2] < 90) & (a[..., 0] - a[..., 1] > 100)
    lab, _ = ndi.label(ndi.binary_closing(red, iterations=2), structure=np.ones((3, 3)))
    boxes = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        hgt, wid = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        npx = int((lab[sl] == i).sum())
        if 55 <= wid <= 130 and 55 <= hgt <= 130 and npx / (wid * hgt) < 0.35:
            boxes.append({"x_px": sl[1].start + wid / 2, "y_px": sl[0].start + hgt / 2,
                          "w_px": wid, "h_px": hgt})
    return boxes


def similarity_fit(src: np.ndarray, dst: np.ndarray, reflect: bool):
    """Least-squares dst ~ s R src + t (Umeyama); optional reflection of src first."""
    s_ = src * np.array([1.0, -1.0]) if reflect else src
    ms, md = s_.mean(0), dst.mean(0)
    a, b = s_ - ms, dst - md
    u, sing, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(sing) @ d) / (a ** 2).sum(1).mean())
    t = md - scale * rot @ ms
    pred = (scale * (rot @ s_.T)).T + t
    res = np.hypot(*(pred - dst).T)
    angle = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return {"scale": scale, "rot": rot, "t": t, "angle_deg": angle, "res": res, "reflect": reflect}


# ----------------------------------------------------------------------------- main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", type=Path, default=None,
                    help="root of the HIM image archive (not part of this deposit)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    # ---------------------------------------------------------- private inputs
    if args.archive is None:
        raise SystemExit(NOT_DEPOSITED)
    archive: Path = args.archive
    manifest_path = archive / ARCHIVE_MANIFEST
    required = [manifest_path] + [archive / p for p in (OPTICAL_MAP, OPTICAL_FULL_GRID, GUI_FRAME)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing archive files: " + "; ".join(missing) + "\n" + NOT_DEPOSITED)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = [f"grid_positions.py  python {platform.python_version()}  numpy {np.__version__}", ""]

    def note_input(path: Path, role: str, label: str, text: bool = False) -> str:
        return sha256(path)

    # ---------------------------------------------------------- constants
    grid_d_um, det_d_um, elev_deg = GRID_D_UM, DET_D_UM, ELEV_DEG
    consts = [f"grid diameter {grid_d_um:.0f} um (main text, Methods: TEM grid of 3.05 mm diameter)",
              f"sample-to-Faraday-cage distance {det_d_um:.0f} um (main text, Methods: ~9 mm)",
              f"detector elevation {elev_deg:.0f} deg (main text, Methods: ~16 deg above the horizontal stage plane)"]
    write_text(out / "manuscript_constants.txt", "\n".join(consts) + "\n")

    # ---------------------------------------------------------- frames and roles
    manifest = list(csv.DictReader(open(manifest_path, encoding="utf-8-sig")))
    note_input(manifest_path, "image archive manifest", f"<archive>/{ARCHIVE_MANIFEST}", text=True)
    by_sha = {r["sha256"]: r for r in manifest if r["kind"] == "HIM_raw"}
    by_path = {r["archive_path"].replace("\\", "/"): r for r in manifest}
    roles: dict[str, str] = {}
    # Fig. 3 transmission and blocked control: the images this repository analyses
    note_input(MODEL_MANIFEST, "repository manifest (image SHA-256)", "data/manifest.csv", text=True)
    for row in csv.DictReader(open(MODEL_MANIFEST, encoding="utf-8")):
        p = row["path"]
        if p.startswith("data/raw_images/"):
            arch = by_sha[row["sha256"]]
            kind = "Fig3_transmission" if "/transmission/" in p else "blocked_control"
            roles[arch["archive_path"].replace("\\", "/")] = f"{kind} ({Path(p).stem})"
    for tag, role, flake, name in ANALYSED_TAPE_FRAMES:
        hits = [r for r in manifest if r["kind"] == "HIM_raw" and r["session"] in ANALYSED_SESSIONS
                and r["archive_path"].replace("\\", "/").rsplit("/", 1)[-1] == name]
        if len(hits) != 1:
            raise SystemExit(f"{len(hits)} manifest rows for analysed tape frame {name}")
        roles[hits[0]["archive_path"].replace("\\", "/")] = f"{tag}:{role} (flake {flake})"
    frames = [r for r in manifest if r["kind"] == "HIM_raw" and r["session"] in ANALYSED_SESSIONS]
    frames.sort(key=lambda r: r["acquired"])
    log.append(f"frames in analysed sessions: {len(frames)}; analysed (role assigned): "
               f"{sum(1 for r in frames if r['archive_path'].replace(chr(92), '/') in roles)}")

    # ---------------------------------------------------------- 1. XML dump
    meta: dict[str, dict] = {}
    inventory: dict[str, dict] = {}
    for r in frames:
        ap_ = r["archive_path"].replace("\\", "/")
        path = archive / ap_
        if not path.exists():
            raise SystemExit(f"missing archive frame: {ap_}\n{NOT_DEPOSITED}")
        digest = sha256(path)
        if digest != r["sha256"]:
            raise SystemExit(f"file SHA-256 differs from manifest: {path}")
        fields = read_fields(path)
        meta[ap_] = fields
        for k, (v, u) in fields.items():
            inv = inventory.setdefault(k, {"field": k, "units": set(), "n_frames": 0})
            inv["units"].add(u)
            inv["n_frames"] += 1
    inv_rows = []
    for k, inv in inventory.items():
        sel = "position" if POSITION_RE.search(k) else ("auxiliary" if k in AUX_FIELDS else "")
        inv_rows.append({"field": k, "units": " | ".join(sorted(inv["units"])),
                         "n_frames": inv["n_frames"], "selected_as": sel})
    write_csv(out / "xml_field_inventory.csv", inv_rows)
    pos_fields = [r["field"] for r in inv_rows if r["selected_as"] == "position"]
    aux_fields = [f for f in AUX_FIELDS if f in inventory]
    units = {r["field"]: r["units"] for r in inv_rows}

    wide = []
    for r in frames:
        ap_ = r["archive_path"].replace("\\", "/")
        f = meta[ap_]
        row = {"role": roles.get(ap_, "not analysed"), "session": r["session"], "config": r["config"],
               "acquired": r["acquired"], "flake": r["flake"], "loc": r["loc"],
               "original_name": r["original_name"], "archive_path": ap_}
        for k in pos_fields + aux_fields:
            row[f"{k} [{units[k]}]"] = f.get(k, ("", ""))[0]
        wide.append(row)
    write_csv(out / "xml_position_and_settings_by_frame.csv", wide)

    analysed = [w for w in wide if w["role"] != "not analysed"]
    groups = {"Fig. 3 transmission": [w for w in analysed if w["role"].startswith("Fig3")],
              "blocked control": [w for w in analysed if w["role"].startswith("blocked")],
              "tape run 1": [w for w in analysed if w["role"].startswith("tape_run1")],
              "tape run 2": [w for w in analysed if w["role"].startswith("tape_run2")],
              "all analysed": analysed}
    counts = ", ".join(f"{len(v)} {k}" for k, v in groups.items() if k != "all analysed")
    summ_lines = [f"Position-related XML fields over the analysed frames ({len(analysed)} frames: {counts})", ""]
    for k in pos_fields + aux_fields:
        col = f"{k} [{units[k]}]"
        summ_lines.append(f"{k} [{units[k]}]")
        for gname, rows in groups.items():
            vals = [w[col] for w in rows]
            distinct = sorted(set(vals), key=lambda s: (fnum(s) if not math.isnan(fnum(s)) else 0, s))
            shown = ", ".join(distinct[:8]) + (f", ... ({len(distinct)} distinct)" if len(distinct) > 8 else "")
            nums = [fnum(v) for v in vals if not math.isnan(fnum(v))]
            rng = f"  [min {min(nums):g}, max {max(nums):g}]" if nums and len(distinct) > 1 else ""
            summ_lines.append(f"    {gname:20s} n={len(rows):2d}: {shown}{rng}")
    gui_path = archive / GUI_FRAME
    if sha256(gui_path) != by_path[GUI_FRAME]["sha256"]:
        raise SystemExit(f"SHA-256 of {GUI_FRAME} differs from the manifest")
    gf = read_fields(gui_path)
    summ_lines += ["", "Field-name check against the ORION user interface (screenshot of the Columns panel, frame "
                   f"{GUI_FRAME}): panel values read by eye vs XML of that frame",
                   "    " + "; ".join(f"{k} {v:g}" for k, v in GUI_READING.items()),
                   "    XML: " + "; ".join(f"{k} {gf[k][0]} {gf[k][1]}" for k in
                                         ("McXShift", "McYShift", "McXTilt", "McYTilt", "PanX", "PanY")),
                   "    -> McX/YShift = 'Beam Shift', McX/YTilt = 'Beam Tilt' (the XML numbers equal the panel "
                   "microradians x 1e-6, i.e. they are in radians despite the 'Micro Radians' label); "
                   "PanX/PanY = 'Pan' (image shift, microns). The panel Pan differs from the saved frame "
                   "(screenshot taken at another moment)."]
    write_text(out / "position_fields_summary.txt", "\n".join(summ_lines) + "\n")

    # ---------------------------------------------------------- 2. stage geometry
    def geom(ap_: str) -> dict:
        f = meta[ap_]
        for key, unit in (("StageX", "um"), ("StageY", "um"), ("StageZ", "mm"), ("WorkingDistance", "mm"),
                          ("StageTilt", "Degrees"), ("PanX", "Microns"), ("PanY", "Microns")):
            if f[key][1] != unit:
                raise SystemExit(f"unexpected units for {key}: {f[key][1]!r} in {ap_}")
        x, y = float(f["StageX"][0]), float(f["StageY"][0])
        tilt = float(f["StageTilt"][0])
        return {"X_um": x, "Y_um": y, "Z_mm": float(f["StageZ"][0]), "tilt_deg": tilt,
                "rotate_deg": float(f["StageRotate"][0]), "WD_mm": float(f["WorkingDistance"][0]),
                "PanX_um": float(f["PanX"][0]), "PanY_um": float(f["PanY"][0]),
                "Fov_um": float(f["Fov"][0]),
                "u_inplane_um": x, "v_inplane_um": y * math.cos(math.radians(tilt)),
                "h_height_um": y * math.sin(math.radians(tilt))}

    geo_rows = []
    for w in analysed:
        g = geom(w["archive_path"])
        geo_rows.append({"role": w["role"], "session": w["session"], "config": w["config"],
                         "mounting": f"{w['session'][:10]} {w['config']}".strip(),
                         "acquired": w["acquired"], "flake": w["flake"], "loc": w["loc"],
                         "original_name": w["original_name"], **g})
    write_csv(out / "stage_geometry_analysed_frames.csv", geo_rows)

    def pair(a: dict, b: dict) -> dict:
        dx, dy = b["X_um"] - a["X_um"], b["Y_um"] - a["Y_um"]
        dz = 1000 * (b["Z_mm"] - a["Z_mm"])
        th = math.radians(0.5 * (a["tilt_deg"] + b["tilt_deg"]))
        dwd = 1000 * (b["WD_mm"] - a["WD_mm"])
        return {"a": f"{a['acquired'][11:]} {a['flake']} {a['loc']}", "b": f"{b['acquired'][11:]} {b['flake']} {b['loc']}",
                "dX_um": dx, "dY_um": dy, "dZ_um": dz, "stage_travel_3d_um": math.sqrt(dx * dx + dy * dy + dz * dz),
                "inplane_on_grid_um": math.hypot(dx, dy * math.cos(th)),
                "height_pred_dYsin_um": dy * math.sin(th), "dWD_um": dwd,
                "dWD_minus_pred_um": dwd - dy * math.sin(th),
                "dWD_over_9mm_pct": 100 * abs(dwd) / det_d_um,
                "detector_view_change_deg": math.degrees(abs(dwd) * math.cos(math.radians(elev_deg)) / det_d_um),
                "detector_view_change_bound_deg": math.degrees(abs(dwd) / det_d_um),
                "detector_distance_change_um": abs(dwd) * math.sin(math.radians(elev_deg))}

    fig3 = sorted([g for g in geo_rows if g["role"].startswith("Fig3")], key=lambda g: g["flake"])
    fig3_pairs = [pair(a, b) for a, b in itertools.combinations(fig3, 2)]
    write_csv(out / "pairs_fig3.csv", fig3_pairs)

    within = []
    for mnt in sorted({g["mounting"] for g in geo_rows}):
        rows = [g for g in geo_rows if g["mounting"] == mnt]
        for a, b in itertools.combinations(rows, 2):
            within.append({"mounting": mnt, **pair(a, b)})
    write_csv(out / "pairs_within_mounting.csv", within)

    across = []
    for fl in ("flake1", "flake2", "flake3"):
        rows = [g for g in geo_rows if g["flake"] == fl]
        for a, b in itertools.combinations(rows, 2):
            if a["mounting"] != b["mounting"]:
                across.append({"flake": fl, "mounting_a": a["mounting"], "mounting_b": b["mounting"], **pair(a, b)})
    write_csv(out / "pairs_same_flake_across_mountings.csv", across)

    # WD against the stage-Y height term, with one intercept per mounting
    def wd_fit(rows: list[dict], label: str) -> list[str]:
        mounts = sorted({r["mounting"] for r in rows})
        cnt = {m: sum(1 for r in rows if r["mounting"] == m) for m in mounts}
        rows = [r for r in rows if cnt[r["mounting"]] >= 2]
        mounts = sorted({r["mounting"] for r in rows})
        A = np.zeros((len(rows), len(mounts) + 2))
        yv = np.zeros(len(rows))
        for i, r in enumerate(rows):
            A[i, mounts.index(r["mounting"])] = 1.0
            A[i, -2] = r["Y_um"] * math.sin(math.radians(r["tilt_deg"])) / 1000.0
            A[i, -1] = r["X_um"] / 1000.0
            yv[i] = r["WD_mm"]
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        res = yv - A @ coef
        dof = len(rows) - A.shape[1]
        s2 = float(res @ res / dof) if dof > 0 else math.nan
        cov = s2 * np.linalg.pinv(A.T @ A)
        return [f"{label}: {len(rows)} frames, {len(mounts)} mountings (one intercept each)",
                f"    WD = a_mounting + bY * (StageY*sin(tilt)) + bX * StageX   (all in mm)",
                f"    bY = {coef[-2]:.3f} +/- {math.sqrt(cov[-2, -2]):.3f}   (1 expected if WD tracks the grid height)",
                f"    bX = {coef[-1]:.3f} +/- {math.sqrt(cov[-1, -1]):.3f}   (0 expected: X is along the tilt axis)",
                f"    residual rms = {1000 * math.sqrt(float(np.mean(res ** 2))):.1f} um, max |res| = "
                f"{1000 * float(np.max(np.abs(res))):.1f} um", ""]

    e10 = []
    for r in manifest:
        if r["kind"] != "HIM_raw" or r["session"][:10] < "2025-07-11" or r["session"].startswith(OTHER_SAMPLE_SESSION_PREFIX):
            continue
        e10.append({"mounting": f"{r['session']} {r['config']}".strip(), "X_um": fnum(r["StageX"]),
                    "Y_um": fnum(r["StageY"]), "WD_mm": fnum(r["WorkingDistance"]), "tilt_deg": fnum(r["StageTilt"])})
    wd_lines = wd_fit(geo_rows, "Analysed frames") + wd_fit(e10, "All raw frames on this grid from the first "
                                                             "survey on (manifest values; tilt series of "
                                                             "another sample excluded)")
    write_text(out / "wd_vs_stage_fit.txt", "\n".join(wd_lines))

    spread = []
    for mnt in sorted({g["mounting"] for g in geo_rows}):
        rows = [g for g in geo_rows if g["mounting"] == mnt]
        xs, ys, wds = [g["X_um"] for g in rows], [g["Y_um"] for g in rows], [g["WD_mm"] for g in rows]
        inpl = [p_["inplane_on_grid_um"] for p_ in within if p_["mounting"] == mnt] or [0.0]
        spread.append({"mounting": mnt, "n_frames": len(rows), "X_range_um": max(xs) - min(xs),
                       "Y_range_um": max(ys) - min(ys), "max_inplane_separation_um": max(inpl),
                       "WD_min_mm": min(wds), "WD_max_mm": max(wds), "WD_range_um": 1000 * (max(wds) - min(wds)),
                       "max_abs_Pan_um": max(max(abs(g["PanX_um"]), abs(g["PanY_um"])) for g in rows)})
    allx = [g["X_um"] for g in geo_rows]
    ally = [g["Y_um"] for g in geo_rows]
    allwd = [g["WD_mm"] for g in geo_rows]
    spread.append({"mounting": "ALL analysed frames (mixes mountings)", "n_frames": len(geo_rows),
                   "X_range_um": max(allx) - min(allx), "Y_range_um": max(ally) - min(ally),
                   "max_inplane_separation_um": max(p_["inplane_on_grid_um"] for p_ in within + across),
                   "WD_min_mm": min(allwd), "WD_max_mm": max(allwd), "WD_range_um": 1000 * (max(allwd) - min(allwd)),
                   "max_abs_Pan_um": max(max(abs(g["PanX_um"]), abs(g["PanY_um"])) for g in geo_rows)})
    write_csv(out / "spread_by_mounting.csv", spread)

    fov_rows = []
    for fov in sorted({g["Fov_um"] for g in geo_rows}):
        half, corner = fov / 2, fov / math.sqrt(2)
        fov_rows.append({"Fov_um": fov, "half_width_um": half,
                         "edge_angle_deg": math.degrees(half / det_d_um),
                         "corner_angle_deg": math.degrees(corner / det_d_um)})
    write_csv(out / "in_image_view_angles.csv", fov_rows)

    # ---------------------------------------------------------- 3. grid position (optical map)
    full_by_path = {r["archive_path"].replace("\\", "/"): r for r in manifest}
    opt_lines = []
    geo_opt = {}
    for rel, guess in ((OPTICAL_FULL_GRID, 68.0), (OPTICAL_MAP, 114.0)):
        path = archive / rel
        digest = note_input(path, "optical grid image", f"<archive>/{rel}")
        if digest != full_by_path[rel]["sha256"]:
            raise SystemExit(f"optical image SHA-256 differs from manifest: {rel}")
        geo_opt[rel] = grid_geometry(path, guess)
        opt_lines.append(f"{rel}")
        opt_lines.append("    " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                              for k, v in geo_opt[rel].items()}))
    full, mapg = geo_opt[OPTICAL_FULL_GRID], geo_opt[OPTICAL_MAP]
    pitch_um = full["pitch_px"] * grid_d_um / (2 * full["outer_r_px"])
    mesh_d_um = 2 * full["mesh_edge_r_px"] * grid_d_um / (2 * full["outer_r_px"])
    um_per_px_map = pitch_um / mapg["pitch_px"]
    opt_lines += ["",
                  f"Full-grid image: outer rim r = {full['outer_r_px']:.1f} px = {full['outer_r_px'] / full['pitch_px']:.2f} "
                  f"pitches; meshed-area edge r = {full['mesh_edge_r_px']:.1f} px = "
                  f"{full['mesh_edge_r_px'] / full['pitch_px']:.2f} pitches",
                  f"Location map: meshed-area edge r = {mapg['mesh_edge_r_px'] / mapg['pitch_px']:.2f} pitches"
                  + (f"; outer rim r = {mapg['outer_r_px'] / mapg['pitch_px']:.2f} pitches (corner arcs only)"
                     if "outer_r_px" in mapg else ""),
                  f"Scale from the nominal {grid_d_um / 1000:.2f} mm outer diameter: mesh pitch = {pitch_um:.2f} um; "
                  f"meshed area diameter = {mesh_d_um:.0f} um; location map {um_per_px_map:.4f} um/px", ""]

    boxes = red_boxes(archive / OPTICAL_MAP)
    named = []
    for n, (ex, ey) in BOX_READING.items():
        cand = [b for b in boxes if math.hypot(b["x_px"] - ex, b["y_px"] - ey) < 45]
        if len(cand) == 1:
            b = cand[0]
            dxp, dyp = b["x_px"] - mapg["centre_x_px"], b["y_px"] - mapg["centre_y_px"]
            named.append({"box": n, **b, "dx_from_centre_um": dxp * um_per_px_map,
                          "dy_from_centre_um": dyp * um_per_px_map,
                          "r_from_centre_um": math.hypot(dxp, dyp) * um_per_px_map})
        else:
            opt_lines.append(f"box {n}: {len(cand)} single-box candidates (outline touches a neighbour); not used")
    write_csv(out / "optical_boxes.csv", named)

    # HIM stage positions of the same locations in the survey mounting
    survey = {}
    for r in manifest:
        if r["kind"] == "HIM_raw" and r["session"] in SURVEY_SESSIONS and r["loc"]:
            m = re.fullmatch(r"loc(\d+)(?:-1)?", r["loc"])
            if not m:
                continue
            n, sub = int(m.group(1)), r["loc"].endswith("-1")
            # prefer the overview of the location (no sub-index); loc14 exists only as loc14-1
            key = (n, sub)
            if key not in survey or fnum(r["Fov"]) > fnum(survey[key]["Fov"]):
                survey[key] = r
    pairs_opt, labels = [], []
    for b in named:
        r = survey.get((b["box"], False)) or survey.get((b["box"], True))
        if r is None:
            continue
        th = math.radians(fnum(r["StageTilt"]))
        pairs_opt.append(((b["x_px"], b["y_px"]), (fnum(r["StageX"]), fnum(r["StageY"]) * math.cos(th))))
        labels.append((b["box"], r["loc"], r["acquired"], r["Fov"]))
    src = np.array([p_[0] for p_ in pairs_opt])
    dst = np.array([p_[1] for p_ in pairs_opt])
    fits = [similarity_fit(src, dst, refl) for refl in (False, True)]
    best = min(fits, key=lambda f_: float(np.sqrt(np.mean(f_["res"] ** 2))))
    sim_rows = []
    for (box, loc, acq, fov), res, (s_xy, d_uv) in zip(labels, best["res"], pairs_opt):
        sim_rows.append({"box": box, "loc_frame": loc, "acquired": acq, "Fov_um": fov,
                         "opt_x_px": s_xy[0], "opt_y_px": s_xy[1], "stage_u_um": d_uv[0], "stage_v_um": d_uv[1],
                         "residual_um": float(res)})
    write_csv(out / "optical_vs_stage_similarity.csv", sim_rows)
    rms = float(np.sqrt(np.mean(best["res"] ** 2)))
    opt_lines += [f"Similarity fit, optical box centres -> HIM in-plane stage coordinates (StageX, StageY*cos(tilt)), "
                  f"{len(src)} locations of the survey mounting:",
                  f"    reflection {'yes' if best['reflect'] else 'no'}; scale {best['scale']:.4f} um/px "
                  f"(pitch-based {um_per_px_map:.4f}; ratio {best['scale'] / um_per_px_map:.3f}); "
                  f"rotation {best['angle_deg']:.1f} deg; residual rms {rms:.1f} um, max {float(best['res'].max()):.1f} um "
                  f"(mesh pitch {pitch_um:.1f} um)",
                  f"    other handedness: rms {float(np.sqrt(np.mean([f_ for f_ in fits if f_ is not best][0]['res'] ** 2))):.1f} um",
                  ""]

    # analysed fields: survey-mounting frames of the analysed locations, mapped onto the map
    inv_rot = best["rot"].T
    fields_rows = []
    for fl, loc in FLAKE_LOC.items():
        cands = [r for r in manifest if r["kind"] == "HIM_raw" and r["session"] in SURVEY_SESSIONS
                 and r["loc"] in (loc, f"{loc}-1") and r["flake"] == f"flake{fl}"]
        for r in cands:
            th = math.radians(fnum(r["StageTilt"]))
            uv = np.array([fnum(r["StageX"]), fnum(r["StageY"]) * math.cos(th)])
            s_ = inv_rot @ (uv - best["t"]) / best["scale"]
            xy = s_ * np.array([1.0, -1.0]) if best["reflect"] else s_
            rr = math.hypot(xy[0] - mapg["centre_x_px"], xy[1] - mapg["centre_y_px"]) * um_per_px_map
            fields_rows.append({"flake": fl, "loc": r["loc"], "acquired": r["acquired"], "Fov_um": r["Fov"],
                                "map_x_px": float(xy[0]), "map_y_px": float(xy[1]), "r_from_grid_centre_um": rr})
    write_csv(out / "analysed_fields_on_grid.csv", fields_rows)
    for fl, loc in FLAKE_LOC.items():
        b = [x for x in named if x["box"] == int(loc[3:])]
        fr = [x["r_from_grid_centre_um"] for x in fields_rows if x["flake"] == fl]
        opt_lines.append(f"Flake {fl} ({loc}): box centre {fmt(b[0]['r_from_centre_um'], 0) if b else 'n/a'} um from the grid "
                         f"centre; its HIM fields {fmt(min(fr), 0)}-{fmt(max(fr), 0)} um")
    r_all = [x["r_from_grid_centre_um"] for x in fields_rows] + \
            [x["r_from_centre_um"] for x in named if x["box"] in (4, 12, 13)]
    opt_lines += [f"All analysed flakes: {fmt(min(r_all), 0)}-{fmt(max(r_all), 0)} um from the grid centre = "
                  f"{100 * max(r_all) / (grid_d_um / 2):.0f}% of the grid radius ({grid_d_um / 2:.0f} um), "
                  f"{100 * max(r_all) / (mesh_d_um / 2):.0f}% of the meshed-area radius ({mesh_d_um / 2:.0f} um)"]
    write_text(out / "grid_position_check.txt", "\n".join(opt_lines) + "\n")

    # overlay
    im = Image.open(archive / OPTICAL_MAP).convert("RGB")
    dr = ImageDraw.Draw(im)
    cx, cy = mapg["centre_x_px"], mapg["centre_y_px"]
    for rad_um, col in ((500, (255, 200, 0)), (1000, (255, 200, 0)), (mesh_d_um / 2, (0, 200, 255))):
        rp = rad_um / um_per_px_map
        dr.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline=col, width=6)
    dr.line([cx - 40, cy, cx + 40, cy], fill=(255, 200, 0), width=6)
    dr.line([cx, cy - 40, cx, cy + 40], fill=(255, 200, 0), width=6)
    for x in fields_rows:
        dr.ellipse([x["map_x_px"] - 14, x["map_y_px"] - 14, x["map_x_px"] + 14, x["map_y_px"] + 14],
                   outline=(0, 255, 0), width=5)
    im.resize((1200, int(1200 * im.size[1] / im.size[0]))).save(out / "grid_position_overlay.png")

    # ---------------------------------------------------------- summary
    pan_all = [max(abs(g["PanX_um"]), abs(g["PanY_um"])) for g in geo_rows]
    sh = [(w[f"McXShift [{units['McXShift']}]"], w[f"McYShift [{units['McYShift']}]"]) for w in analysed]
    rot = sorted({w[f"ScanRotate [{units['ScanRotate']}]"] for w in analysed})
    fig3_wd = [g["WD_mm"] for g in fig3]
    fig3_pan = [max(abs(g["PanX_um"]), abs(g["PanY_um"])) for g in fig3]
    summary = [
        "Stage positions of the analysed frames and grid position of the analysed flakes -- key numbers",
        "",
        f"Analysed frames: {len(geo_rows)} ({counts})",
        f"Image shift (PanX/PanY, um): max |value| = {max(fig3_pan):.2f} um (Fig. 3 frames), "
        f"{max(pan_all):.2f} um (all analysed frames)",
        f"Beam shift (McXShift/McYShift, recorded units '{units['McXShift']}'): values {sorted(set(sh))}",
        f"Scan rotation (ScanRotate): {rot}",
        f"StageZ: {sorted({g['Z_mm'] for g in geo_rows})} mm; StageTilt {min(g['tilt_deg'] for g in geo_rows)}-"
        f"{max(g['tilt_deg'] for g in geo_rows)} deg; StageRotate {sorted({g['rotate_deg'] for g in geo_rows})} deg",
        "",
        "Fig. 3 frames (flake: StageX, StageY um; WD mm; Fov um):",
    ] + [f"    {g['flake']} {g['loc']}: {g['X_um']:.2f}, {g['Y_um']:.2f}; WD {g['WD_mm']:.3f}; Fov {g['Fov_um']:g}"
         for g in fig3] + [
        "Fig. 3 pairs (stage travel / in-plane on grid / dWD / predicted dY*sin(tilt), um):",
    ] + [f"    {p_['a']} -> {p_['b']}: {p_['stage_travel_3d_um']:.1f} / {p_['inplane_on_grid_um']:.1f} / "
         f"{p_['dWD_um']:+.1f} / {p_['height_pred_dYsin_um']:+.1f}" for p_ in fig3_pairs] + [
        f"Fig. 3 WD range: {min(fig3_wd):.3f}-{max(fig3_wd):.3f} mm ({1000 * (max(fig3_wd) - min(fig3_wd)):.0f} um = "
        f"{100 * (max(fig3_wd) - min(fig3_wd)) * 1000 / det_d_um:.1f}% of {det_d_um / 1000:.0f} mm); "
        f"view-angle change at 16 deg elevation {max(p_['detector_view_change_deg'] for p_ in fig3_pairs):.2f} deg, "
        f"distance change {max(p_['detector_distance_change_um'] for p_ in fig3_pairs):.0f} um",
        f"All analysed frames: WD {min(allwd):.3f}-{max(allwd):.3f} mm; within one mounting the largest WD range is "
        f"{max(s['WD_range_um'] for s in spread[:-1]):.0f} um",
        f"Max in-plane separation between analysed flakes within one mounting: "
        f"{max(p_['inplane_on_grid_um'] for p_ in within):.0f} um (grid diameter {grid_d_um:.0f} um)",
        "",
        "In-image view angle (+/- half-field / 9 mm; corners in brackets):",
    ] + [f"    Fov {r['Fov_um']:g} um: +/-{r['edge_angle_deg']:.3f} deg (+/-{r['corner_angle_deg']:.3f})" for r in fov_rows] + [
        "",
    ] + wd_lines + [ln for ln in opt_lines if ln.startswith(("Scale from", "Similarity", "    reflection",
                                                              "    other", "Flake", "All analysed"))]
    write_text(out / "summary.txt", "\n".join(summary) + "\n")
    log.append(f"outputs: {sorted(p_.name for p_ in out.iterdir()) + ['run_log.txt']}")
    write_text(out / "run_log.txt", "\n".join(log) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

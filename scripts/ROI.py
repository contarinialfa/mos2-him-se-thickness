# ------------------------------------------------------------
# SE ROI measurement tool with GROUPS + SAVE/LOAD + ALIGN
# - Draw polygons; assign each to a group as you go
# - Save polygon vertices to JSON for later reuse
# - Load saved polygons on a new image, interactively ALIGN (translate/rotate/scale)
#   and re-measure intensities at the same locations
# - Per-ROI stats + per-GROUP summaries:
#     pooled_mean/std (pixel-wise), simple_mean/std of ROI means (unweighted),
#     area-weighted mean of ROI means
# - Color-coded overlay by group
# ------------------------------------------------------------

# --- Force a Qt interactive backend BEFORE importing pyplot ---
import matplotlib
try:
    matplotlib.use('QtAgg')   # Matplotlib ≥ 3.8
except Exception:
    matplotlib.use('Qt5Agg')  # Older Matplotlib

import os
import json
import math
import time
import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from PIL import Image
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

plt.rcParams['figure.raise_window'] = True

# ------------------ USER SETTINGS ------------------
IMAGE_PATH = r"2-1tape-90deg.tif"  # <-- set me

# Analyze top part only (crop out bottom legend).
# Your target: 2048×2047 (W×H) → keep rows 0..2046
CROP_ROWS_END = 2047
CROP_COLS_END = None

# Optional metadata
THICKNESS_VALUES = []     # e.g., [10, 20, 40]  (same order you draw ROIs)
PIXEL_SIZE_NM = None      # e.g., 5.0 (nm/pixel) to compute area in μm²

# Optional background subtraction (applies to ROI means before grouping)
SUBTRACT_BACKGROUND = False
BACKGROUND_INDEX = 0

# Output folder (None → same as image)
OUTPUT_DIR = None

# Saved ROI geometry file (JSON). If None, auto = <imagebase>_roi_polygons.json
ROIS_JSON_PATH = None
# ---------------------------------------------------

GROUP_COLORS = [
    'tab:blue','tab:orange','tab:green','tab:red','tab:purple',
    'tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan'
]

def color_for_group(g):
    if g is None: return 'yellow'
    i = int(g) % len(GROUP_COLORS)
    return GROUP_COLORS[i]

# --------- Robust grayscale loader with cropping ---------

def _to_float01(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        a = arr.astype(np.float32)
        mx = np.nanmax(a) if a.size else 1.0
        if mx > 1.0:
            a = a / (255.0 if mx <= 255.0 else 65535.0)
        return np.clip(a, 0.0, 1.0).astype(np.float32)
    imax = np.iinfo(arr.dtype).max
    return (arr.astype(np.float32) / float(imax)).clip(0.0, 1.0)

def _rgba_over_background(rgba, bg=(1.0, 1.0, 1.0)):
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    rgb = _to_float01(rgb)
    a = _to_float01(a)
    bg = np.array(bg, dtype=np.float32).reshape((1, 1, 3))
    return rgb * a + bg * (1.0 - a)

def _rgb_to_gray(rgb):
    rgb = _to_float01(rgb)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)

def load_image_grayscale(path, crop_rows_end=None, crop_cols_end=None):
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2:
        gray = _to_float01(arr)
    elif arr.ndim == 3:
        ch = arr.shape[2]
        if ch == 1:   gray = _to_float01(arr[..., 0])
        elif ch == 2: gray = _to_float01(arr[..., 0])                 # LA
        elif ch == 3: gray = _rgb_to_gray(arr)                         # RGB
        elif ch == 4: gray = _rgb_to_gray(_rgba_over_background(arr))  # RGBA
        else:          raise ValueError(f"Unexpected channels: {ch}")
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    if crop_rows_end is not None:
        gray = gray[:crop_rows_end, :]
    if crop_cols_end is not None:
        gray = gray[:, :crop_cols_end]
    return gray

# ---------------- ROI collection (group-aware) ----------------
class ROICollector:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.rois = []          # list of dicts: {"verts": Nx2, "mask": HxW bool, "group": int, "line": artist, "label": artist}
        self.finished = False
        self.active_group = 1   # <-- current group while drawing
        self.cid_key = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        self._instr = None
        self.poly = None
        self._make_instructions()
        self._new_selector()    # start first selector

    def _title_text(self):
        return f"SEM ROI — Draw polygons | Active Group: {self.active_group} | [ / ] to change, 0-9 to jump, u undo, q finish"

    def _make_instructions(self):
        if self._instr is not None:
            try: self._instr.remove()
            except Exception: pass
        instr = ("Draw polygon ROIs: left-click to add points, Enter/double-click to close.\n"
                 "Press [ / ] to change ACTIVE GROUP (or 0–9 to jump). Press 'u' to undo last, 'q' to finish.")
        self._instr = self.ax.text(0.01, 0.01, instr, transform=self.ax.transAxes,
                                   fontsize=9, va='bottom', ha='left',
                                   bbox=dict(boxstyle="round", alpha=0.3))
        self.ax.set_title(self._title_text())
        self.ax.figure.canvas.draw_idle()

    def _new_selector(self):
        if self.poly is not None:
            try:
                self.poly.disconnect_events()
            except Exception:
                pass
        # Create a fresh selector (no blit → more reliable in Spyder)
        self.poly = PolygonSelector(self.ax, self.onselect, useblit=False)

    def onselect(self, verts):
        # Called when a polygon is completed (Enter or double-click)
        verts = np.asarray(verts, dtype=float)
        if not np.allclose(verts[0], verts[-1]):
            verts = np.vstack([verts, verts[0]])
        h, w = self.image.shape[:2]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        path = Path(verts)
        mask = path.contains_points(coords).reshape(h, w)

        g = int(self.active_group)
        line = self.ax.plot(verts[:, 0], verts[:, 1],
                            linewidth=1.8, color=color_for_group(g))[0]
        ys, xs = np.nonzero(mask)
        label_artist = None
        if xs.size:
            cx, cy = xs.mean(), ys.mean()
            label_artist = self.ax.text(cx, cy, f"{len(self.rois)+1} (G{g})",
                                        ha='center', va='center', fontsize=10,
                                        color=color_for_group(g))
        self.ax.figure.canvas.draw_idle()

        self.rois.append({"verts": verts.copy(),
                          "mask": mask.copy(),
                          "group": g,
                          "line": line,
                          "label": label_artist})

        # re-arm for next polygon
        self._new_selector()

    def _undo_last(self):
        if not self.rois:
            return
        r = self.rois.pop()
        try:
            if r.get("line") is not None:
                r["line"].remove()
        except Exception:
            pass
        try:
            if r.get("label") is not None:
                r["label"].remove()
        except Exception:
            pass
        # renumber remaining labels
        for idx, roi in enumerate(self.rois, start=1):
            if roi.get("label") is not None:
                roi["label"].set_text(f"{idx} (G{roi['group']})")
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        k = event.key

        # Finish
        if k == 'q':
            self.finished = True
            try:
                if self.poly is not None:
                    self.poly.disconnect_events()
                self.ax.figure.canvas.mpl_disconnect(self.cid_key)
            except Exception:
                pass
            plt.close(self.ax.figure)
            return

        # Undo
        if k == 'u':
            self._undo_last()
            return

        # Change active group with [ / ]
        if k == '[':
            self.active_group = max(0, int(self.active_group) - 1)
            self._make_instructions()
            return
        if k == ']':
            self.active_group = int(self.active_group) + 1
            self._make_instructions()
            return

        # Jump to specific group with numeric keys 0..9
        if k in list('0123456789'):
            self.active_group = int(k)
            self._make_instructions()
            return

# ---------------- SAVE/LOAD ROI GEOMETRY ----------------

def save_rois_json(rois, json_path, image_shape, meta=None):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = {
        "schema": 1,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "image_shape": tuple(map(int, image_shape[:2])),  # (H, W)
        "rois": [
            {"region_id": i+1,
             "group": int(r.get("group", 0)) if r.get("group", None) is not None else None,
             "verts": np.asarray(r["verts"], dtype=float).tolist()}
            for i, r in enumerate(rois)
        ],
        "meta": meta or {}
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_rois_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    rois = []
    for item in payload.get("rois", []):
        rois.append({
            "region_id": int(item.get("region_id", len(rois)+1)),
            "group": item.get("group", None),
            "verts": np.array(item["verts"], dtype=float)
        })
    image_shape = tuple(payload.get("image_shape", (None, None)))
    meta = payload.get("meta", {})
    return rois, image_shape, meta

# ---------------- Measurements & saving ----------------

def measure_rois(image, rois):
    rows = []
    for i, roi in enumerate(rois, start=1):
        m = roi["mask"]
        vals = image[m]
        if vals.size == 0:
            stats = dict(region_id=i, group_id=roi.get("group", None), area_px=0,
                         mean_0_1=np.nan, median_0_1=np.nan,
                         std_0_1=np.nan, min_0_1=np.nan, max_0_1=np.nan)
        else:
            stats = dict(region_id=i, group_id=roi.get("group", None), area_px=int(m.sum()),
                         mean_0_1=float(vals.mean()),
                         median_0_1=float(np.median(vals)),
                         std_0_1=float(vals.std(ddof=1) if vals.size > 1 else 0.0),
                         min_0_1=float(vals.min()),
                         max_0_1=float(vals.max()))
        rows.append(stats)
    df = pd.DataFrame(rows)
    for col in ["mean_0_1", "median_0_1", "min_0_1", "max_0_1"]:
        df[col.replace("_0_1", "_0_255")] = (df[col] * 255.0).round(2)
    return df


def apply_background(df, bg_idx):
    if df.empty or not (0 <= bg_idx < len(df)):
        return df
    bg_mean = df.loc[bg_idx, "mean_0_1"]
    bg_median = df.loc[bg_idx, "median_0_1"]
    df["mean_0_1"] = df["mean_0_1"] - bg_mean
    df["median_0_1"] = df["median_0_1"] - bg_median
    for col in ["mean_0_1", "median_0_1"]:
        df[col.replace("_0_1", "_0_255")] = (df[col] * 255.0).round(2)
    return df


def add_thickness_and_area(df, thickness_values, pixel_size_nm):
    if thickness_values and len(thickness_values) == len(df):
        df["thickness"] = thickness_values
    else:
        df["thickness"] = None
    if isinstance(pixel_size_nm, numbers.Number) and pixel_size_nm > 0:
        area_um2 = df["area_px"] * (pixel_size_nm ** 2) / 1e6
        df["area_um2"] = area_um2.round(6)
    else:
        df["area_um2"] = None
    return df


def pooled_group_stats(df_group):
    """Return pooled mean/std across pixels using ROI means/stds and pixel counts."""
    n = df_group["area_px"].sum()
    if n <= 1:
        mean = float(df_group["mean_0_1"].mean())
        std = float(df_group["std_0_1"].mean())
        return mean, std, n
    mean = float((df_group["mean_0_1"] * df_group["area_px"]).sum() / n)  # area-weighted
    ssw = ((df_group["area_px"] - 1) * (df_group["std_0_1"] ** 2)).fillna(0.0).sum()
    ssm = (df_group["area_px"] * (df_group["mean_0_1"] - mean) ** 2).fillna(0.0).sum()
    var_pooled = float((ssw + ssm) / (n - 1))
    std = np.sqrt(max(var_pooled, 0.0))
    return mean, std, int(n)


def summarize_groups(df):
    if "group_id" not in df.columns:
        return pd.DataFrame()
    groups = []
    for g, dfg in df.groupby("group_id", dropna=False):
        n_rois = int(dfg.shape[0])
        total_area = int(dfg["area_px"].sum())

        # pooled (pixel-wise) mean/std
        pooled_mean_0_1, pooled_std_0_1, Npix = pooled_group_stats(dfg)

        # simple (unweighted) stats of ROI means
        simple_mean_0_1 = float(dfg["mean_0_1"].mean())
        simple_std_0_1  = float(dfg["mean_0_1"].std(ddof=1)) if n_rois > 1 else np.nan

        # area-weighted average of ROI means (equivalent to pooled mean)
        if total_area > 0:
            w_mean_0_1 = float((dfg["mean_0_1"] * dfg["area_px"]).sum() / total_area)
        else:
            w_mean_0_1 = np.nan

        row = dict(
            group_id=g,
            n_rois=n_rois,
            total_area_px=total_area,
            pooled_mean_0_1=pooled_mean_0_1,
            pooled_std_0_1=pooled_std_0_1,
            simple_mean_of_roi_means_0_1=simple_mean_0_1,
            simple_std_of_roi_means_0_1=simple_std_0_1,
            area_weighted_mean_0_1=w_mean_0_1,
        )

        # 0..255 convenience versions
        for k in ["pooled_mean_0_1", "pooled_std_0_1",
                  "simple_mean_of_roi_means_0_1", "simple_std_of_roi_means_0_1",
                  "area_weighted_mean_0_1"]:
            row[k.replace("_0_1", "_0_255")] = None if np.isnan(row[k]) else round(255.0 * row[k], 2)

        groups.append(row)
    return pd.DataFrame(groups).sort_values(by="group_id")


def save_overlay(image, rois, out_path, title=None):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for i, roi in enumerate(rois, start=1):
        verts = roi["verts"]
        g = roi.get("group", None)
        ax.plot(verts[:, 0], verts[:, 1], linewidth=1.8, color=color_for_group(g))
        ys, xs = np.nonzero(roi["mask"]) if isinstance(roi.get("mask"), np.ndarray) else (np.array([]), np.array([]))
        if xs.size:
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, f"{i} (G{g})", ha='center', va='center', fontsize=10,
                    color=color_for_group(g))
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def maybe_plot_thickness_vs_intensity(df, out_path_png):
    if "thickness" not in df.columns:
        return
    try:
        x = pd.to_numeric(df["thickness"])
    except Exception:
        return
    if x.isna().any():
        return
    y = df["mean_0_255"]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("Mean SE intensity (0–255)")
    ax.set_title("Thickness vs. SE Mean Intensity")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200)
    plt.close(fig)


def ensure_outdir(image_path, output_dir=None):
    if output_dir and os.path.isdir(output_dir):
        return output_dir
    d = os.path.dirname(image_path)
    return d if d else os.getcwd()

# --------------- ROI ALIGNMENT (translate/rotate/scale) ---------------
class ROIAligner:
    """
    Interactive aligner for previously saved polygons.

    Keys:
      Arrow keys : translate ±1 px (Left/Right/Up/Down)
      WASD       : translate ±5 px (A/D = x, W/S = y)
      ',' '.'    : rotate ±0.1°
      '[' ']'    : rotate ±1.0°
      '-' '='    : scale −/+ by 0.1% (uniform)
      '_' '+'    : scale −/+ by 1.0% (hold Shift)
      'r'        : reset transform (tx=ty=0, scale=1, rot=0)
      'm'        : measure now with current transform (keeps window open)
      'q'        : measure & finish (closes window)
      'h'        : toggle help text
    """
    def __init__(self, ax, image, rois_template):
        self.ax = ax
        self.image = image
        self.H, self.W = image.shape[:2]
        self.cx = self.W/2.0
        self.cy = self.H/2.0

        # Template polygons (untransformed)
        self.template = []
        for r in rois_template:
            verts = np.asarray(r["verts"], dtype=float)
            if verts.shape[0] < 3:
                continue
            # Ensure closed
            if not np.allclose(verts[0], verts[-1]):
                verts = np.vstack([verts, verts[0]])
            self.template.append({
                "region_id": int(r.get("region_id", len(self.template)+1)),
                "group": r.get("group", None),
                "verts": verts
            })

        # Current transform parameters
        self.tx = 0.0
        self.ty = 0.0
        self.scale = 1.0
        self.rot_deg = 0.0

        # Steps
        self.trans_step_px = 1.0
        self.trans_step_px_fast = 5.0
        self.rot_step_small = 0.1
        self.rot_step_big = 1.0
        self.scl_step_small = 0.001
        self.scl_step_big = 0.01

        # Artists
        self.lines = []
        self.labels = []
        self._help_artist = None

        # Draw initial
        self._rebuild_artists()
        self._update_title()
        self.cid_key = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)

        self.finished = False
        self.last_measured_df = None
        self.last_rois = None

    # --- math helpers ---
    def _apply_transform(self, verts):
        v = np.asarray(verts, dtype=float)
        v = v - np.array([[self.cx, self.cy]])
        if not np.isclose(self.scale, 1.0):
            v = v * self.scale
        if not np.isclose(self.rot_deg, 0.0):
            th = math.radians(self.rot_deg)
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]])
            v = v @ R.T
        v = v + np.array([[self.cx + self.tx, self.cy + self.ty]])
        return v

    def _transformed_polys(self):
        out = []
        for r in self.template:
            tv = self._apply_transform(r["verts"])
            out.append({"region_id": r["region_id"], "group": r.get("group"), "verts": tv})
        return out

    # --- draw/update ---
    def _rebuild_artists(self):
        # clear existing
        for ln in self.lines:
            try: ln.remove()
            except Exception: pass
        for lb in self.labels:
            try: lb.remove()
            except Exception: pass
        self.lines, self.labels = [], []

        self.ax.imshow(self.image, cmap='gray')
        for r in self._transformed_polys():
            v = r["verts"]
            g = r.get("group")
            ln, = self.ax.plot(v[:,0], v[:,1], color=color_for_group(g), linewidth=1.8)
            self.lines.append(ln)
            # simple centroid label
            cy = np.mean(v[:,1])
            cx = np.mean(v[:,0])
            lb = self.ax.text(cx, cy, f"{r['region_id']} (G{g})", color=color_for_group(g),
                              fontsize=9, ha='center', va='center')
            self.labels.append(lb)
        self.ax.set_axis_off()
        self.ax.figure.canvas.draw_idle()

    def _update_title(self):
        self.ax.set_title(f"Align ROIs — tx={self.tx:.2f}px, ty={self.ty:.2f}px, scale={self.scale:.5f}, rot={self.rot_deg:.2f}°  |  q: finish, m: measure, r: reset, h: help")
        self.ax.figure.canvas.draw_idle()

    def _toggle_help(self):
        if self._help_artist is not None:
            try:
                self._help_artist.remove()
            except Exception:
                pass
            self._help_artist = None
            self.ax.figure.canvas.draw_idle()
            return
        text = (
            "Arrow: move 1px  |  WASD: move 5px\n"
            ",[ / ]: rot ±1°  |  , .: rot ±0.1°\n"
            "-/_: scale −0.1/−1%  |  =/+: scale +0.1/+1%\n"
            "r: reset  |  m: measure  |  q: measure+finish")
        self._help_artist = self.ax.text(0.01, 0.01, text, transform=self.ax.transAxes,
                                         fontsize=9, va='bottom', ha='left',
                                         bbox=dict(boxstyle="round", alpha=0.3))
        self.ax.figure.canvas.draw_idle()

    def _measure_now(self):
        # Build masks from transformed polygons
        rois = []
        H, W = self.image.shape[:2]
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        for r in self._transformed_polys():
            v = r["verts"]
            # ensure closed
            if not np.allclose(v[0], v[-1]):
                v = np.vstack([v, v[0]])
            path = Path(v)
            m = path.contains_points(coords).reshape(H, W)
            rois.append({"verts": v, "mask": m, "group": r.get("group", None)})
        df = measure_rois(self.image, rois)
        self.last_measured_df = df
        self.last_rois = rois
        return df, rois

    def on_key(self, event):
        k = (event.key or '').lower()
        # translate fine (arrows)
        if k == 'left':  self.tx -= self.trans_step_px
        elif k == 'right': self.tx += self.trans_step_px
        elif k == 'up':    self.ty -= self.trans_step_px
        elif k == 'down':  self.ty += self.trans_step_px
        # translate coarse (wasd)
        elif k == 'a': self.tx -= self.trans_step_px_fast
        elif k == 'd': self.tx += self.trans_step_px_fast
        elif k == 'w': self.ty -= self.trans_step_px_fast
        elif k == 's': self.ty += self.trans_step_px_fast
        # rotate
        elif k == ',': self.rot_deg -= self.rot_step_small
        elif k == '.': self.rot_deg += self.rot_step_small
        elif k == '[': self.rot_deg -= self.rot_step_big
        elif k == ']': self.rot_deg += self.rot_step_big
        # scale
        elif k == '-': self.scale *= (1.0 - self.scl_step_small)
        elif k == '=': self.scale *= (1.0 + self.scl_step_small)
        elif k == '_': self.scale *= (1.0 - self.scl_step_big)
        elif k == '+': self.scale *= (1.0 + self.scl_step_big)
        # reset
        elif k == 'r':
            self.tx = self.ty = 0.0
            self.scale = 1.0
            self.rot_deg = 0.0
        # help
        elif k == 'h':
            self._toggle_help()
            return
        # measure (non-blocking)
        elif k == 'm':
            df, _ = self._measure_now()
            print("\n[Align] Temporary measurement:")
            with pd.option_context('display.max_columns', None, 'display.width', 160):
                print(df)
        # finish
        elif k == 'q':
            # Do a final measurement and close
            self._measure_now()
            self.finished = True
            try:
                self.ax.figure.canvas.mpl_disconnect(self.cid_key)
            except Exception:
                pass
            plt.close(self.ax.figure)
            return
        else:
            # unhandled key
            return

        # if we get here, we updated transform → refresh artists & title
        self._rebuild_artists()
        self._update_title()

# ----------------------------- Main -----------------------------

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")

    outdir = ensure_outdir(IMAGE_PATH, OUTPUT_DIR)
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # Load & crop
    img = load_image_grayscale(IMAGE_PATH, crop_rows_end=CROP_ROWS_END, crop_cols_end=CROP_COLS_END)
    print("Working image shape (H, W):", img.shape)

    # Decide JSON path
    json_path = ROIS_JSON_PATH or os.path.join(outdir, f"{base}_roi_polygons.json")

    # Ask user whether to create new ROIs or reuse existing
    use_new = None
    while use_new is None:
        ans = input("Do you want to set NEW ROIs? [y/n]: ").strip().lower()
        if ans in ('y', 'yes'): use_new = True
        elif ans in ('n', 'no'): use_new = False
        else: print("Please answer y or n.")

    if use_new:
        # Show ROI window to draw fresh ROIs
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        collector = ROICollector(ax, img)

        # Bring window to front
        try:
            mgr = plt.get_current_fig_manager()
            try: mgr.set_window_title("SEM ROI — draw polygons, Enter to close, [/] change group, 'u' undo, 'q' finish")
            except Exception: pass
            try:
                mgr.window.raise_(); mgr.window.activateWindow()
            except Exception:
                pass
        except Exception:
            pass

        # Keep UI alive until 'q' pressed
        plt.show(block=False)
        print("ROI window open. Draw polygons. Press [ / ] to change ACTIVE GROUP (or 0–9 to jump). 'u' undo. 'q' finish.")
        while not collector.finished and plt.fignum_exists(fig.number):
            plt.pause(0.05)

        rois = collector.rois
        if not rois:
            print("No ROIs were drawn. Exiting.")
            return

        # Save polygon vertices (without masks/artist handles)
        save_rois_json(rois, json_path, img.shape, meta={
            "source_image": os.path.basename(IMAGE_PATH),
            "note": "Freshly drawn ROIs"
        })
        print(f"💾 Saved ROI polygons to: {json_path}")

        # Measure & save (per-ROI)
        df = measure_rois(img, rois)

        # Optional background subtraction BEFORE grouping
        if SUBTRACT_BACKGROUND:
            df = apply_background(df, BACKGROUND_INDEX)

        # Add thickness & area µm²
        df = add_thickness_and_area(df, THICKNESS_VALUES, PIXEL_SIZE_NM)

        # ---- Group summary ----
        group_df = summarize_groups(df)

        # Save files
        csv_path = os.path.join(outdir, f"{base}_se_intensity_per_roi.csv")
        group_csv_path = os.path.join(outdir, f"{base}_se_intensity_group_summary.csv")
        overlay_path = os.path.join(outdir, f"{base}_se_roi_overlay.png")

        df.to_csv(csv_path, index=False)
        group_df.to_csv(group_csv_path, index=False)
        save_overlay(img, rois, overlay_path, title="New ROIs")

        # Optional plot
        plot_path = os.path.join(outdir, f"{base}_thickness_vs_intensity.png")
        maybe_plot_thickness_vs_intensity(df, plot_path)

        print("✅ Saved per-ROI CSV:", csv_path)
        print("✅ Saved group summary CSV:", group_csv_path)
        print("✅ Saved ROI overlay image:", overlay_path)
        if os.path.exists(plot_path):
            print("✅ Saved plot (if thickness numeric):", plot_path)

        with pd.option_context('display.max_columns', None, 'display.width', 160):
            print("\nPer-ROI measurements:")
            print(df)
            if not group_df.empty:
                print("\nPer-GROUP summary:")
                print(group_df)
            else:
                print("\n(No groups found in ROIs)")
        return

    # ---- Load existing polygons and ALIGN ----
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ROI JSON not found: {json_path}\nCreate new ROIs first (answer 'y' next time).")

    saved_rois, saved_shape, meta = load_rois_json(json_path)
    if saved_shape != img.shape:
        print(f"[warn] Saved ROI shape {saved_shape} differs from current image shape {img.shape}. Alignment tools will compensate (translate/rotate/scale).")

    print(f"Loaded {len(saved_rois)} saved polygons from {json_path}")

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    aligner = ROIAligner(ax, img, saved_rois)

    # Bring window to front
    try:
        mgr = plt.get_current_fig_manager()
        try: mgr.set_window_title("SEM ROI — ALIGN saved polygons: arrows/WASD move, [,] & , . rotate, - = scale, r reset, m measure, q finish")
        except Exception: pass
        try:
            mgr.window.raise_(); mgr.window.activateWindow()
        except Exception:
            pass
    except Exception:
        pass

    plt.show(block=False)
    print("Align window open. Use arrows/WASD to translate, [/] and , . to rotate, -/= to scale, r reset, m measure, q finish.")
    while not aligner.finished and plt.fignum_exists(fig.number):
        plt.pause(0.05)

    # After finish, we should have aligner.last_rois & last_measured_df
    if aligner.last_rois is None or aligner.last_measured_df is None:
        print("No measurement produced. Exiting.")
        return

    # Save transformed polygons as a new JSON for reproducibility
    aligned_json = os.path.join(outdir, f"{base}_roi_polygons_aligned.json")
    save_rois_json(
        [{"verts": r["verts"], "group": r.get("group", None)} for r in aligner.last_rois],
        aligned_json,
        img.shape,
        meta={
            "aligned_from": os.path.basename(json_path),
            "transform": {"tx_px": aligner.tx, "ty_px": aligner.ty, "scale": aligner.scale, "rot_deg": aligner.rot_deg}
        }
    )
    print(f"💾 Saved ALIGNED ROI polygons to: {aligned_json}")

    # Optional background subtraction BEFORE grouping
    df = aligner.last_measured_df
    if SUBTRACT_BACKGROUND:
        df = apply_background(df, BACKGROUND_INDEX)

    # Add thickness & area µm²
    df = add_thickness_and_area(df, THICKNESS_VALUES, PIXEL_SIZE_NM)

    group_df = summarize_groups(df)

    # Save outputs (use suffix _aligned)
    csv_path = os.path.join(outdir, f"{base}_se_intensity_per_roi_aligned.csv")
    group_csv_path = os.path.join(outdir, f"{base}_se_intensity_group_summary_aligned.csv")
    overlay_path = os.path.join(outdir, f"{base}_se_roi_overlay_aligned.png")

    df.to_csv(csv_path, index=False)
    group_df.to_csv(group_csv_path, index=False)

    # also save a static overlay image with transformed polygons
    # attach masks to pass into save_overlay for centroid labels
    rois_with_masks = aligner.last_rois
    save_overlay(img, rois_with_masks, overlay_path, title="Aligned ROIs")

    print("✅ Saved per-ROI CSV:", csv_path)
    print("✅ Saved group summary CSV:", group_csv_path)
    print("✅ Saved ROI overlay image:", overlay_path)

    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print("\nPer-ROI measurements (aligned):")
        print(df)
        if not group_df.empty:
            print("\nPer-GROUP summary (aligned):")
            print(group_df)
        else:
            print("\n(No groups found in ROIs)")


if __name__ == "__main__":
    main()

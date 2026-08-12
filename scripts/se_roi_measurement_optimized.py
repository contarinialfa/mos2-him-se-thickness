"""Interactive SEM ROI measurement with grouped, reusable polygons.

Performance-sensitive design:
* Polygon membership is evaluated only inside each polygon's bounding box.
* Full-image Boolean masks are not retained for every ROI.
* Alignment updates existing line/text artists in place.
* Matplotlib's native event loop is used instead of a polling loop.
"""

import json
import math
import numbers
import os
from datetime import datetime, timezone

import matplotlib

try:
    matplotlib.use("QtAgg")
except Exception:
    matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

plt.rcParams["figure.raise_window"] = True

# ------------------ USER SETTINGS ------------------
IMAGE_PATH = r"2-1tape-90deg.tif"

# Analyze the top portion only (crop out the bottom legend).
CROP_ROWS_END = 2047
CROP_COLS_END = None

# Optional metadata.
THICKNESS_VALUES = []  # Same order as the ROIs.
PIXEL_SIZE_NM = None   # For example, 5.0 nm/pixel.

# Optional background subtraction.
SUBTRACT_BACKGROUND = False
BACKGROUND_INDEX = 0

# None means the image's directory.
OUTPUT_DIR = None

# None means <imagebase>_roi_polygons.json.
ROIS_JSON_PATH = None
# ---------------------------------------------------

GROUP_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def color_for_group(group):
    if group is None:
        return "yellow"
    return GROUP_COLORS[int(group) % len(GROUP_COLORS)]


# ------------------ Image loading ------------------

def _to_float01(array):
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32)
        maximum = np.nanmax(result) if result.size else 1.0
        if maximum > 1.0:
            result /= 255.0 if maximum <= 255.0 else 65535.0
        return np.clip(result, 0.0, 1.0)

    maximum = np.iinfo(array.dtype).max
    return np.clip(array.astype(np.float32) / float(maximum), 0.0, 1.0)


def _rgba_over_background(rgba, background=(1.0, 1.0, 1.0)):
    rgb = _to_float01(rgba[..., :3])
    alpha = _to_float01(rgba[..., 3:4])
    background = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    return rgb * alpha + background * (1.0 - alpha)


def _rgb_to_gray(rgb):
    rgb = _to_float01(rgb)
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def load_image_grayscale(path, crop_rows_end=None, crop_cols_end=None):
    # The context manager closes the TIFF promptly after loading it.
    with Image.open(path) as source:
        array = np.array(source)

    if array.ndim == 2:
        gray = _to_float01(array)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels in (1, 2):
            gray = _to_float01(array[..., 0])
        elif channels == 3:
            gray = _rgb_to_gray(array)
        elif channels == 4:
            gray = _rgb_to_gray(_rgba_over_background(array))
        else:
            raise ValueError(f"Unexpected channel count: {channels}")
    else:
        raise ValueError(f"Unexpected image shape: {array.shape}")

    if crop_rows_end is not None:
        gray = gray[:crop_rows_end, :]
    if crop_cols_end is not None:
        gray = gray[:, :crop_cols_end]
    return np.ascontiguousarray(gray)


# ------------------ Fast polygon operations ------------------

def _closed_vertices(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("ROI vertices must be an N x 2 array")
    if vertices.shape[0] and not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))
    return vertices


def polygon_pixel_data(image, vertices):
    """Return pixels and their centroid, testing only the polygon bounding box.

    Pixel coordinates intentionally remain integer-valued to preserve the
    original script's Path.contains_points behavior.
    """
    vertices = _closed_vertices(vertices)
    if vertices.shape[0] < 4:  # Three vertices plus the closing vertex.
        return np.empty(0, dtype=image.dtype), None

    height, width = image.shape[:2]
    x0 = max(0, int(math.ceil(np.min(vertices[:, 0]))))
    x1 = min(width - 1, int(math.floor(np.max(vertices[:, 0]))))
    y0 = max(0, int(math.ceil(np.min(vertices[:, 1]))))
    y1 = min(height - 1, int(math.floor(np.max(vertices[:, 1]))))
    if x0 > x1 or y0 > y1:
        return np.empty(0, dtype=image.dtype), None

    # np.indices allocates only for this ROI's bounding rectangle, not the
    # complete image. The two-column array is also limited to that rectangle.
    local_y, local_x = np.indices((y1 - y0 + 1, x1 - x0 + 1))
    local_x += x0
    local_y += y0
    points = np.empty((local_x.size, 2), dtype=float)
    points[:, 0] = local_x.ravel()
    points[:, 1] = local_y.ravel()

    inside = Path(vertices).contains_points(points)
    if not np.any(inside):
        return np.empty(0, dtype=image.dtype), None

    pixels = image[y0:y1 + 1, x0:x1 + 1].ravel()[inside]
    centroid = (float(points[inside, 0].mean()), float(points[inside, 1].mean()))
    return pixels, centroid


def polygon_label_position(vertices):
    """Cheap fallback label position that excludes a repeated closing point."""
    vertices = _closed_vertices(vertices)
    core = vertices[:-1] if vertices.shape[0] > 1 else vertices
    if not core.size:
        return 0.0, 0.0
    return float(core[:, 0].mean()), float(core[:, 1].mean())


# ------------------ ROI collection ------------------

class ROICollector:
    def __init__(self, axes, image):
        self.ax = axes
        self.image = image
        self.rois = []
        self.finished = False
        self.active_group = 1
        self.cid_key = self.ax.figure.canvas.mpl_connect(
            "key_press_event", self.on_key
        )
        self._instructions = None
        self.selector = None
        self._make_instructions()
        self._new_selector()

    def _title_text(self):
        return (
            "SEM ROI — Draw polygons"
            f" | Active group: {self.active_group}"
            " | [ / ] group, 0–9 jump, u undo, q finish"
        )

    def _make_instructions(self):
        if self._instructions is None:
            self._instructions = self.ax.text(
                0.01,
                0.01,
                "Left-click: add points | Enter/double-click: close polygon\n"
                "[ / ]: change group | 0–9: select group | u: undo | q: finish",
                transform=self.ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "alpha": 0.3},
            )
        self.ax.set_title(self._title_text())
        self.ax.figure.canvas.draw_idle()

    def _new_selector(self):
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
        self.selector = PolygonSelector(self.ax, self.onselect, useblit=True)

    def onselect(self, vertices):
        vertices = _closed_vertices(vertices)
        if vertices.shape[0] < 4:
            self._new_selector()
            return

        group = int(self.active_group)
        line = self.ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            linewidth=1.8,
            color=color_for_group(group),
        )[0]

        # Restrict the one-time centroid calculation to the ROI bounding box.
        _, centroid = polygon_pixel_data(self.image, vertices)
        if centroid is None:
            centroid = polygon_label_position(vertices)
        label = self.ax.text(
            *centroid,
            f"{len(self.rois) + 1} (G{group})",
            ha="center",
            va="center",
            fontsize=10,
            color=color_for_group(group),
        )

        # Do not retain a full H x W mask for each ROI.
        self.rois.append(
            {
                "verts": vertices.copy(),
                "group": group,
                "line": line,
                "label": label,
            }
        )
        self.ax.figure.canvas.draw_idle()
        self._new_selector()

    def _undo_last(self):
        if not self.rois:
            return
        roi = self.rois.pop()
        for artist_name in ("line", "label"):
            artist = roi.get(artist_name)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass

        for index, remaining in enumerate(self.rois, start=1):
            label = remaining.get("label")
            if label is not None:
                label.set_text(f"{index} (G{remaining['group']})")
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        key = event.key
        if key == "q":
            self.finished = True
            try:
                self.selector.disconnect_events()
                self.ax.figure.canvas.mpl_disconnect(self.cid_key)
            except Exception:
                pass
            plt.close(self.ax.figure)
        elif key == "u":
            self._undo_last()
        elif key == "[":
            self.active_group = max(0, self.active_group - 1)
            self._make_instructions()
        elif key == "]":
            self.active_group += 1
            self._make_instructions()
        elif key in "0123456789":
            self.active_group = int(key)
            self._make_instructions()


# ------------------ ROI persistence ------------------

def save_rois_json(rois, json_path, image_shape, meta=None):
    directory = os.path.dirname(os.path.abspath(json_path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "image_shape": list(map(int, image_shape[:2])),
        "rois": [
            {
                "region_id": index,
                "group": (
                    int(roi["group"]) if roi.get("group") is not None else None
                ),
                "verts": np.asarray(roi["verts"], dtype=float).tolist(),
            }
            for index, roi in enumerate(rois, start=1)
        ],
        "meta": meta or {},
    }
    with open(json_path, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2)


def load_rois_json(json_path):
    with open(json_path, "r", encoding="utf-8") as source:
        payload = json.load(source)

    rois = [
        {
            "region_id": int(item.get("region_id", index)),
            "group": item.get("group"),
            "verts": _closed_vertices(item["verts"]),
        }
        for index, item in enumerate(payload.get("rois", []), start=1)
    ]
    image_shape = tuple(payload.get("image_shape", (None, None)))
    return rois, image_shape, payload.get("meta", {})


# ------------------ Measurements ------------------

def measure_rois(image, rois):
    rows = []
    for index, roi in enumerate(rois, start=1):
        # Accept old in-memory ROI objects too, while favoring the fast path.
        mask = roi.get("mask")
        if isinstance(mask, np.ndarray) and mask.shape == image.shape[:2]:
            values = image[mask]
        else:
            values, _ = polygon_pixel_data(image, roi["verts"])

        if values.size == 0:
            stats = {
                "region_id": int(roi.get("region_id", index)),
                "group_id": roi.get("group"),
                "area_px": 0,
                "mean_0_1": np.nan,
                "median_0_1": np.nan,
                "std_0_1": np.nan,
                "min_0_1": np.nan,
                "max_0_1": np.nan,
            }
        else:
            stats = {
                "region_id": int(roi.get("region_id", index)),
                "group_id": roi.get("group"),
                "area_px": int(values.size),
                "mean_0_1": float(values.mean()),
                "median_0_1": float(np.median(values)),
                "std_0_1": float(values.std(ddof=1) if values.size > 1 else 0.0),
                "min_0_1": float(values.min()),
                "max_0_1": float(values.max()),
            }
        rows.append(stats)

    columns = [
        "region_id",
        "group_id",
        "area_px",
        "mean_0_1",
        "median_0_1",
        "std_0_1",
        "min_0_1",
        "max_0_1",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("mean_0_1", "median_0_1", "min_0_1", "max_0_1"):
        frame[column.replace("_0_1", "_0_255")] = (
            frame[column] * 255.0
        ).round(2)
    return frame


def apply_background(frame, background_index):
    if frame.empty or not 0 <= background_index < len(frame):
        return frame
    frame = frame.copy()
    background_mean = frame.iloc[background_index]["mean_0_1"]
    background_median = frame.iloc[background_index]["median_0_1"]
    frame["mean_0_1"] -= background_mean
    frame["median_0_1"] -= background_median
    for column in ("mean_0_1", "median_0_1"):
        frame[column.replace("_0_1", "_0_255")] = (
            frame[column] * 255.0
        ).round(2)
    return frame


def add_thickness_and_area(frame, thickness_values, pixel_size_nm):
    frame = frame.copy()
    frame["thickness"] = (
        thickness_values
        if thickness_values and len(thickness_values) == len(frame)
        else None
    )
    if isinstance(pixel_size_nm, numbers.Number) and pixel_size_nm > 0:
        frame["area_um2"] = (
            frame["area_px"] * pixel_size_nm**2 / 1e6
        ).round(6)
    else:
        frame["area_um2"] = None
    return frame


def pooled_group_stats(group):
    pixel_count = int(group["area_px"].sum())
    if pixel_count <= 1:
        return (
            float(group["mean_0_1"].mean()),
            float(group["std_0_1"].mean()),
            pixel_count,
        )

    mean = float(
        (group["mean_0_1"] * group["area_px"]).sum() / pixel_count
    )
    within = (
        (group["area_px"] - 1) * group["std_0_1"] ** 2
    ).fillna(0.0).sum()
    between = (
        group["area_px"] * (group["mean_0_1"] - mean) ** 2
    ).fillna(0.0).sum()
    standard_deviation = math.sqrt(
        max(float((within + between) / (pixel_count - 1)), 0.0)
    )
    return mean, standard_deviation, pixel_count


def summarize_groups(frame):
    if "group_id" not in frame.columns or frame.empty:
        return pd.DataFrame()

    rows = []
    for group_id, group in frame.groupby("group_id", dropna=False):
        roi_count = len(group)
        total_area = int(group["area_px"].sum())
        pooled_mean, pooled_std, _ = pooled_group_stats(group)
        simple_mean = float(group["mean_0_1"].mean())
        simple_std = (
            float(group["mean_0_1"].std(ddof=1))
            if roi_count > 1
            else np.nan
        )
        weighted_mean = (
            float((group["mean_0_1"] * group["area_px"]).sum() / total_area)
            if total_area > 0
            else np.nan
        )

        row = {
            "group_id": group_id,
            "n_rois": int(roi_count),
            "total_area_px": total_area,
            "pooled_mean_0_1": pooled_mean,
            "pooled_std_0_1": pooled_std,
            "simple_mean_of_roi_means_0_1": simple_mean,
            "simple_std_of_roi_means_0_1": simple_std,
            "area_weighted_mean_0_1": weighted_mean,
        }
        for key in (
            "pooled_mean_0_1",
            "pooled_std_0_1",
            "simple_mean_of_roi_means_0_1",
            "simple_std_of_roi_means_0_1",
            "area_weighted_mean_0_1",
        ):
            row[key.replace("_0_1", "_0_255")] = (
                np.nan if pd.isna(row[key]) else round(255.0 * row[key], 2)
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group_id", na_position="last")


# ------------------ Output figures ------------------

def save_overlay(image, rois, output_path, title=None):
    figure, axes = plt.subplots()
    axes.imshow(image, cmap="gray")
    for index, roi in enumerate(rois, start=1):
        vertices = np.asarray(roi["verts"])
        group = roi.get("group")
        axes.plot(
            vertices[:, 0],
            vertices[:, 1],
            linewidth=1.8,
            color=color_for_group(group),
        )
        center_x, center_y = polygon_label_position(vertices)
        axes.text(
            center_x,
            center_y,
            f"{roi.get('region_id', index)} (G{group})",
            ha="center",
            va="center",
            fontsize=10,
            color=color_for_group(group),
        )
    if title:
        axes.set_title(title)
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def maybe_plot_thickness_vs_intensity(frame, output_path):
    if "thickness" not in frame.columns:
        return
    thickness = pd.to_numeric(frame["thickness"], errors="coerce")
    if thickness.isna().any():
        return

    figure, axes = plt.subplots()
    axes.scatter(thickness, frame["mean_0_255"])
    axes.set_xlabel("Thickness (nm)")
    axes.set_ylabel("Mean SE intensity (0–255)")
    axes.set_title("Thickness vs. SE Mean Intensity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def ensure_output_directory(image_path, output_directory=None):
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        return output_directory
    return os.path.dirname(os.path.abspath(image_path))


# ------------------ ROI alignment ------------------

class ROIAligner:
    """Translate, rotate, and scale saved polygons interactively."""

    def __init__(self, axes, image, roi_template):
        self.ax = axes
        self.image = image
        self.height, self.width = image.shape[:2]
        self.center_x = self.width / 2.0
        self.center_y = self.height / 2.0

        self.template = []
        for roi in roi_template:
            vertices = _closed_vertices(roi["verts"])
            if vertices.shape[0] >= 4:
                self.template.append(
                    {
                        "region_id": int(
                            roi.get("region_id", len(self.template) + 1)
                        ),
                        "group": roi.get("group"),
                        "verts": vertices,
                    }
                )

        self.tx = 0.0
        self.ty = 0.0
        self.scale = 1.0
        self.rotation_degrees = 0.0

        self.translation_step = 1.0
        self.fast_translation_step = 5.0
        self.small_rotation_step = 0.1
        self.large_rotation_step = 1.0
        self.small_scale_step = 0.001
        self.large_scale_step = 0.01

        self.lines = []
        self.labels = []
        self.help_artist = None
        self.finished = False
        self.last_measured_df = None
        self.last_rois = None

        self._create_artists()
        self._refresh_artists()
        self.cid_key = self.ax.figure.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

    def _apply_transform(self, vertices):
        transformed = np.asarray(vertices, dtype=float).copy()
        transformed -= (self.center_x, self.center_y)
        transformed *= self.scale
        if not np.isclose(self.rotation_degrees, 0.0):
            theta = math.radians(self.rotation_degrees)
            cosine, sine = math.cos(theta), math.sin(theta)
            rotation = np.array(((cosine, -sine), (sine, cosine)))
            transformed = transformed @ rotation.T
        transformed += (
            self.center_x + self.tx,
            self.center_y + self.ty,
        )
        return transformed

    def _transformed_polygons(self):
        return [
            {
                "region_id": roi["region_id"],
                "group": roi.get("group"),
                "verts": self._apply_transform(roi["verts"]),
            }
            for roi in self.template
        ]

    def _create_artists(self):
        # The caller has already drawn the image. Create each overlay once.
        for roi in self.template:
            color = color_for_group(roi.get("group"))
            line = self.ax.plot([], [], color=color, linewidth=1.8)[0]
            label = self.ax.text(
                0,
                0,
                f"{roi['region_id']} (G{roi.get('group')})",
                color=color,
                fontsize=9,
                ha="center",
                va="center",
            )
            self.lines.append(line)
            self.labels.append(label)
        self.ax.set_axis_off()

    def _refresh_artists(self):
        # Updating existing artists avoids adding an image layer on every key.
        for line, label, roi in zip(
            self.lines, self.labels, self._transformed_polygons()
        ):
            vertices = roi["verts"]
            line.set_data(vertices[:, 0], vertices[:, 1])
            label.set_position(polygon_label_position(vertices))

        self.ax.set_title(
            "Align ROIs — "
            f"tx={self.tx:.2f}px, ty={self.ty:.2f}px, "
            f"scale={self.scale:.5f}, rot={self.rotation_degrees:.2f}°"
            " | q finish, m measure, r reset, h help"
        )
        self.ax.figure.canvas.draw_idle()

    def _toggle_help(self):
        if self.help_artist is not None:
            self.help_artist.remove()
            self.help_artist = None
        else:
            self.help_artist = self.ax.text(
                0.01,
                0.01,
                "Arrows: move 1 px | WASD: move 5 px\n"
                ", / .: rotate ±0.1° | [ / ]: rotate ±1°\n"
                "- / =: scale ±0.1% | _ / +: scale ±1%\n"
                "r: reset | m: measure | q: measure and finish",
                transform=self.ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "alpha": 0.3},
            )
        self.ax.figure.canvas.draw_idle()

    def _measure_now(self):
        rois = self._transformed_polygons()
        frame = measure_rois(self.image, rois)
        self.last_measured_df = frame
        self.last_rois = rois
        return frame, rois

    def on_key(self, event):
        key = event.key or ""
        normalized = key.lower()

        if normalized == "left":
            self.tx -= self.translation_step
        elif normalized == "right":
            self.tx += self.translation_step
        elif normalized == "up":
            self.ty -= self.translation_step
        elif normalized == "down":
            self.ty += self.translation_step
        elif normalized == "a":
            self.tx -= self.fast_translation_step
        elif normalized == "d":
            self.tx += self.fast_translation_step
        elif normalized == "w":
            self.ty -= self.fast_translation_step
        elif normalized == "s":
            self.ty += self.fast_translation_step
        elif key == ",":
            self.rotation_degrees -= self.small_rotation_step
        elif key == ".":
            self.rotation_degrees += self.small_rotation_step
        elif key == "[":
            self.rotation_degrees -= self.large_rotation_step
        elif key == "]":
            self.rotation_degrees += self.large_rotation_step
        elif key == "-":
            self.scale *= 1.0 - self.small_scale_step
        elif key == "=":
            self.scale *= 1.0 + self.small_scale_step
        elif key == "_":
            self.scale *= 1.0 - self.large_scale_step
        elif key == "+":
            self.scale *= 1.0 + self.large_scale_step
        elif normalized == "r":
            self.tx = self.ty = 0.0
            self.scale = 1.0
            self.rotation_degrees = 0.0
        elif normalized == "h":
            self._toggle_help()
            return
        elif normalized == "m":
            frame, _ = self._measure_now()
            print("\n[Align] Temporary measurement:")
            with pd.option_context(
                "display.max_columns", None, "display.width", 160
            ):
                print(frame)
            return
        elif normalized == "q":
            self._measure_now()
            self.finished = True
            try:
                self.ax.figure.canvas.mpl_disconnect(self.cid_key)
            except Exception:
                pass
            plt.close(self.ax.figure)
            return
        else:
            return

        self._refresh_artists()


# ------------------ Main workflow ------------------

def _bring_window_forward(title):
    try:
        manager = plt.get_current_fig_manager()
        manager.set_window_title(title)
        manager.window.raise_()
        manager.window.activateWindow()
    except Exception:
        pass


def _print_results(frame, group_frame, heading):
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(f"\n{heading}:")
        print(frame)
        if group_frame.empty:
            print("\n(No groups found in ROIs)")
        else:
            print("\nPer-group summary:")
            print(group_frame)


def _finish_measurement(frame):
    if SUBTRACT_BACKGROUND:
        frame = apply_background(frame, BACKGROUND_INDEX)
    return add_thickness_and_area(frame, THICKNESS_VALUES, PIXEL_SIZE_NM)


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")

    output_directory = ensure_output_directory(IMAGE_PATH, OUTPUT_DIR)
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    image = load_image_grayscale(
        IMAGE_PATH,
        crop_rows_end=CROP_ROWS_END,
        crop_cols_end=CROP_COLS_END,
    )
    print("Working image shape (H, W):", image.shape)

    json_path = ROIS_JSON_PATH or os.path.join(
        output_directory, f"{base}_roi_polygons.json"
    )

    use_new = None
    while use_new is None:
        answer = input("Do you want to set NEW ROIs? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            use_new = True
        elif answer in ("n", "no"):
            use_new = False
        else:
            print("Please answer y or n.")

    if use_new:
        figure, axes = plt.subplots()
        axes.imshow(image, cmap="gray")
        collector = ROICollector(axes, image)
        _bring_window_forward(
            "SEM ROI — draw polygons, Enter to close, q to finish"
        )
        print(
            "ROI window open. Draw polygons; [ / ] changes group, "
            "u undoes, and q finishes."
        )
        # Qt's native event loop uses no busy polling while the UI is idle.
        plt.show(block=True)

        rois = collector.rois
        if not rois:
            print("No ROIs were drawn. Exiting.")
            return

        save_rois_json(
            rois,
            json_path,
            image.shape,
            meta={
                "source_image": os.path.basename(IMAGE_PATH),
                "note": "Freshly drawn ROIs",
            },
        )
        print("Saved ROI polygons:", json_path)

        frame = _finish_measurement(measure_rois(image, rois))
        group_frame = summarize_groups(frame)
        csv_path = os.path.join(
            output_directory, f"{base}_se_intensity_per_roi.csv"
        )
        group_csv_path = os.path.join(
            output_directory, f"{base}_se_intensity_group_summary.csv"
        )
        overlay_path = os.path.join(
            output_directory, f"{base}_se_roi_overlay.png"
        )
        plot_path = os.path.join(
            output_directory, f"{base}_thickness_vs_intensity.png"
        )

        frame.to_csv(csv_path, index=False)
        group_frame.to_csv(group_csv_path, index=False)
        save_overlay(image, rois, overlay_path, title="New ROIs")
        maybe_plot_thickness_vs_intensity(frame, plot_path)

        print("Saved per-ROI CSV:", csv_path)
        print("Saved group summary CSV:", group_csv_path)
        print("Saved ROI overlay image:", overlay_path)
        if os.path.exists(plot_path):
            print("Saved thickness plot:", plot_path)
        _print_results(frame, group_frame, "Per-ROI measurements")
        return

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"ROI JSON not found: {json_path}\n"
            "Create new ROIs first (answer 'y' next time)."
        )

    saved_rois, saved_shape, _ = load_rois_json(json_path)
    if saved_shape != image.shape:
        print(
            f"[warning] Saved ROI shape {saved_shape} differs from "
            f"the current image shape {image.shape}."
        )
    print(f"Loaded {len(saved_rois)} saved polygons from {json_path}")

    figure, axes = plt.subplots()
    axes.imshow(image, cmap="gray")
    aligner = ROIAligner(axes, image, saved_rois)
    _bring_window_forward(
        "SEM ROI — align saved polygons; q measures and finishes"
    )
    print(
        "Align window open. Arrows/WASD move, [/] and ,/. rotate, "
        "-/= scale, r resets, m measures, and q finishes."
    )
    plt.show(block=True)

    if aligner.last_rois is None or aligner.last_measured_df is None:
        print("No measurement produced. Exiting.")
        return

    aligned_json_path = os.path.join(
        output_directory, f"{base}_roi_polygons_aligned.json"
    )
    save_rois_json(
        aligner.last_rois,
        aligned_json_path,
        image.shape,
        meta={
            "aligned_from": os.path.basename(json_path),
            "transform": {
                "tx_px": aligner.tx,
                "ty_px": aligner.ty,
                "scale": aligner.scale,
                "rot_deg": aligner.rotation_degrees,
            },
        },
    )
    print("Saved aligned ROI polygons:", aligned_json_path)

    frame = _finish_measurement(aligner.last_measured_df)
    group_frame = summarize_groups(frame)
    csv_path = os.path.join(
        output_directory, f"{base}_se_intensity_per_roi_aligned.csv"
    )
    group_csv_path = os.path.join(
        output_directory, f"{base}_se_intensity_group_summary_aligned.csv"
    )
    overlay_path = os.path.join(
        output_directory, f"{base}_se_roi_overlay_aligned.png"
    )

    frame.to_csv(csv_path, index=False)
    group_frame.to_csv(group_csv_path, index=False)
    save_overlay(
        image, aligner.last_rois, overlay_path, title="Aligned ROIs"
    )

    print("Saved per-ROI CSV:", csv_path)
    print("Saved group summary CSV:", group_csv_path)
    print("Saved ROI overlay image:", overlay_path)
    _print_results(frame, group_frame, "Per-ROI measurements (aligned)")


if __name__ == "__main__":
    main()

"""Interactively measure grouped polygon ROIs in an SEM image.

The script supports three workflows:

``manual``
    Draw a fresh set of polygon regions and measure them.
``create-template``
    Mark alignment points, draw regions, and save both to a JSON template.
``apply-template``
    Mark the corresponding alignment points in another image, transform a
    saved template, and measure the transformed regions.

Each run writes a per-ROI table, a group summary, and an annotated overlay.
The first 2,047 rows are analyzed by default so that the instrument footer is
excluded from the 2,048-pixel-high source images in this repository.
"""

from __future__ import annotations

import argparse
import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

try:
    matplotlib.use("QtAgg")
except ImportError:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath
from matplotlib.widgets import PolygonSelector
from PIL import Image

GROUP_COLORS = (
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
)


@dataclass
class Region:
    """One polygon ROI and its group assignment."""

    vertices: np.ndarray
    mask: np.ndarray
    group_id: int
    line: Any | None = None
    label: Any | None = None


def color_for_group(group_id: int) -> str:
    """Return a repeatable display color for a numeric group."""
    return GROUP_COLORS[int(group_id) % len(GROUP_COLORS)]


def to_float01(array: np.ndarray) -> np.ndarray:
    """Convert integer or floating-point image data to the range 0--1."""
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        converted = array.astype(np.float32)
        maximum = np.nanmax(converted) if converted.size else 1.0
        if maximum > 1.0:
            converted /= 255.0 if maximum <= 255.0 else 65535.0
        return np.clip(converted, 0.0, 1.0)

    maximum = np.iinfo(array.dtype).max
    return np.clip(array.astype(np.float32) / float(maximum), 0.0, 1.0)


def composite_rgba(
    rgba: np.ndarray,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite an RGBA image over a solid RGB background."""
    rgb = to_float01(rgba[..., :3])
    alpha = to_float01(rgba[..., 3:4])
    background_array = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    return rgb * alpha + background_array * (1.0 - alpha)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB data to luminance using Rec. 709 coefficients."""
    rgb = to_float01(rgb)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )


def load_grayscale_image(
    path: Path,
    crop_rows_end: int | None,
    crop_cols_end: int | None,
) -> np.ndarray:
    """Load an image as normalized grayscale and apply optional end crops."""
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 2:
        gray = to_float01(array)
    elif array.ndim == 3:
        channels = array.shape[2]
        if channels in (1, 2):
            gray = to_float01(array[..., 0])
        elif channels == 3:
            gray = rgb_to_gray(array)
        elif channels == 4:
            gray = rgb_to_gray(composite_rgba(array))
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if crop_rows_end is not None:
        gray = gray[:crop_rows_end, :]
    if crop_cols_end is not None:
        gray = gray[:, :crop_cols_end]
    return gray


def vertices_to_mask(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon vertices into a Boolean image mask."""
    height, width = shape
    yy, xx = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
    return PolygonPath(vertices).contains_points(coordinates).reshape(shape)


class ROICollector:
    """Collect grouped polygon ROIs from a Matplotlib image window."""

    def __init__(self, axes: Any, image: np.ndarray) -> None:
        self.axes = axes
        self.image = image
        self.regions: list[Region] = []
        self.finished = False
        self.active_group = 1
        self.selector: PolygonSelector | None = None
        self.instructions: Any | None = None
        self.key_connection = self.axes.figure.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )
        self.refresh_instructions()
        self.start_selector()

    def refresh_instructions(self) -> None:
        """Refresh the active-group label and keyboard help."""
        if self.instructions is not None:
            self.instructions.remove()

        help_text = (
            "Draw a polygon: click vertices, then press Enter or double-click.\n"
            "[ and ] change group; 0--9 select a group; u undoes; q finishes."
        )
        self.instructions = self.axes.text(
            0.01,
            0.01,
            help_text,
            transform=self.axes.transAxes,
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "alpha": 0.3},
        )
        self.axes.set_title(f"SEM ROI selection — active group {self.active_group}")
        self.axes.figure.canvas.draw_idle()

    def start_selector(self) -> None:
        """Start a fresh polygon selector for the next ROI."""
        if self.selector is not None:
            self.selector.disconnect_events()
        self.selector = PolygonSelector(
            self.axes,
            self.on_select,
            useblit=False,
        )

    def on_select(self, vertices: list[tuple[float, float]]) -> None:
        """Store a completed polygon and immediately arm the next selector."""
        vertex_array = np.asarray(vertices, dtype=float)
        if len(vertex_array) < 3:
            return
        if not np.allclose(vertex_array[0], vertex_array[-1]):
            vertex_array = np.vstack([vertex_array, vertex_array[0]])

        mask = vertices_to_mask(vertex_array, self.image.shape)
        color = color_for_group(self.active_group)
        line = self.axes.plot(
            vertex_array[:, 0],
            vertex_array[:, 1],
            linewidth=1.8,
            color=color,
        )[0]

        yy, xx = np.nonzero(mask)
        label = None
        if len(xx):
            label = self.axes.text(
                xx.mean(),
                yy.mean(),
                f"{len(self.regions) + 1} (G{self.active_group})",
                color=color,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

        self.regions.append(
            Region(
                vertices=vertex_array.copy(),
                mask=mask,
                group_id=self.active_group,
                line=line,
                label=label,
            )
        )
        self.axes.figure.canvas.draw_idle()
        self.start_selector()

    def undo_last(self) -> None:
        """Remove the most recently drawn ROI."""
        if not self.regions:
            return

        region = self.regions.pop()
        if region.line is not None:
            region.line.remove()
        if region.label is not None:
            region.label.remove()

        for index, remaining in enumerate(self.regions, start=1):
            if remaining.label is not None:
                remaining.label.set_text(f"{index} (G{remaining.group_id})")
        self.axes.figure.canvas.draw_idle()

    def on_key(self, event: Any) -> None:
        """Handle group selection, undo, and finish keyboard shortcuts."""
        key = event.key
        if key == "q":
            self.finished = True
            if self.selector is not None:
                self.selector.disconnect_events()
            self.axes.figure.canvas.mpl_disconnect(self.key_connection)
            plt.close(self.axes.figure)
        elif key == "u":
            self.undo_last()
        elif key == "[":
            self.active_group = max(0, self.active_group - 1)
            self.refresh_instructions()
        elif key == "]":
            self.active_group += 1
            self.refresh_instructions()
        elif key in tuple("0123456789"):
            self.active_group = int(key)
            self.refresh_instructions()


def collect_regions(
    image: np.ndarray,
    alignment_points: np.ndarray | None = None,
) -> list[Region]:
    """Open the polygon editor and return all completed ROIs."""
    figure, axes = plt.subplots()
    axes.imshow(image, cmap="gray")
    if alignment_points is not None:
        plot_alignment_points(axes, alignment_points)

    collector = ROICollector(axes, image)
    plt.show(block=False)
    while not collector.finished and plt.fignum_exists(figure.number):
        plt.pause(0.05)
    return collector.regions


def collect_alignment_points(
    image: np.ndarray,
    count: int,
    mode: str,
) -> np.ndarray:
    """Ask the user to click ordered reference points in the current image."""
    figure, axes = plt.subplots()
    axes.imshow(image, cmap="gray")
    axes.set_title(
        f"{mode}: click {count} corresponding alignment points in order",
        color="darkred",
    )
    points = plt.ginput(n=count, timeout=0, show_clicks=True)
    plt.close(figure)

    if len(points) != count:
        raise RuntimeError(
            f"Expected {count} alignment points; received {len(points)}."
        )
    return np.asarray(points, dtype=float)


def plot_alignment_points(axes: Any, points: np.ndarray) -> None:
    """Draw alignment-point markers and labels on an axes."""
    if len(points) == 0:
        return
    axes.plot(
        points[:, 0],
        points[:, 1],
        "rx",
        markersize=10,
        markeredgewidth=2,
    )
    for index, point in enumerate(points, start=1):
        axes.text(
            point[0] + 15,
            point[1] - 15,
            f"P{index}",
            color="red",
            fontsize=10,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.5,
                "pad": 1,
            },
        )


def save_template(
    path: Path,
    alignment_points: np.ndarray,
    regions: list[Region],
) -> None:
    """Save reference points, polygons, and group assignments as JSON."""
    payload = {
        "pinpoints": alignment_points.tolist(),
        "rois": [
            {
                "group": region.group_id,
                "verts": region.vertices.tolist(),
            }
            for region in regions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_template(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load a saved ROI template."""
    if not path.is_file():
        raise FileNotFoundError(f"ROI template not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["pinpoints"], dtype=float), payload["rois"]


def transform_template_regions(
    image: np.ndarray,
    template_path: Path,
    new_points: np.ndarray,
    transform_type: str,
) -> list[Region]:
    """Transform template polygons to alignment points in a new image."""
    try:
        from skimage import transform
    except ImportError as error:
        raise RuntimeError(
            "Template alignment requires scikit-image. Install requirements.txt."
        ) from error

    old_points, saved_regions = load_template(template_path)
    if len(old_points) != len(new_points):
        raise ValueError(
            "The template and current image must have the same number of "
            "alignment points."
        )

    geometric_transform = transform.estimate_transform(
        transform_type,
        old_points,
        new_points,
    )
    regions = []
    for saved_region in saved_regions:
        vertices = geometric_transform(np.asarray(saved_region["verts"], dtype=float))
        regions.append(
            Region(
                vertices=vertices,
                mask=vertices_to_mask(vertices, image.shape),
                group_id=int(saved_region["group"]),
            )
        )
    return regions


def measure_regions(image: np.ndarray, regions: list[Region]) -> pd.DataFrame:
    """Calculate per-ROI intensity statistics."""
    rows = []
    for region_id, region in enumerate(regions, start=1):
        values = image[region.mask]
        if len(values) == 0:
            row = {
                "region_id": region_id,
                "group_id": region.group_id,
                "area_px": 0,
                "mean_0_1": np.nan,
                "median_0_1": np.nan,
                "std_0_1": np.nan,
                "min_0_1": np.nan,
                "max_0_1": np.nan,
            }
        else:
            row = {
                "region_id": region_id,
                "group_id": region.group_id,
                "area_px": int(region.mask.sum()),
                "mean_0_1": float(values.mean()),
                "median_0_1": float(np.median(values)),
                "std_0_1": float(values.std(ddof=1) if len(values) > 1 else 0.0),
                "min_0_1": float(values.min()),
                "max_0_1": float(values.max()),
            }
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in ("mean_0_1", "median_0_1", "min_0_1", "max_0_1"):
        frame[column.replace("_0_1", "_0_255")] = (frame[column] * 255.0).round(2)
    return frame


def subtract_background(frame: pd.DataFrame, region_id: int) -> pd.DataFrame:
    """Subtract one ROI's mean and median from all ROI means and medians."""
    matches = frame.index[frame["region_id"] == region_id]
    if len(matches) != 1:
        raise ValueError(f"Background region {region_id} was not found exactly once.")

    frame = frame.copy()
    background = frame.loc[matches[0]]
    frame["mean_0_1"] -= background["mean_0_1"]
    frame["median_0_1"] -= background["median_0_1"]
    frame["mean_0_255"] = (frame["mean_0_1"] * 255.0).round(2)
    frame["median_0_255"] = (frame["median_0_1"] * 255.0).round(2)
    return frame


def parse_thickness_map(specification: str | None) -> dict[int, float]:
    """Parse ``group:thickness`` pairs separated by commas."""
    if not specification:
        return {}

    mapping: dict[int, float] = {}
    for item in specification.split(","):
        group_text, thickness_text = item.split(":", maxsplit=1)
        mapping[int(group_text.strip())] = float(thickness_text.strip())
    return mapping


def add_metadata(
    frame: pd.DataFrame,
    thickness_by_group: dict[int, float],
    pixel_size_nm: float | None,
) -> pd.DataFrame:
    """Add group-level thickness and optional physical ROI area."""
    frame = frame.copy()
    frame["thickness"] = frame["group_id"].map(thickness_by_group)
    if isinstance(pixel_size_nm, numbers.Number) and pixel_size_nm > 0:
        frame["area_um2"] = (
            frame["area_px"] * float(pixel_size_nm) ** 2 / 1_000_000.0
        ).round(6)
    else:
        frame["area_um2"] = np.nan
    return frame


def pooled_group_statistics(group: pd.DataFrame) -> tuple[float, float, int]:
    """Pool pixel-level mean and variance from ROI summary statistics."""
    pixel_count = int(group["area_px"].sum())
    if pixel_count <= 1:
        return (
            float(group["mean_0_1"].mean()),
            float(group["std_0_1"].mean()),
            pixel_count,
        )

    mean = float((group["mean_0_1"] * group["area_px"]).sum() / pixel_count)
    within_sum = ((group["area_px"] - 1) * group["std_0_1"].pow(2)).fillna(0.0).sum()
    between_sum = (
        (group["area_px"] * (group["mean_0_1"] - mean).pow(2)).fillna(0.0).sum()
    )
    variance = float((within_sum + between_sum) / (pixel_count - 1))
    return mean, float(np.sqrt(max(variance, 0.0))), pixel_count


def summarize_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Create pixel-pooled and ROI-level summaries for each group."""
    rows = []
    for group_id, group in frame.groupby("group_id", dropna=False):
        pooled_mean, pooled_std, pixel_count = pooled_group_statistics(group)
        roi_count = len(group)
        roi_mean = float(group["mean_0_1"].mean())
        roi_std = float(group["mean_0_1"].std(ddof=1)) if roi_count > 1 else np.nan
        area_weighted_mean = (
            float((group["mean_0_1"] * group["area_px"]).sum() / group["area_px"].sum())
            if pixel_count
            else np.nan
        )
        row = {
            "group_id": group_id,
            "n_rois": roi_count,
            "total_area_px": pixel_count,
            "pooled_mean_0_1": pooled_mean,
            "pooled_std_0_1": pooled_std,
            "simple_mean_of_roi_means_0_1": roi_mean,
            "simple_std_of_roi_means_0_1": roi_std,
            "area_weighted_mean_0_1": area_weighted_mean,
        }
        for name in (
            "pooled_mean_0_1",
            "pooled_std_0_1",
            "simple_mean_of_roi_means_0_1",
            "simple_std_of_roi_means_0_1",
            "area_weighted_mean_0_1",
        ):
            value = row[name]
            row[name.replace("_0_1", "_0_255")] = (
                np.nan if pd.isna(value) else round(255.0 * float(value), 2)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)


def save_overlay(
    image: np.ndarray,
    regions: list[Region],
    alignment_points: np.ndarray | None,
    path: Path,
) -> None:
    """Save the grayscale image with ROI boundaries and optional pinpoints."""
    figure, axes = plt.subplots()
    axes.imshow(image, cmap="gray")
    for index, region in enumerate(regions, start=1):
        color = color_for_group(region.group_id)
        axes.plot(
            region.vertices[:, 0],
            region.vertices[:, 1],
            linewidth=1.8,
            color=color,
        )
        yy, xx = np.nonzero(region.mask)
        if len(xx):
            axes.text(
                xx.mean(),
                yy.mean(),
                f"{index} (G{region.group_id})",
                color=color,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

    if alignment_points is not None:
        plot_alignment_points(axes, alignment_points)
    axes.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="SEM image to measure.")
    parser.add_argument(
        "--mode",
        choices=("manual", "create-template", "apply-template"),
        default="manual",
        help="ROI selection workflow (default: manual).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Template to create or apply in template modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: directory containing the image).",
    )
    parser.add_argument(
        "--crop-rows-end",
        type=int,
        default=2047,
        help="Exclusive final row; use 0 to disable the crop (default: 2047).",
    )
    parser.add_argument(
        "--crop-cols-end",
        type=int,
        help="Exclusive final column (default: no column crop).",
    )
    parser.add_argument(
        "--alignment-points",
        type=int,
        default=4,
        help="Number of ordered points used for alignment (default: 4).",
    )
    parser.add_argument(
        "--transform",
        choices=("similarity", "affine", "projective"),
        default="projective",
        help="Geometric transform used when applying a template.",
    )
    parser.add_argument(
        "--thickness-map",
        help='Comma-separated group-to-thickness pairs, for example "1:33,2:95".',
    )
    parser.add_argument(
        "--pixel-size-nm",
        type=float,
        help="Pixel size used to calculate ROI area in square micrometres.",
    )
    parser.add_argument(
        "--background-region",
        type=int,
        help="One-based region ID whose mean and median are subtracted.",
    )
    return parser


def main() -> None:
    """Run the selected interactive ROI workflow and save measurements."""
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_rows_end = args.crop_rows_end or None
    image = load_grayscale_image(
        image_path,
        crop_rows_end=crop_rows_end,
        crop_cols_end=args.crop_cols_end,
    )
    print(f"Working image shape: {image.shape[1]} × {image.shape[0]} pixels")

    alignment_points = None
    if args.mode == "manual":
        regions = collect_regions(image)
    else:
        if args.template is None:
            raise ValueError("--template is required for template modes.")
        template_path = args.template.resolve()
        alignment_points = collect_alignment_points(
            image,
            count=args.alignment_points,
            mode=args.mode,
        )
        if args.mode == "create-template":
            regions = collect_regions(image, alignment_points)
            if regions:
                save_template(template_path, alignment_points, regions)
                print(f"Saved ROI template: {template_path}")
        else:
            regions = transform_template_regions(
                image,
                template_path=template_path,
                new_points=alignment_points,
                transform_type=args.transform,
            )

    if not regions:
        raise RuntimeError("No ROIs were available to measure.")

    measurements = measure_regions(image, regions)
    if args.background_region is not None:
        measurements = subtract_background(
            measurements,
            region_id=args.background_region,
        )
    measurements = add_metadata(
        measurements,
        thickness_by_group=parse_thickness_map(args.thickness_map),
        pixel_size_nm=args.pixel_size_nm,
    )
    group_summary = summarize_groups(measurements)

    stem = image_path.stem
    roi_path = output_dir / f"{stem}_se_intensity_per_roi.csv"
    group_path = output_dir / f"{stem}_se_intensity_group_summary.csv"
    overlay_path = output_dir / f"{stem}_se_roi_overlay.png"
    measurements.to_csv(roi_path, index=False)
    group_summary.to_csv(group_path, index=False)
    save_overlay(
        image,
        regions,
        alignment_points=alignment_points,
        path=overlay_path,
    )

    print(f"Saved per-ROI measurements: {roi_path}")
    print(f"Saved group summary: {group_path}")
    print(f"Saved ROI overlay: {overlay_path}")


if __name__ == "__main__":
    main()

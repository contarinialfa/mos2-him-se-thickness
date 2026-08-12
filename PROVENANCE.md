# Data and code provenance

## Image mapping

| Condition | Repository file | Original acquisition filename |
|---|---|---|
| Transmission, flake 1 / location 4 | `data/raw_images/transmission/flake_1_loc4.tif` | `20250814_loc4_topcoverremoverd_100Vbias_noflood_20umaper.tif` |
| Transmission, flake 2 / location 12 | `data/raw_images/transmission/flake_2_loc12.tif` | `20250814_loc12-1_topcoverremoverd_100Vbias_noflood_20umaper.tif` |
| Transmission, flake 3 / location 13 | `data/raw_images/transmission/flake_3_loc13.tif` | `20250814_loc13-1_topcoverremoverd_100Vbias_noflood_20umaper.tif` |
| Backside blocked, flake 1 / location 4 | `data/raw_images/backside_blocked/flake_1_loc4_backside_blocked.tif` | `20250806_loc4_backcovered_topcoverremoverd_samesettings.tif` |
| Backside blocked, flake 2 / location 12 | `data/raw_images/backside_blocked/flake_2_loc12_backside_blocked.tif` | `12.tif` in the 2025-08-06 backside-blocked/back-covered analysis folder |

Each image is a single-frame 2,048 × 2,240 RGBA TIFF. The ROI script analyzes
rows 0 through 2,046 by default, matching the original workflow's exclusion
of the instrument footer.

## ROI-table reconstruction

The raw exports in `data/roi_exports/` are direct copies of the per-ROI tables
created during image measurement. The rules in
`config/roi_preparation.json` explicitly record:

- the source region IDs or source group retained for each thickness;
- the final group number;
- the EELS thickness assigned to that group;
- the separate carbon-reference exports appended to transmission flakes 1
  and 2.

`scripts/prepare_roi_tables.py --check-only` reconstructs all five analysis
tables and numerically compares them with the preserved files in
`data/roi_measurements/`.

## Code ancestry

The repository scripts consolidate the final logic from the following working
copies without changing those originals:

- `extract_roi_intensity.py`: manual grouped-ROI measurement and later
  projective template-alignment versions of `ROI.py`;
- `prepare_roi_tables.py`: explicit reconstruction of the previously manual
  ROI selection, regrouping, and thickness-assignment step;
- `bayesian_normalize.py`: the final transmission and backside-blocked
  Bayesian normalization scripts;
- `fit_model_b.py`: the final frozen-effective-variance Model-B fit from the
  SRIM campaign analysis;
- `plot_model_fit.py`: the final APL-formatted data-plus-model plot.

The five external source images, working scripts, and every original file in
the parent project were treated as read-only. Only this new repository folder
contains the cleaned and reorganized copies.

## Copper-tape converter series

The tape-series images and ROI files under `data/tape_series/` are direct
copies of the acquisition-and-measurement folders for the three converter
configurations. Filenames follow `<set>-<config>` with set 1-3 the three
image sets and config `0` = no tape (control), `1tape-90deg` = small tape
(single strip), `2tape-90deg` = large tape (two adjacent strips):

| Repository folder | Configuration | Contents |
|---|---|---|
| `data/tape_series/control/` | No tape | `1-0.tif`, `2-0.tif`, `3-0.tif` with per-ROI tables and the shared ROI polygon sets |
| `data/tape_series/small_tape/` | Single strip, 90 deg | `1-1tape-90deg.tif`, `2-1tape-90deg.tif`, `3-1tape-90deg.tif` with ROI polygons, aligned polygons, per-ROI tables, and overlay images |
| `data/tape_series/large_tape/` | Two strips, 90 deg | `1-2tape-90deg.tif`, `2-2tape-90deg.tif`, `3-2tape-90deg.tif` with the corresponding ROI files |

The same polygon sets were reused across configurations (projectively aligned
per image, the `*_aligned.json` files), so the per-thickness points are
matched between configurations. ROI definition used `scripts/ROI.py`;
intensity measurement used `scripts/se_roi_measurement_optimized.py`. The
downstream matched-ratio statistics are in `analysis/tape/`.

## Four-configuration comparison (supplementary Fig. S3)

The four panels of supplementary Fig. S3 show the same flake (image set 1)
under identical detector settings in four configurations:

| Panel | Configuration | Repository file |
|---|---|---|
| Top left | Control, no tape | `data/tape_series/control/1-0.tif` |
| Top right | Backside blocked (adhesive side up) | `data/tape_series/backside_blocked/1-1tape-0deg.tif` |
| Bottom left | 45 deg direct-beam converter | `data/tape_series/45deg_converter/1-1tape-45deg.tif` |
| Bottom right | 90 deg small tape | `data/tape_series/small_tape/1-1tape-90deg.tif` |

The control and 90 deg panels reuse the tape-series files listed above; the
backside-blocked and 45 deg images are single-configuration acquisitions from
the same session (the `0deg`/`45deg` labels record the tape angle).

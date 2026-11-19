# Revised manuscript analysis (v1.1.0)

Release v1.0.0 of this repository preserves the analysis from the initial submission. This directory, added in v1.1.0,
contains the analysis scripts and outputs for the revised manuscript. All data and code files from v1.0.0 are
unchanged.

## Summary of additions

In the revised manuscript, the escape integrals for the three SE-escape models (symmetric, symmetric with a free escape
depth, and asymmetric) are evaluated using the finite-thickness SRIM ionization profile $D(z/s;\, t/s)$ for each
membrane thickness rather than a single thick-target (300 nm) profile across all thicknesses. The finite-thickness
profile multiplies the 300 nm profile by the exit-surface correction fitted to the nine finite-thickness SRIM runs
(10–200 nm) in `data/SRIM-Thickness-Series/` via `scripts/thickness_sweep_ionization.py`.

Additional scripts in this folder extract acquisition settings and stage coordinates from the Zeiss ORION TIFF metadata,
evaluate the transmitted-ion fraction and recoil-ionization share from the SRIM datasets, and run supplementary
sensitivity and covariance checks.

## Running the scripts

Install the dependencies in `revision/requirements.txt` (or the top-level `requirements.txt`) and run the scripts from
the repository root:

| Step | Command | Typical runtime |
|---|---|---|
| 1 | `python revision/scripts/refit_thickness_dependent_D.py` | ~6 min |
| 2 | `python revision/scripts/make_old_vs_new_table.py` | < 1 s |
| 3 | `python revision/scripts/verify_thickness_dependent_profile.py` | ~4 min |
| 4 | `python revision/scripts/fig3_model_comparison.py` | ~15 s |
| 5 | `python revision/scripts/within_image_covariance.py` | < 2 s |
| 6 | `python revision/scripts/refit_supplement.py` | ~2 min |
| 7 | `python revision/scripts/recoil_ionization_share.py` | < 1 s |
| 8 | `python revision/scripts/srim_transmitted_fraction.py` | < 1 s |
| 9 | `python -B revision/scripts/imaging_settings_and_positions.py` | < 1 s |
| 10 | `python -B revision/scripts/imaging_settings_and_positions.py --from-table revision/outputs/analysed_frames_metadata/frames.csv` | < 1 s |
| 11 | `python -B revision/scripts/detector_distance_bound.py` | < 1 s |

Steps 2–4 read `revision/outputs/refit_thickness_dependent_D/results.json` produced by step 1. Every script accepts
`--out <folder>` to write outputs to an alternate directory. To inspect the raw ORION XML metadata (TIFF tag 65000) of
an individual HIM image, run:

```
python -B revision/scripts/zeiss_metadata.py data/raw_images/transmission/flake_1_loc4.tif
```

## Map to manuscript figures, tables, and sections

| Manuscript section / figure / table | Script | Output directory / file |
|---|---|---|
| Main text & SI Sec. SVIII (Table S2): three-model fits with $D(z/s;\, t/s)$ ($\chi^2/\mathrm{dof} = 60.4/12$, $14.9/11$, $14.3/11$), fitted parameters, exit-side escape depth $\lambda_b = 10.2 \pm 1.2\text{ nm}$ ($10.2 \pm 2.0\text{--}2.4\text{ nm}$ including the 15–20% thickness-scale uncertainty), model peak positions (13 nm and 40.3 nm), and exit-to-blocked ratio (4.7) | `refit_thickness_dependent_D.py` | `outputs/refit_thickness_dependent_D/` |
| SI Sec. SVIII (Table S3): common-weighting $\chi^2$ and AIC, nested $\Delta\chi^2$ (41.3 and 43.1), entrance-surface contributions (0.321–0.330), sensitivity checks (scale, point thickness error, Flake 3, leave-one-out, entrance escape depth), and comparison against the thick-target profile | `refit_thickness_dependent_D.py`, `make_old_vs_new_table.py` | `outputs/refit_thickness_dependent_D/results.json`, `old_vs_new.md` |
| SI Sec. SVIII: validation of the finite-thickness correction against raw SRIM escape-weighted sums (<1% deviation vs. 4–17% for the uncorrected 300 nm profile), behavior for $t/s > 200\text{ nm}$, and quadrature/weighting convergence checks | `verify_thickness_dependent_profile.py` | `outputs/verify_thickness_dependent_profile/` |
| Main text, Fig. 3: model curves, common-weighting normalized residuals, and right-hand axis exit-surface ionization $D(t/s;\, t/s)/D_{\max}$ | `fig3_model_comparison.py` | `outputs/fig3_model_comparison/` |
| SI Sec. SVII: within-image posterior ratio correlations ($0.14\text{--}0.78$, median $0.39$) | `within_image_covariance.py` | `outputs/within_image_correlations/` |
| SI Secs. SVII–SIX: parameter correlation matrices, five-parameter fit, generalized least-squares refit under within-image posterior covariance, forward-warp deposition test, stretch range ($s = 0.62\text{--}0.73$), and falling-branch slope | `refit_supplement.py` | `outputs/refit_supplement/` |
| SI Sec. SVIII: recoil-ionization share of total SRIM ionization (1.8% over 0–60 nm) | `recoil_ionization_share.py` | `outputs/recoil_ionization_share/` |
| Main text & SI Secs. SII, SVII: transmitted-ion fraction (1–7% at $t = 186\text{--}212\text{ nm}$), thickness-axis conversions, and angular/geometric interception metrics | `srim_transmitted_fraction.py` | `outputs/srim_transmitted_fraction/` |
| Main text (Methods) & SI Sec. SI: line-scan acquisition settings ($2048 \times 2048\text{ px}$, $100\ \mu\text{s}$ dwell, 2-line averaging, $+500\text{ V}$ ET grid bias, $+100\text{ V}$ nominal sample bias), inter-flake separations ($0.10\text{--}0.22\text{ mm}$), working-distance range ($\le 0.20\text{ mm}$), acquisition time span (39 min), beam shift ($0$), and image shift ($\le 1\ \mu\text{m}$) | `imaging_settings_and_positions.py` | `outputs/imaging_settings_and_positions/`, `outputs/imaging_settings_and_positions_from_table/` |
| SI Sec. SI: upper bound on sample-to-detector distance variation across the Fig. 3 flakes ($\le 0.27\text{ mm}$, or 3% of 9 mm) | `detector_distance_bound.py` | `outputs/detector_distance_bound/` |
| SI Sec. SI: radial positions of the analysed fields on the 3.05 mm TEM grid ($\lesssim 0.5\text{ mm}$ from the grid centre) | `grid_positions.py` | `outputs/grid_positions/` |

The Fig. 3 data points and uncertainties are read from `data/reference_summaries/` (generated from
`analysis/bayesian_outputs/byasian2/`), unchanged from v1.0.0.

## Data notes

- `outputs/analysed_frames_metadata/frames.csv` tabulates the extracted ORION XML metadata for all 33 analysed HIM
  frames across the transmission, backside-blocked, and two copper-tape series. Sixteen of these raw TIFF frames are
  stored directly in `data/raw_images/` and `data/tape_series/` (analysed by step 9); step 10 evaluates the same
  summary across all 33 frames from `frames.csv`.
- `grid_positions.py` registers the HIM stage coordinates against the stereo-microscope optical location map and the
  full survey session archive. Because the survey-session raw TIFFs and optical grid maps are not stored in the
  repository, the derived coordinate tables (`analysed_fields_on_grid.csv`, `pairs_fig3.csv`, and
  `grid_position_check.txt`) are provided in `outputs/grid_positions/`. The additional raw survey frames are available
  from the corresponding author on reasonable request.
- `inputs/long_target_error_budget.csv` stores the reference parameter table from the v1.0.0 thick-target fit so that
  `refit_thickness_dependent_D.py` can verify consistency of the thick-target baseline alongside the
  thickness-dependent refit.

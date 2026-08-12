# Thickness-aware fits report: the paper models on the finite-thickness deposition profile

**Date:** 2026-07-29 · **Code:** `scripts/thickness_aware_fits.py` ·
**Figure:** `outputs/thickness_aware_model_comparison.png` · **Companions:**
`DL/06_thickness_sweep_report.md` (the deposition model),
`DL/07_paper_fit_report.md` (the paper fits this report stress-tests)

**Audience and purpose.** The paper fits (report 07) assume the deposition
profile in a t-nm flake is the first t nm of the profile in a very thick
target. Report 06 measured that assumption's error with a SRIM thickness
sweep and packaged the correction as a fitted model D(z; t). This report
integrates that correction into the paper fits and answers one question:
**does the semi-infinite assumption bias the paper's conclusion?** Answer:
no — removing the assumption makes the null *worse* and leaves the two
asymmetric models essentially unchanged, so the paper's claim is robust in
the conservative direction.

## 1. What changes in the model

Report 07's forward model evaluates the SE source from the semi-infinite
SRIM profile:

    S(z) = D(z / s)                        (baseline; report 07, Eq. 2)     (1)

Here the source instead uses the finite-thickness deposition model of
report 06, which multiplies the long-run profile by a fitted few-percent
correction surface, D(z; t) = R̃(t, z/t) · D_long(z). For a data point of
measured thickness x, both the depth *and* the film thickness map from the
measured axis to the SRIM axis through the same stretch:

    S(z; x) = D( z/s ; x/s )                                                (2)

Everything else — the physical escape kernels, the three models, the
iterative effective-variance weighting, the optimizer settings, bounds, and
starting points — is identical to report 07.

**Numerical consequence.** With Eq. (1), the deposition is the same
function for every data point, so all thicknesses can be evaluated from one
cumulative integral on a shared grid. With Eq. (2) the deposition differs
per data point, so each thickness x is integrated on its own [0, x] grid
(n = 2048; the per-cell kernels of report 07, Eqs. 11–15, are exact for any
grid spacing, so the finer/coarser dt is harmless). To make the comparison
airtight, the script runs the *baseline through the same per-point
machinery*: its results reproduce the production fits of report 07 to the
printed precision, so every difference below is deposition physics, not
numerics.

## 2. Results

All values in the physical kernel convention (report 07, section 10.1).
"Baseline" = semi-infinite D, identical to report 07 section 9;
"t-aware" = Eq. (2).

**Fit quality** (each fit under its own converged σ_eff; common-σ_eff
comparison in parentheses, using each variant's null weighting):

| model | k | baseline χ²/dof | t-aware χ²/dof | baseline AIC | t-aware AIC |
|---|---|---|---|---|---|
| Symmetric Escape (null) | 3 | 48.2 / 12 = 4.02 | 60.4 / 12 = 5.03 | 54.2 | 66.4 |
| Asymmetric Escape | 4 | 14.9 / 11 = 1.35 | 14.3 / 11 = 1.30 | 25.5 | 25.3 |
| Simplified Channelling | 4 | 17.4 / 11 = 1.58 | 17.2 / 11 = 1.57 | 26.8 | 26.9 |

Nested null → asymmetric-escape test (report 07, Eq. 21):
Δχ² = 33.3 baseline → **46.1** thickness-aware (1 dof; both ≫ 5σ).

**Parameters** (shift quoted in units of the baseline 1σ):

Asymmetric Escape:

| parameter | baseline | t-aware | shift/σ |
|---|---|---|---|
| C_o | 0.01114 ± 0.00033 | 0.01109 | −0.14 |
| C_fb | 0.649 ± 0.083 | 0.665 ± 0.078 | +0.19 |
| λ_b | 9.63 ± 1.28 nm | 10.17 ± 1.22 nm | +0.43 |
| s | 0.625 ± 0.034 | 0.628 | +0.10 |

Simplified Channelling:

| parameter | baseline | t-aware | shift/σ |
|---|---|---|---|
| A | 0.03445 ± 0.00065 | 0.03672 | +3.4 |
| s | 0.638 ± 0.030 | 0.639 | +0.04 |
| δ_∞ | 11.75 ± 1.96 nm | 12.47 ± 1.85 nm | +0.37 |
| ρ | 0.822 ± 0.027 | 0.841 | +0.70 |

Symmetric Escape (null): C_fb 2.99 → 3.32 (+2.1σ), s 0.701 → 0.734
(+1.1σ) — quoted for completeness only; the parameters of a failing model
carry no physical weight, and the shifts are the model contorting harder.

## 3. Reading

1. **The null gets decisively worse** (χ² 48.2 → 60.4; nested rejection
   strengthens from 33.3 to 46.1, i.e. 5.8σ → 6.8σ). Physics: the
   finite-thickness correction *lowers* deposition in thin films — the
   opposite of what the null needs, since it already underpredicts the thin
   side. Removing the semi-infinite assumption therefore cannot rescue the
   null; it buries it deeper.
2. **The asymmetric models are insensitive.** Every shape parameter moves
   by well under 1σ (λ_b +0.4σ, δ_∞ +0.4σ, ρ +0.7σ, C_fb +0.2σ, s ≤ 0.1σ),
   both fits improve marginally, and the split-vs-channelling ordering and
   ΔAIC are unchanged. The +3.4σ shift in the channelling amplitude A is a
   pure normalization effect (the corrected deposition is a few percent
   lower, so the amplitude rises ~6.5% to compensate) and carries no
   physical content.
3. **Systematic for the paper.** If the paper quotes parameter values, the
   thickness-aware shifts can be cited as the deposition-model systematic:
   ±0.5 nm on λ_b, ±0.7 nm on δ_∞, ±0.02 on C_fb — all comfortably inside
   the statistical errors.

![Thickness-aware fits](../outputs/thickness_aware_model_comparison.png)

The figure repeats the paper figure's layout with the thickness-aware fits;
compared with `outputs/paper_model_comparison.png`, the null's residual
swing widens (−5σ to +4σ) while the two asymmetric models are unchanged to
the eye.

## 4. Caveats

1. **The deposition correction is itself a fitted model** (report 06:
   five parameters, LOO-validated to rms ≲ 0.006 in the ratio, thickness
   clamped to the 10–300 nm data range). Its sub-percent uncertainty is
   negligible against the few-percent correction it applies, so no error
   was propagated from it into these fits.
2. **The stretch convention is a choice.** Both z and t pass through the
   stretch (Eq. 2) because the deposition model lives entirely on the SRIM
   axis. The alternative (thickness in measured nm) would mix axes; it was
   considered and rejected when the integration was first built.
3. **Below t ≈ 6.5 nm measured** (10 nm raw at s ≈ 0.65) the deposition
   model clamps to its thinnest-run shape. All data lie well above this.

## 5. How to reproduce

```
python scripts/thickness_aware_fits.py
```

builds the report-06 deposition model (printing its selection tables), runs
all six fits (three models × two deposition variants) through identical
per-point machinery, prints every number in section 2 including the
per-parameter shift table, and writes the figure. The deposition model
itself comes from `build_ionization_model` in
`DL/thickness_sweep_ionization.py`; the earlier exploration version of this
integration (unit-area kernels) is preserved as `fit_and_plot_models_r03`
in `scripts/david_model_exploration.py`.

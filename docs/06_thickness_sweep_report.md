# Thickness-sweep report: finite-thickness correction to the SRIM ionization profile

**Date:** 2026-07-28 · **Harness:** `docs/thickness_sweep_ionization.py` ·
**Data:** `data/SRIM-Thickness-Series/` + `data/SRIM 30 keV - Monolayer Long/` ·
**Outputs:** `outputs/thickness_sweep_*.png` (regenerated on every run) ·
**Companion to:** `docs/05_refined_model_report.md`

## 1. Purpose

Every SE-intensity model in this repository weights its escape integrals with the
SRIM ionization profile D(z) from the quasi-infinite "Monolayer Long" run — i.e. it
assumes the deposition profile in a t-nm flake is the first t nm of the profile in
a very thick target. The thickness sweep tests that assumption directly: nine
otherwise-identical SRIM runs with the target truncated at t = 10, 20, 30, 40, 50,
70, 100, 150, 200 nm.

The assumption fails at the few-percent level, in a systematic, thickness-dependent
way. This report documents the effect and the deliverable that captures it: a
five-parameter fitted model exposed as `ionization(z_array, thickness_nm)`, giving
a defensible ionization-vs-depth curve for an arbitrary flake thickness.

All numbers and figures are produced head-to-tail by

```
python docs/thickness_sweep_ionization.py
```

Figures use the repository style sheet `config/dlister_r01.mplstyle`.

## 2. Data

Each sweep run is 30 keV He into the MoS₂ layer (99,998 ions), with `IONIZ.txt`
reporting ionization by ions in 100 equal bins spanning the slab, so all runs share
the fractional-depth grid z/t = j/100. The Monolayer Long run is itself a 300 nm
slab (100 × 3 nm bins, 999,999 ions; full simulation conditions in report 07,
section 3) — it is the t = 300 nm member of the same family, and it has by far the
most statistical power per bin: 10× the ions and up to 30× more track length per
bin than the 10 nm run's 0.1 nm bins, ~300× combined.

**Boundary artifacts.** Every run's final bin collapses at the exit boundary
(e.g. 14 → 0.7 eV/Å in the last 0.1 nm of the 10 nm run) — a SRIM binning /
transmission bookkeeping artifact, not smooth physics. The plunge spans **two**
bins in the 10 nm run and one elsewhere. Handling: the final bin is always dropped,
and any remaining trailing bins with ratio < 0.8 (`ARTIFACT_RATIO_FLOOR`; observed
physical suppression never goes below ~0.92) are trimmed, with a note printed. The
gradual decline *approaching* the exit is retained — it is the physics being
modeled.

![Per-thickness comparison grid](../outputs/thickness_sweep_ionization.png)

## 3. The observed effect

![All runs overlaid](../outputs/thickness_sweep_ionization_overlay.png)

- **Finite thickness suppresses deposition through the whole film, not just near
  the exit.** The 10 nm run sits ~3.5% below the long profile *at the entrance
  surface*; the deficit grows toward the exit face (~7% at z → 10 nm).
- **The suppression shrinks with thickness**: each successive run steps up toward
  the long profile, and by 150–200 nm the curves track it closely until their own
  exit region.
- **The exit-face deficit grows with absolute depth**: ~7% for an exit at 10 nm but
  ~17% for an exit at 200 nm — slower ions near end-of-range scatter more, so more
  of their deposition depends on material beyond the boundary.

The physical picture: truncating the target removes the backscattered-ion flux
that material beyond the exit face would have returned into the film. The deficit
at depth z therefore depends mainly on the distance to the exit face u = t − z,
with an amplitude set by the local ion state (i.e. absolute z).

For the fits in `scripts/david_model_exploration.py` this means the long-profile
D(z) overestimates deposition by a few percent precisely in the thin-flake region
where the HIM data are densest — a t-dependent bias no current model includes.

## 4. Model

The returned curve is always the long profile times a small fitted correction —
the raw sweep numerics (per-bin MC noise σ ≈ 0.002 in ratio units) never enter
directly:

**D(z; t) = R(z, t) · D_long(z)**, with the deficit relative to a truly infinite
target

**d∞(z, t) = A(z) · K(t − z, z)**,  A(z) = A₀ + A₁z + A₂z²,

self-consistently normalized to the long run (a 300 nm slab, not an infinite one):

**R(z, t) = [1 − d∞(z, t)] / [1 − d∞(z, 300)]**  ⇒ R ≡ 1 at t = 300 by construction.

### 4.1 Kernel selection (data-driven)

Stage 1 fitted each run separately with d(u) = a·exp(−u/λ): amplitude a rises
smoothly with t (0.053 → 0.195), but the fitted λ rises to ~26 nm by 40 nm then
**falls to ~12 nm at 200 nm** — a depth-dependent decay length, not a second scale.
Stage 2 fitted four kernel families globally to all 890 retained bins, weighted by
each run's MC noise:

| kernel | form | k | χ²/dof | AIC |
|---|---|---|---|---|
| exponential | e^(−u/λ) | 4 | 11.31 | 10025.6 |
| power tail | (1 + u/λ)^(−p) | 5 | 11.52 | 10204.4 |
| double exponential | (1−w)e^(−u/λ₁) + w·e^(−u/λ₂) | 6 | 11.14 | 9860.5 (λ₂ pinned at bound) |
| **exponential, z-dependent λ** | e^(−u/λ(z)), λ(z) = λ₀e^(κz) | 5 | **4.11** | **3644.9** |

The z-dependent-λ kernel wins by ΔAIC ≈ 6200 with one *fewer* parameter than the
runner-up, and it is the form the stage-1 λ trend independently suggested.

### 4.2 Fitted parameters (weighted least squares, errors from J^T J)

| parameter | value | 1σ | meaning |
|---|---|---|---|
| A₀ | 3.934×10⁻² | 1.7×10⁻⁴ | deficit amplitude at the surface |
| A₁ | 1.232×10⁻³ /nm | 9.5×10⁻⁶ | amplitude growth with depth |
| A₂ | −1.758×10⁻⁶ /nm² | 7.5×10⁻⁸ | mild saturation of the growth |
| λ₀ | 39.06 nm | 0.29 | backscatter return length at the surface |
| κ | −6.489×10⁻³ /nm | 7.8×10⁻⁵ | λ(200) ≈ 10.7 nm — slower ions return over shorter distances |

![Ratio fits and interpolated family](../outputs/thickness_sweep_interpolation.png)

## 5. Validation

**Per-run agreement** (global model vs the fit points; rms includes the MC noise
σ listed):

| t (nm) | rms | max dev | σ (MC) |
|---|---|---|---|
| 10 | 0.0043 | 0.0185 | 0.0015 |
| 20 | 0.0034 | 0.0091 | 0.0022 |
| 30 | 0.0031 | 0.0084 | 0.0018 |
| 40 | 0.0037 | 0.0112 | 0.0024 |
| 50 | 0.0031 | 0.0090 | 0.0022 |
| 70 | 0.0028 | 0.0071 | 0.0019 |
| 100 | 0.0039 | 0.0088 | 0.0017 |
| 150 | 0.0047 | 0.0119 | 0.0017 |
| 200 | 0.0041 | 0.0099 | 0.0022 |

**Leave-one-out** (the five-parameter model refitted without each interior run,
then predicting it): rms 0.0028–0.0055, essentially identical to the in-sample
agreement — the form generalizes rather than memorizing runs. Worst case is 150 nm
(rms 0.0055, max 0.0137).

**Sanity:** R(300) ≡ 1 exactly; body-mean R(10) = 0.961; thickness clamped to
[10, 300] nm.

Since R is a few-percent correction, rms 0.004 in R is a **sub-0.5% error on any
returned D(z; t) curve** — comfortably below the effect being corrected.

## 6. Deliverable and usage

```python
import sys
sys.path.insert(0, "scripts")  # the scripts folder is not a package

from thickness_sweep_ionization import (
    load_ionization, discover_sweep, build_ionization_model,
    DEFAULT_LONG_IONIZ, DEFAULT_SWEEP_DIR,
)

long_depth, long_ioniz = load_ionization(DEFAULT_LONG_IONIZ)
sweep = [(t, *load_ionization(p)) for t, p in discover_sweep(DEFAULT_SWEEP_DIR)]
ionization, model = build_ionization_model(long_depth, long_ioniz, sweep)

d = ionization(z_array, thickness_nm)   # eV/Å at depths z for a t-nm flake
```

Semantics: z spans the requested film and D = 0 outside [0, t]; the thickness
argument is clamped **fraction-preservingly** to [10, 300] nm (a t = 5 nm query
uses the 10 nm suppression shape mapped onto the 5 nm film — a conservative choice,
no invented trend below the data). `model` carries the fitted θ, per-run tables,
raw ratio grid, and `ratio_fn(z, t)` for diagnostics.

## 7. Caveats and next steps

1. **χ²/dof = 4.1**: residual systematics of ~2× the MC noise remain — the
   separable A(z)·K(u, z) form slightly smooths the 10 nm run's entrance bins
   (max dev 0.019). Deliberately left rather than adding parameters; revisit only
   if a fit downstream becomes sensitive at the 0.5% level.
2. **Below 10 nm the shape is clamped**, and the measured flakes reach ~3 nm. The
   fitted form *could* extrapolate honestly (it is smooth in t), but that is a
   physics decision not yet taken.
3. **Method history** (for defensibility): a spline surface over the ratio grid
   (per-row smoothing + PCHIP in log t) was built first; its LOO error was
   comparable (rms 0.003–0.009) but it imprinted thin-run MC noise on the returned
   curves and rang at grid edges before artifact-trim fixes. The parametric model
   replaced it at equal-or-better LOO accuracy with 5 parameters instead of ~900
   grid values.
4. **Next step (not yet done):** wire `ionization(z, t)` into
   `fit_and_plot_models` in `scripts/david_model_exploration.py` as a
   thickness-aware replacement for `ionization_interp(z/stretch)`. The interaction
   between the stretch calibration and the physical thickness argument needs its
   own decision before this swap.

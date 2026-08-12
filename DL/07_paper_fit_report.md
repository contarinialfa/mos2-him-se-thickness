# [ARCHIVED] Paper-figure fit report: models, mathematics, and fitting procedure

> **ARCHIVED 2026-07-29 — superseded by `DL/09_paper_fit_report_v2.md`.**
> The paper model set changed: the free-λ symmetric model (discovered via
> `fit_and_plot_models_r06`; section 12 item 1 here) is now presented as a
> third fitted model rather than a caveat, and the figure shows three
> curves. This document remains the record of the channelling model
> analysis (sections 4, 9, 10), the C_fb kernel-convention derivation
> (section 10.1), and the C_fb transplant/projection studies (sections
> 10.2–10.3), which report 09 references rather than repeats.

**Date:** 2026-07-28 · **Code:** `scripts/david_model_exploration.py` ·
**Figure:** `outputs/paper_model_comparison.pdf` · **Companions:**
`DL/05_refined_model_report.md` (model campaign),
`DL/06_thickness_sweep_report.md` (finite-thickness deposition)

**Audience.** A physicist who does not work with fitting or statistics daily.
The language stays plain, but every equation needed to reimplement the fits
appears explicitly, and every statistical step names the standard concept it
uses so you can look it up independently.

**The claim the figure supports.** The naive symmetric model (the null)
fails to describe the data. Two simple asymmetric extensions succeed. The
published figure shows the null against one of them (the asymmetric-escape
model, legend "Symmetric" vs "Asymmetric"); the channelling alternative is
omitted from the figure because section 10.3 shows it observationally
degenerate with the asymmetric-escape curve on these data — plotting it
adds no information. It remains fully analyzed here, and we make no claim
about which asymmetry mechanism is correct.

## 1. Notation (one meaning per symbol)

| symbol | meaning | units |
|---|---|---|
| t | flake thickness, from EELS | nm (measured axis) |
| z | depth below the entrance (beam-side) surface | nm (measured axis) |
| ζ = z/s | depth on the SRIM axis | nm (SRIM axis) |
| D(ζ) | energy deposition profile of the 30 keV He beam, from SRIM | eV/Å |
| s | stretch: depth-axis calibration between EELS and SRIM | — |
| λ | an SE escape depth (λ_esc when fixed, λ_b when fitted) | nm (measured axis) |
| I_n | normalized SE transmission intensity | — |
| y_i, σ_i | measured intensity and its 1σ error at thickness t_i | — |

The "front" surface is the entrance surface. The "back" surface is the exit
surface. These words keep these meanings through the whole report.

## 2. The data

`data/reference_summaries/transmission_bayesian_summary.csv` contains N = 15
points from 3 suspended MoS₂ flakes, thicknesses 14–212 nm. Each point has a
normalized intensity y_i and an intensity error σ_i, the symmetric average
of the 68% Bayesian posterior interval (look up: *credible interval*):

    σ_i = ½ (σ_lower,68,i + σ_upper,68,i)                                (1)

Thickness has its own uncertainty, taken as 10% of t (repository convention
`THICKNESS_FRACTIONAL_ERROR = 0.10`). Section 6 shows exactly how it enters
the fit. The loader sorts points by thickness and pools all three flakes;
no point is excluded or reweighted by flake.

## 3. The shared forward model

Every model computes I_n(t) from the same ingredients. The SE source at
measured depth z is proportional to the SRIM deposition evaluated on the
SRIM axis:

    S(z) = D(z / s)                                                      (2)

D comes from `data/SRIM 30 keV - Monolayer Long/IONIZ.txt` (100 points,
3–300 nm), interpolated with a cubic spline and clamped to its end values
outside the table. The stretch s is a fitted calibration between the EELS
thickness axis and the SRIM depth axis.

**SRIM simulation conditions (Monolayer Long run**, from the run's
`TDATA.txt`**).**

- Code: SRIM-2013.00 (TRIM damage calculation).
- Ion: He (Z = 2, mass 4.003 u), 30 keV, normal incidence
  ("Ion Angle to Surface = 0°").
- Target: one homogeneous layer, 300 nm deep (equal to the tabulation
  range); stoichiometry Mo 33.3 at% / S 66.6 at%; density
  5.710×10²² atoms/cm³ = 5.06 g/cm³.
- Damage energies (Mo / S): displacement 25 / 25 eV, lattice binding
  3 / 3 eV, surface binding 6.83 / 2.88 eV.
- Statistics: 999,999 ions. 4.43% backscattered; 0.55% transmitted through
  the 300 nm slab. Mean projected range 166.7 nm, straggle 63.1 nm (SRIM
  axis).
- Output used: the "IONIZ. by IONS" column of `IONIZ.txt` — 100 bins of
  3 nm, units eV/(Å·ion). The recoil-ionization column (~1.5% of the ion
  column) is not used.
- File-header caveat: the first line of every output reads
  "H (10) into Layer 1" — a stale setup label; the ion record
  (Ion = He, 30 keV) governs.

The thickness-sweep runs behind reports 06 and 08 use identical beam and
target settings with the layer truncated at 10–200 nm and ~10⁵ ions per
run; the long run's 10× ion count and 30× wider bins are why it carries
the statistical weight in the D(z; t) construction (report 06).

An SE born at depth z escapes a surface with probability that decays
exponentially with the distance to that surface (look up: *secondary
electron escape depth*); an SE born *at* a surface has escape odds set by
surface physics alone, independent of λ. The two escape yields for a film
of thickness t are

    Y_f(t; λ) = ∫₀ᵗ S(z) e^(−z/λ) dz          (front / entrance)          (3)

    Y_b(t; λ) = ∫₀ᵗ S(z) e^(−(t−z)/λ) dz      (back / exit)               (4)

Each depth slice contributes independently, so nothing normalizes the
kernel, and the emitted count grows with λ (section 10.1 spells this out).
A second, *unit-area* convention divides each kernel by λ; it was used
during exploration because it decorrelates λ from the amplitude parameters
during optimization. The two conventions are pure reparameterizations of
each other — identical curves and χ², amplitudes mapping as C → C·λ. The
production fit uses the physical form above.

The detector collects the two surfaces' contributions with different
efficiencies, so the general intensity model is

    I_n(t) = C_o [ Y_f(t; λ_front) + C_fb · Y_b(t; λ_back) ]              (5)

C_o sets the overall scale; C_fb weights the back channel relative to the
front. What C_fb *means* physically depends on the kernel convention —
section 10 treats this carefully.

**The escape depth is an input, not a fit parameter.** All models fix the
front escape depth at λ_esc = 2 nm. Two reasons. First, independent physics
puts the SE escape depth near 2 nm. Second, the data start at t ≈ 14 nm and
cannot constrain a length much shorter than that; a parameter the data
cannot constrain destabilizes a fit (look up: *parameter identifiability*).
The planned systematic is to repeat everything at λ_esc = 1.5 and 2.5 nm.

## 4. The three models

**Symmetric Escape (null).** One shared escape depth for both surfaces.
Free parameters p = (C_o, C_fb, s), k = 3:

    I_null(t) = C_o [ Y_f(t; λ_esc) + C_fb · Y_b(t; λ_esc) ] ,  λ_esc = 2 nm   (6)

C_fb stays free even in the null: detector geometry alone makes the two
collection efficiencies differ. The null hypothesis is symmetric
*transport* (equal λ), not equal collection.

**Asymmetric Escape.** The back surface gets its own escape depth; the
front keeps λ_esc = 2 nm. Free parameters p = (C_o, C_fb, λ_b, s), k = 4:

    I_asym(t) = C_o [ Y_f(t; λ_esc) + C_fb · Y_b(t; λ_b) ]                (7)

Physical picture: the beam drives delta electrons forward, so effective SE
transport toward the exit surface can differ from transport toward the
entrance. Setting λ_b = λ_esc recovers Eq. (6) exactly — the null is
*nested* inside this model, which enables the cleanest comparison test
(section 8).

**Simplified Channelling.** Escape stays symmetric and equal-weight
(λ_esc both sides, single amplitude A, no C_fb — section 10.3 justifies
this); instead the deposition is warped forward on the SRIM axis. Define,
with λ_d = δ_∞/ρ,

    δ(ζ) = δ_∞ (1 − e^(−ζ/λ_d))                                          (8)

    D̃(ζ) = (1 − ρ e^(−ζ/λ_d)) · D(ζ − δ(ζ))                              (9)

The prefactor in Eq. (9) equals d(ζ − δ(ζ))/dζ, the Jacobian of the depth
remap, so the warp conserves deposited energy: it moves energy deeper by up
to δ_∞ without creating or destroying any. ρ < 1 keeps the remap monotone.
The model is

    I_chan(t) = A [ Y_f(t; λ_esc) + Y_b(t; λ_esc) ]  with  S(z) = D̃(z/s)  (10)

Free parameters p = (A, s, δ_∞, ρ), k = 4. We fit (δ_∞, ρ) rather than
(ρ, λ_d) because the data constrain δ_∞ well and λ_d poorly; the
reparameterization removes a near-unidentifiable direction.

**Fit setup per model** (bounds keep the optimizer out of unphysical or
overflow-prone regions; no best fit ended on a bound):

| model | parameters p | initial guess | lower bounds | upper bounds |
|---|---|---|---|---|
| null | C_o, C_fb, s | 0.01, 3.0, 0.68 | 0, 0, 0.2 | ∞, ∞, 3 |
| asym. escape | C_o, C_fb, λ_b, s | null fit + λ_b = 2 | 0, 0, 0.2, 0.2 | ∞, ∞, 50, 3 |
| channelling | A, s, δ_∞, ρ | 0.03, s_null, 6.0, 0.7 | 0, 0.2, 0, 0.05 | ∞, 3, 30, 0.95 |

The two alternative models start from the converged null fit (warm start),
which speeds convergence and makes the nested comparison well-posed.

## 5. Numerical evaluation (exact per-cell kernels)

The integrals (3)–(4) have no closed form, so they are computed on a
uniform grid: n = 8192 points t_j = j·Δ spanning [0, max t], where max t is
the largest thickness in the evaluation request (212 nm during fitting;
1.05 × 212 nm for the plotted curves). Source samples S_j = S(t_j) come
from Eq. (2) (or Eq. (9) for channelling).
Between grid points S is treated as linear; for a linear function times an
exponential every cell integral has a closed form, so the only
discretization error is the piecewise-linear representation of S itself.

Define the per-cell constants

    a = Δ/λ ,   e = e^(−a) ,   φ = (1 − e)/a ,   ψ = (φ − e)/a           (11)

(φ and ψ are the exact cell integrals of the kernel against the constant
and linear parts: ∫₀^Δ e^(−u/λ) du = Δφ and ∫₀^Δ (u/Δ) e^(−u/λ) du = Δψ.)

**Front integral.** The cell [t_j, t_{j+1}] contributes, exactly,

    F_j = e^(−t_j/λ) · Δ [ (φ − ψ) S_j + ψ S_{j+1} ]                      (12)

and the cumulative sum gives the front yield at every grid point:

    Y_f(t_k; λ) = Σ_{j<k} F_j                                             (13)

**Back integral.** The exit-weighted integral obeys an exact one-step
recursion, because moving the upper limit by one cell multiplies all
earlier contributions by e^(−Δ/λ):

    B_0 = 0 ;   B_k = e · B_{k−1} + Δ [ (φ − ψ) S_k + ψ S_{k−1} ]         (14)

    Y_b(t_k; λ) = B_k                                                     (15)

The code evaluates Eq. (14) with `scipy.signal.lfilter([c1, c0], [1, −e],
S, zi=[−c1·S_0])`, where c1 = Δ(φ−ψ) and c0 = Δψ (written as
`c1 = Δ(1−φ)/a`, `c0 = Δ(φ−(1−φ)/a)` in the code; the identities
(1−φ)/a = φ−ψ and φ−(1−φ)/a = ψ connect the forms). The `zi` term sets
B_0 = 0. These formulas are exact for any ratio Δ/λ.

The model is evaluated once on the grid and linearly interpolated to the
requested thicknesses.

## 6. The fit

**The objective.** For a trial parameter set p, the weighted residual of
point i is

    r_i(p) = [ M(t_i; p) − y_i ] / σ_eff,i                               (16)

and the fit minimizes

    χ²(p) = Σ_i r_i(p)²                                                   (17)

Dividing by σ gives precise points more weight; for Gaussian errors this
*weighted least squares* is identical to maximum likelihood (look up:
*weighted least squares*, *chi-squared*). No prior terms are used: the
fixed λ_esc replaced the Gaussian prior on λ that earlier exploration
revisions carried.

**Thickness errors via effective variance.** The 10% thickness error is an
x-axis error, which plain least squares cannot represent. The standard
*effective variance* approximation converts it to an equivalent y error
through the local slope of the model (look up: *errors-in-variables*):

    σ_eff,i² = σ_i² + [ M′(t_i) · 0.10 · t_i ]²                           (18)

where M′(t) is the slope of the *fitted* curve. Where the curve is flat
(the plateau beyond ~180 nm) the thickness error adds nothing; where it is
steep (below ~30 nm) it dominates.

**Iteration.** Eq. (18) needs the fitted slope, and the fit needs the
weights, so the routine `wfit` iterates three times: (i) minimize Eq. (17)
with the current σ_eff (first pass: σ_eff = σ); (ii) evaluate the fitted
curve on a dense axis of 500 points spanning [0, 300 nm] (the SRIM profile
range), take its numerical gradient (`np.gradient`, second-order central
differences), and linearly interpolate the slope to each t_i; (iii) update
σ_eff by Eq. (18) and restart the optimizer from the current best p. Three
passes converge at the printed precision. The σ_eff frozen after the null
fit's final pass is saved for the comparison table (section 8).

**Optimizer.** `scipy.optimize.least_squares` with the trust-region
reflective method (the default with bounds; look up: *trust-region
reflective algorithm*), bounds from the table in section 4,
`x_scale='jac'` (automatic parameter rescaling by Jacobian column norms, so
magnitudes 0.02 and 10 condition equally), and `ftol = xtol = 1e-12`.
After the final pass the code checks each parameter against its bounds and
warns if any sits there, because Eq. (19) is invalid at a bound. No
parameter hit a bound in the three headline fits.

## 7. Parameter uncertainties

Near the minimum χ²(p) is approximately quadratic, and its curvature gives
the uncertainties. With J the Jacobian J_ij = ∂r_i/∂p_j at the best fit,

    C = (Jᵀ J)⁻¹                                                          (19)

    σ(p_j) = √(C_jj) ,   corr(p_j, p_k) = C_jk / √(C_jj C_kk)             (20)

(look up: *Gauss-Newton approximation*, *covariance matrix*). Because the
residuals already contain σ_eff, C needs no further scaling; the errors are
meaningful to the extent the section-6 error model is trusted.

Strong correlation (|corr| → 1) means the data constrain a combination of
parameters, not each separately. Example: corr(C_o, C_fb) = −0.98 in the
null fit, because the plateau intensity fixes roughly their product. This
is expected and harmless for the model comparison, which rests on curve
shape. One warning that matters: linearized errors from Eq. (19) can badly
understate how fast χ² really rises far from the minimum when the problem
is near-degenerate — section 10.3 shows a concrete case.

## 8. How the models are compared

**Goodness of fit.** χ²/dof with dof = N − k should be near 1 for a model
that describes the data within errors (look up: *reduced chi-squared*).
A value of 4 means the typical point sits two error bars from the curve.
The quantitative version is the tail probability
p = P(χ²_dof ≥ χ²_observed) — the chance of a fit this bad or worse if the
model and error model are correct (look up: *chi-squared goodness-of-fit
test*). This is an *absolute* test: it can reject a model without
reference to any alternative.

**The nested test (the decisive one).** The null is Eq. (7) with λ_b
pinned at 2 nm — nested, with exactly one extra degree of freedom. Under
the null hypothesis the improvement

    Δχ² = χ²_null − χ²_asym                                               (21)

follows a χ² distribution with 1 dof (look up: *likelihood-ratio test*,
*Wilks' theorem*). Observed: Δχ² = 33.3 with each fit under its own
converged σ_eff, or 30.7 under the common weighting below — 5.8σ or 5.5σ
(p ≈ 10⁻⁸). Either reading rejects the null decisively. This is the single
strongest statement the figure makes.

**The non-nested comparison.** Channelling touches the null only at the
boundary δ_∞ → 0, where ρ becomes meaningless, so Wilks' theorem does not
apply cleanly (look up: *testing on the boundary of the parameter space*).
For that comparison we report

    AIC = χ² + 2k                                                         (22)

(lower is better; differences above ~10 are decisive; look up: *Akaike
information criterion*).

**One common weighting for the table.** χ² values under different weights
are not comparable, and each model's wfit converges to its own σ_eff. The
comparison table therefore evaluates all three best-fit models under the
*null fit's* converged σ_eff — a choice that, if anything, favors the null:

    χ²_common(model) = Σ_i [ (M(t_i; p̂) − y_i) / σ_eff,i^(null) ]²        (23)

## 9. Results

Per-model fits (each under its own converged σ_eff):

| model | k | χ² | dof | χ²/dof | GoF p | key parameters |
|---|---|---|---|---|---|---|
| Symmetric Escape (null) | 3 | 48.2 | 12 | 4.02 | 2.9×10⁻⁶ | C_fb = 2.99 ± 0.16 |
| Asymmetric Escape | 4 | 14.9 | 11 | 1.35 | 0.19 | λ_b = 9.6 ± 1.3 nm, C_fb = 3.12 ± 0.13 |
| Simplified Channelling | 4 | 17.4 | 11 | 1.58 | 0.10 | δ_∞ = 11.7 ± 2.0 nm (SRIM axis), ρ = 0.82 ± 0.03 |

Comparison under the common σ_eff (Eqs. 22–23):

| model | χ² | k | dof | χ²/dof | AIC |
|---|---|---|---|---|---|
| Symmetric Escape (null) | 48.2 | 3 | 12 | 4.02 | 54.2 |
| Asymmetric Escape | 17.5 | 4 | 11 | 1.59 | 25.5 |
| Simplified Channelling | 18.8 | 4 | 11 | 1.71 | 26.8 |

Reading, in order of strength:

1. The null fails twice over: on the *absolute* goodness-of-fit test
   (p = 2.9×10⁻⁶ — a fit this bad happens ~3 times in a million if the
   model were right, with no alternative model invoked), and at 5.8σ on
   the nested test against the asymmetric extension.
2. Either single asymmetry repairs the fit: both extensions reach
   χ²/dof ≈ 1.4–1.7 with acceptable absolute goodness of fit (p = 0.19
   and 0.10) and sit ~28 AIC units below the null.
3. The two extensions are statistically indistinguishable (ΔAIC = 1.3;
   section 10.3 quantifies just how indistinguishable). The paper must not
   claim one over the other.

The fitted scales are physically reasonable. λ_b ≈ 10 nm exceeds the 2 nm
entrance-side escape depth, consistent with forward-directed delta
electrons. δ_∞ ≈ 12 nm on the SRIM axis (≈ s·δ_∞ ≈ 7.5 nm measured) is a
plausible channelling-relocation depth. The stretch lands at s ≈ 0.62–0.70
in all fits, consistent with the campaign fits (report 05).

### Full fit parameters and correlations

All fits converged on the optimizer's tolerance criteria with no parameter
on a bound, so the Eq. (19) covariances are valid. Amplitude-type
parameters are listed in both kernel conventions (section 10.1); shape
parameters (s, λ_b, δ_∞, ρ) are identical in the two conventions. The
production fit works in the physical convention; the unit-area values are
given for cross-reference with the exploration history.

**Symmetric Escape (null)** — χ² = 48.2, dof = 12, χ²/ν = 4.02:

| parameter | physical convention | unit-area convention |
|---|---|---|
| C_o | 0.01104 ± 0.00044 | 0.0221 ± 0.0009 |
| C_fb | 2.99 ± 0.16 (λ ratio = 1) | 2.99 ± 0.16 |
| s | 0.701 ± 0.030 | same |

| corr | C_o | C_fb | s |
|---|---|---|---|
| C_o | 1 | −0.98 | −0.40 |
| C_fb | | 1 | +0.35 |
| s | | | 1 |

**Asymmetric Escape** — χ² = 14.9, dof = 11, χ²/ν = 1.35:

| parameter | physical convention | unit-area convention |
|---|---|---|
| C_o | 0.01114 ± 0.00033 | 0.0223 ± 0.0007 |
| C_fb | **0.649 ± 0.083** | 3.12 ± 0.13 |
| λ_b | 9.63 ± 1.28 nm | same |
| s | 0.625 ± 0.034 | same |

Derived: λ_b − λ_esc = 7.63 ± 1.28 nm; nested Δχ² vs null = 33.3 (1 dof).

| corr (physical) | C_o | C_fb | λ_b | s |
|---|---|---|---|---|
| C_o | 1 | −0.37 | +0.07 | −0.31 |
| C_fb | | 1 | −0.95 | +0.54 |
| λ_b | | | 1 | −0.50 |
| s | | | | 1 |

(corr(C_fb, λ_b) = −0.95 because in the physical convention the two
parameters jointly set the back-channel strength; in the unit-area
convention it is +0.29 — the decorrelation that made unit-area kernels the
exploration convention. Same fit, different chart of the same surface.)

**Simplified Channelling** — χ² = 17.4, dof = 11, χ²/ν = 1.58:

| parameter | physical convention | unit-area convention |
|---|---|---|
| A | 0.03445 ± 0.00065 | 0.0689 ± 0.0013 |
| s | 0.638 ± 0.030 | same |
| δ_∞ | 11.75 ± 1.96 nm (SRIM axis) | same |
| ρ | 0.822 ± 0.027 | same |

Derived: λ_d = δ_∞/ρ = 14.3 nm (SRIM axis); measured-axis shift
s·δ_∞ = 7.50 ± 1.06 nm (error includes the s–δ_∞ covariance).

| corr | A | s | δ_∞ | ρ |
|---|---|---|---|---|
| A | 1 | −0.27 | +0.69 | −0.16 |
| s | | 1 | −0.62 | +0.46 |
| δ_∞ | | | 1 | −0.76 |
| ρ | | | | 1 |

## 10. What C_fb means — conventions, transplants, and an effective value

This section collects everything about the back-channel weight in one
place, because it is the easiest parameter to misread.

### 10.1 Kernel normalization: why the unit-area convention reports C_fb ≈ 3

Geometrically the back surface should collect *worse* than the front
(C_fb < 1), and the production fit indeed finds C_fb = 0.649. The
exploration fits (unit-area kernels, r02) reported ≈ 3.1 for the same fit —
an alarming-looking number that is a bookkeeping consequence of that
convention. The unit-area kernel is normalized over escape
*distance*, but the yield integral runs over generation *depth* weighted by
D — so the normalized integral is a weighted *average* of D over the escape
zone, not a *total*. The physical yield is a total: each slice dz generates
D(z)dz/ε secondaries (ε = energy per SE), and each escapes with probability
∝ e^(−d/λ) — a per-slice probability with no normalization, because
independent sources share no budget. Summing slices gives
γ ∝ ∫ D(z) e^(−d/λ) dz with no 1/λ, so the physical yield grows with λ
(for uniform D, doubling λ doubles the escaping count while the unit-area
Y stays fixed). The unit-area convention therefore parks that factor in the
amplitude, and the fitted ratio is the product

    C_fb(fitted, r02) = (collection ratio) × (λ_b/λ_f) × (generation asym.)   (24)

With λ_b/λ_f = 9.63/2 = 4.81 from the asymmetric-escape fit, the physical
back/front factor is 3.12/4.81 = 0.65 — and the production fit, working
directly with the physical kernels, measures it as

    C_fb(physical) = 0.649 ± 0.083                                        (25)

(`fit_and_plot_models_r05` first established that the two conventions are a
pure reparameterization: identical χ² and curves.) The value is below 1, as
geometry demands, and is an *upper bound* on the pure collection penalty,
since it still contains any exit-side generation enhancement from
forward-peaked delta electrons. The two conventions split the labor: the
unit-area form decorrelates amplitude from λ during fitting
(corr(C_fb, λ_b) = +0.29 there vs −0.95 in the physical form) and served
the exploration; the physical form gives the parameter its meaning and is
what the production fit and the paper use.

The null's C_fb = 2.99 admits no such decomposition (its λ ratio is 1): a
symmetric-transport model can match the data's plateau only by claiming a
3× back-collection advantage that geometry forbids. That is one more face
of the null's failure.

### 10.2 Does the channelling model need a C_fb? (No — and it cannot use one)

Eq. (10) weights the two surfaces equally while the other models carry a
free collection ratio. Three checks justify this (all reproducible with
`fit_and_plot_models_r04`):

- **Freeing C_fb is statistically empty.** I = A[Y_f + C_fb·Y_b] improves
  χ² by 0.28 for one added parameter, with a shallow valley at C_fb ≈ 1.6
  and linearized correlations |corr(C_fb, A)|, |corr(C_fb, ρ)| ≈ 1.00. The
  data cannot split the front/back imbalance between a collection ratio and
  a deposition warp. (δ_∞ is robust to the choice: 11.7 ± 2.0 equal-weight
  vs 10.3 ± 2.3 free.)
- **Transplanting the r02-convention "3" is invalid and kills the model.**
  That number bundles the split's λ_b/λ_f (Eq. 24). Pinned there, the warp
  dies (δ_∞ → 0), the fit collapses onto the null (χ² = 48.3–50.0), and the
  model falls Δχ² ≈ +33 behind the split at equal k. This also shows the
  free-C_fb valley is strongly non-parabolic: moving C_fb from 1.6 to 3.0
  costs ~31 units where the linearized ±0.7 predicted ~4 (the section-7
  warning in action).
- **Transplanting the physical ratio 0.649 ± 0.083 is legitimate — and the
  model survives it, strained.** χ² = 19.1 (+4.2 vs the split at equal k),
  with δ_∞ = 14.5 ± 2.2 nm and ρ pinned at its 0.95 monotonicity bound.
  Sensitivity across the ratio's 1σ is strong: χ² = 17.0 at 0.73 (ρ = 0.94,
  unpinned) but 27.7 at 0.57 (pinned). The warp can coexist with the
  split's collection ratio only by running at the edge of its allowed
  parameter space.

### 10.3 The effective C_fb of the warp

Since the published channelling form has no C_fb, one can instead *derive*
the back/front weighting its warp effectively produces. Two definitions,
both computed in `fit_and_plot_models_r04`:

**Projection (the single quotable number).** Evaluate the fitted
channelling model, noise-free, at the 15 measured thicknesses; fit the
asymmetric-escape family to that curve. The best fit lands at

    C_fb,eff = 0.650 (physical convention),  λ_b,eff = 9.8 nm,  s = 0.635  (26)

— numerically the split's own real-data fit (0.649, 9.63, 0.625). The
maximum distance between the two families over the measured range is
0.005 in I_n, one fifth of a typical error bar; the expected ability of any
dataset with this design to prefer one family over the other is
Δχ² ≈ Σ(mismatch/σ)² ≈ 0.5. **This is the precise sense in which the two
asymmetric models are observationally degenerate**, and it is why the
ΔAIC = 1.3 in section 9 must not be over-read.

**Exact equivalence function (the fingerprint).** Relative to the unwarped
equal-λ baseline, the warp rebalances the two channels by

    C_eff(t) = [Y_b^warp(t)/Y_f^warp(t)] / [Y_b⁰(t)/Y_f⁰(t)]              (27)

(a ratio of ratios: amplitudes and kernel conventions cancel). The
decomposition is physically transparent: the warp suppresses front emission
by a constant ×0.33 (near-surface deposition evacuated; the front only sees
the first ~λ), and rescales back emission from ×0.80 at t = 15 nm to ×1.35
at 200 nm (the shift hands the exit surface deposition from δ_∞ shallower,
which matters more as the profile steepens). C_eff(t) therefore runs from
≈2.0 to ≈4.4 across the data — *no constant C_fb reproduces the warp
exactly*. The split family absorbs this thickness dependence through its
free λ_b, which is how Eq. (26) can hold to σ/5 anyway. The residual
t-dependence is the in-principle fingerprint that separates the mechanisms;
at current precision it is invisible.

### 10.4 Bottom line

Both surviving models can share the physical collection ratio ≈ 0.65. They
differ in what carries the rest of the front/back imbalance: a long
back-side transport length (split, λ_b ≈ 10 nm) versus a deposition warp at
the monotonicity limit (channelling, ρ → 0.95). These 15 points cannot
separate the readings (Δχ² ≈ 0.5 by Eq. 26's projection). An independent
constraint on the detector's true front/back collection ratio is the
discriminating measurement: the channelling fit moves by 10 χ² units across
that ratio's current 1σ, so a direct measurement would tighten — possibly
exclude — the channelling reading.

## 11. What the figure shows

Top panel: the 15 pooled points with y error bars (σ_i) and x error bars
(0.10·t_i), plus the two fitted curves — "Symmetric" (the null, dashed
grey) and "Asymmetric" (solid blue). Bottom panel: the normalized residual
of each model,

    r_i = [ y_i − M(t_i; p̂) ] / σ_eff,i^(null)                            (28)

with the common σ_eff so the two point sets are directly comparable. The
null's residuals swing systematically: −4σ at the thinnest points, +3.5σ
near the peak, negative again at 60–90 nm. Underestimated errors would
scatter points randomly; a systematic S-shaped pattern is the signature of
a wrong model *shape*. The asymmetric model scatters around zero within
about ±2.5σ. (The channelling curve is not plotted: it would lie within
0.005 of the blue curve everywhere — section 10.3.)

**Figure specification** (for exact reproduction):

- Canvas: 3.37 × 3.6 in (APL single-column width), two stacked panels
  sharing the t axis, height ratio 3:1, hspace 0.07.
- Style: `DL/dlister_r01.mplstyle`, then these rc overrides for column
  format: `figure.autolayout = False`, `font.size = 9`,
  `axes.labelsize = 10`, `xtick.labelsize = ytick.labelsize = 8`,
  `legend.fontsize = 7`.
- Data: black open circles (`fmt="o"`, `markerfacecolor="none"`,
  markersize 3.0, elinewidth 0.6), y errors = σ_i, x errors = 0.10·t_i,
  label "Data".
- Curves: evaluated on 400 points spanning [0, 1.05·max t]. Null:
  grey `#686868`, dashed, label "Symmetric". Asymmetric escape: blue
  `#2e0acc`, solid, label "Asymmetric". Linewidth 1.2. The grey/blue pair
  plus distinct linestyles passes colorblind-separation checks.
- Axes: top ylabel `$I_n$`, x range [0, 1.05·max t], y from 0; legend at
  `loc="lower left"`, `bbox_to_anchor=(0.10, 0.03)` (the empty band below
  the rising edge). No title.
- Residual panel: Eq. (28) per model, markers 2.2 pt in the model color, no
  lines; grey zero line (0.8 pt); y limits symmetric at ±1.15 × max |r_i|;
  xlabel `$t$ (nm)`, ylabel `$\Delta I_n/\sigma$`.
- Output: vector PDF (`bbox_inches="tight"`) plus a 600 dpi PNG preview of
  the same canvas.

## 12. Assumptions and cross-checks

1. **λ_esc = 2 nm is assumed, not fitted — and the null's failure is
   conditional on it.** Freeing the shared escape depth in the null
   (`fit_and_plot_models_r06`) takes λ to 8.3 ± 1.0 nm and χ² from 48.2 to
   15.5: statistically acceptable. The symmetric model is excluded by
   *physics*, not by curve shape alone — its rescue needs a shared escape
   depth ~4× the known SE escape depth on both surfaces AND it retains the
   geometry-forbidden C_fb ≈ 3.1. Do not claim a free λ_esc converges near
   2 nm; it does not. (Freeing λ_f in the asymmetric model instead is
   unidentified: C_fb = 9.6 ± 8.5, λ_f = 18 ± 11 — the reason λ_esc is an
   input.) Sweep 1.5/2.0/2.5 nm for the final systematic (planned; not in
   this report).
2. **D is the semi-infinite SRIM run.** The finite-thickness correction
   (report 06) changes deposition by a few percent. Report 08
   (`scripts/thickness_aware_fits.py`) refits all three models with the
   thickness-aware profile D(z; t): the null gets *worse* (χ² 48 → 60, the
   nested Δχ² grows from 33 to 46) while every asymmetric-model shape
   parameter moves by under 1σ. The paper claim is robust against this
   assumption, in the conservative direction.
3. **SE source ∝ deposited energy** (Eq. 2). Standard assumption; nothing
   here tests it.
4. **Error model.** y errors from Eq. (1); x errors through Eq. (18).
   χ²/dof ≈ 1.4 for the good models suggests the combined errors are
   approximately right.
5. **Parameter interpretation.** Quote C_fb in the physical convention
   (Eq. 25). Do not read the channelling δ_∞ as a pure channelling depth —
   it is an effective relocation scale that carries the front/back
   imbalance (section 10).

## 13. How to reproduce

The production entry point is

```
python scripts/paper_model_comparison.py
```

It contains only the code behind the published figure: the null and
asymmetric-escape fits (physical kernels), their printed parameters, and
the two-curve figure written to `outputs/paper_model_comparison.pdf`
(vector) plus a 600 dpi `.png` preview. The channelling fit of section 9 is
reproduced by `fit_and_plot_models_r05` in
`scripts/david_model_exploration.py` (physical convention) and by the
baseline variant of `scripts/thickness_aware_fits.py`. Code map:

| content | location |
|---|---|
| Eqs. (3)–(15): kernels and the two figure models, physical convention | `scripts/paper_model_comparison.py` (`escape_terms`, `null_model`, `split_model` inside `fit_models`) |
| Eq. (10): channelling fit (physical convention) | `fit_and_plot_models_r05` in `scripts/david_model_exploration.py`; also `scripts/thickness_aware_fits.py` |
| Eqs. (16)–(20): weighting, iteration, covariance | `wfit`, `report` inside `fit_models` |
| Eqs. (22)–(23), (28) and the figure | comparison block in `fit_models`; `create_paper_figure` |
| Section 10.2–10.3: free/fixed/effective C_fb study | `fit_and_plot_models_r04` in `scripts/david_model_exploration.py` |
| Eqs. (24)–(25): physical-kernel cross-check | `fit_and_plot_models_r05` in `scripts/david_model_exploration.py` |
| Section 12 item 2: thickness-aware deposition refit | `fit_and_plot_models_r03` in `scripts/david_model_exploration.py` |

The exploration script (`scripts/david_model_exploration.py`) preserves the
full revision history (r01–r05) and the diagnostic figures; the production
script is a faithful extraction of its r05 path (physical kernels — the
r02 unit-area path is the same fit in different coordinates) plus the paper
figure (full rendering specification in section 11).

Everything is deterministic: single-start bounded least squares from the
fixed initial guesses of section 4, no random numbers anywhere, so a rerun
reproduces every digit. Environment: Python 3.x with NumPy, SciPy
(`least_squares`, `lfilter`, `interp1d`), Matplotlib, pandas; no other
dependencies.

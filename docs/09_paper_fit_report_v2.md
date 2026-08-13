# Paper-figure fit report v2: symmetric, free-λ symmetric, and asymmetric models

**Date:** 2026-07-29 · **Code:** `scripts/paper_model_comparison.py` ·
**Figure:** `outputs/paper_model_comparison.pdf` · **Companions:**
`docs/06_thickness_sweep_report.md` (finite-thickness deposition),
`docs/08_thickness_aware_fits_report.md` (deposition robustness check),
`docs/07_paper_fit_report.md` (archived: channelling analysis, kernel-convention
derivation, C_fb studies)

**Audience.** A physicist who does not work with fitting or statistics
daily. The language stays plain, but every equation needed to reimplement
the fits appears explicitly, and each statistical step names the standard
concept it uses.

**The claim the figure supports.** A symmetric SE-escape model with the
physical escape depth fails the data outright. Freeing the escape depth
rescues the fit *statistically* — but only at parameter values the physics
forbids (a shared 8 nm escape depth and a 3× back-collection advantage).
A single back-side asymmetry fits equally well with every parameter
physical. The shape of the data alone does not choose between the two
rescues; physical admissibility does.

## 1. Notation (one meaning per symbol)

| symbol | meaning | units |
|---|---|---|
| t | flake thickness, from EELS | nm (measured axis) |
| z | depth below the entrance (beam-side) surface | nm (measured axis) |
| D(ζ) | energy deposition profile of the 30 keV He beam, from SRIM | eV/Å |
| s | stretch: depth-axis calibration between EELS and SRIM | — |
| λ | an SE escape depth | nm (measured axis) |
| I_n | normalized SE transmission intensity | — |
| y_i, σ_i | measured intensity and its 1σ error at thickness t_i | — |

The "front" surface is the entrance surface; the "back" surface is the
exit surface, throughout.

## 2. The data

`data/reference_summaries/transmission_bayesian_summary.csv`: N = 15 points
from 3 suspended MoS₂ flakes, thicknesses 14–212 nm, sorted by thickness
and pooled (no point excluded or reweighted by flake). Each point has a
normalized intensity y_i and error σ_i, the symmetric average of the 68%
Bayesian posterior interval (look up: *credible interval*):

    σ_i = ½ (σ_lower,68,i + σ_upper,68,i)                                (1)

Thickness carries its own 10% uncertainty
(`THICKNESS_FRACTIONAL_ERROR = 0.10`); section 6 shows how it enters.

## 3. The shared forward model

The SE source at measured depth z is proportional to the SRIM deposition
evaluated on the SRIM axis:

    S(z) = D(z / s)                                                      (2)

D comes from `data/SRIM 30 keV - Monolayer Long/IONIZ.txt` (100 points,
3–300 nm), cubic-spline interpolated and clamped to its end values outside
the table. SRIM conditions for that run (from its `TDATA.txt`):
SRIM-2013.00, He (4.003 u) at 30 keV, normal incidence; one homogeneous
layer 300 nm deep, Mo 33.3 / S 66.6 at%, 5.710×10²² atoms/cm³ =
5.06 g/cm³; damage energies (Mo/S) displacement 25/25 eV, binding 3/3 eV,
surface 6.83/2.88 eV; 999,999 ions, 4.43% backscattered, 0.55% transmitted;
mean projected range 166.7 nm, straggle 63.1 nm (SRIM axis). Only the
"IONIZ. by IONS" column is used (the recoil column is ~1.5% and omitted).
Header caveat: the stale first line "H (10) into Layer 1" is a setup label;
the ion record (He, 30 keV) governs.

An SE born at depth z escapes a surface with probability that decays
exponentially with the distance to that surface (look up: *secondary
electron escape depth*); an SE born *at* a surface has escape odds set by
surface physics alone, independent of λ. The two escape yields for a film
of thickness t are

    Y_f(t; λ) = ∫₀ᵗ S(z) e^(−z/λ) dz          (front / entrance)          (3)

    Y_b(t; λ) = ∫₀ᵗ S(z) e^(−(t−z)/λ) dz      (back / exit)               (4)

Each depth slice contributes independently, so nothing normalizes the
kernel and the emitted count grows with λ. These are the *physical*
kernels: the fitted back-channel weight C_fb below reads directly as the
back/front collection-and-generation ratio. (A unit-area convention,
kernels divided by λ, was used during exploration because it decorrelates
λ from the amplitudes; the two are pure reparameterizations — archived
report 07, section 10.1, derives the mapping.)

The detector collects the two surfaces with different efficiencies, so the
general model is

    I_n(t) = C_o [ Y_f(t; λ_front) + C_fb · Y_b(t; λ_back) ]              (5)

## 4. The three models

**Symmetric (the null).** One escape depth, fixed at the physical value
λ_esc = 2 nm, shared by both surfaces. Free parameters p = (C_o, C_fb, s),
k = 3:

    I_sym(t) = C_o [ Y_f(t; λ_esc) + C_fb · Y_b(t; λ_esc) ] , λ_esc = 2 nm  (6)

C_fb stays free: detector geometry alone makes the two collection
efficiencies differ. The null hypothesis is symmetric *transport* with a
physical escape depth, not equal collection.

**Symmetric with free λ.** The same model with the shared escape depth
fitted. Free parameters p = (C_o, C_fb, s, λ), k = 4:

    I_symλ(t) = C_o [ Y_f(t; λ) + C_fb · Y_b(t; λ) ]                       (7)

This is the null's only statistical rescue, and it is presented as a model
rather than hidden as a caveat: the reader sees exactly what freeing λ
buys, and what it costs (section 10).

**Asymmetric.** The back surface gets its own escape depth; the front
keeps the physical λ_esc = 2 nm. Free parameters p = (C_o, C_fb, λ_b, s),
k = 4:

    I_asym(t) = C_o [ Y_f(t; λ_esc) + C_fb · Y_b(t; λ_b) ]                 (8)

Physical picture: the beam drives delta electrons forward, so effective SE
transport toward the exit surface can differ from transport toward the
entrance; alternatively λ_b acts as an effective correction for deposition
physics SRIM omits (e.g. channelling — archived report 07, sections 4 and
10.3, shows an explicit channelling model is observationally degenerate
with this one).

Both k = 4 models nest the null exactly (λ → 2 nm, or λ_b → 2 nm), each
with one extra degree of freedom. They do not nest each other.

**Why λ_esc is fixed in Eqs. (6) and (8).** Freeing the front escape depth
*and* keeping the free back channel is unidentified — the fit returns
C_fb = 9.6 ± 8.5, λ_f = 18 ± 11 (`fit_and_plot_models_r06`): the data
cannot constrain both escape depths and the collection ratio. Fixing
λ_esc to the independently known ~2 nm SE escape depth is the physics
input that makes the asymmetric model identifiable (look up: *parameter
identifiability*).

**Fit setup** (bounds keep the optimizer out of unphysical or
overflow-prone regions; no best fit ended on a bound):

| model | parameters p | initial guess | lower bounds | upper bounds |
|---|---|---|---|---|
| symmetric | C_o, C_fb, s | 0.01, 3.0, 0.68 | 0, 0, 0.2 | ∞, ∞, 3 |
| symmetric, λ free | C_o, C_fb, s, λ | null fit + λ = 2 | 0, 0, 0.2, 0.2 | ∞, ∞, 3, 50 |
| asymmetric | C_o, C_fb, λ_b, s | null fit + λ_b = 2 | 0, 0, 0.2, 0.2 | ∞, ∞, 50, 3 |

Both extensions warm-start from the converged null fit, which makes the
nested comparisons well-posed.

## 5. Numerical evaluation (exact per-cell kernels)

The integrals (3)–(4) are computed on a uniform grid of n = 8192 points
t_j = j·Δ spanning [0, max t] of the evaluation request (212 nm during
fitting; 1.05 × 212 nm for the plotted curves). S is treated as linear
inside each cell; for a linear function times an exponential each cell
integral is exact, so the only discretization error is the
piecewise-linear representation of S. With

    a = Δ/λ ,   e = e^(−a) ,   φ = (1 − e)/a ,   ψ = (φ − e)/a           (9)

the front cell contribution and cumulative yield are

    F_j = e^(−t_j/λ) · Δ [ (φ − ψ) S_j + ψ S_{j+1} ]                     (10)

    Y_f(t_k; λ) = Σ_{j<k} F_j                                            (11)

and the back yield obeys the exact one-step recursion

    B_0 = 0 ;   B_k = e · B_{k−1} + Δ [ (φ − ψ) S_k + ψ S_{k−1} ]        (12)

    Y_b(t_k; λ) = B_k                                                    (13)

evaluated with `scipy.signal.lfilter([c1, c0], [1, −e], S, zi=[−c1·S_0])`,
c1 = Δ(φ−ψ), c0 = Δψ. The formulas are exact for any Δ/λ. Model values at
arbitrary t are linear interpolations on the grid.

## 6. The fit

For a trial parameter set p, the weighted residual and objective are

    r_i(p) = [ M(t_i; p) − y_i ] / σ_eff,i ,   χ²(p) = Σ_i r_i(p)²       (14)

(look up: *weighted least squares*; identical to maximum likelihood for
Gaussian errors). No prior terms are used.

**Thickness errors (effective variance).** The 10% thickness error is an
x-axis error; the standard *effective variance* approximation converts it
to a y error via the local slope of the fitted curve (look up:
*errors-in-variables*):

    σ_eff,i² = σ_i² + [ M′(t_i) · 0.10 · t_i ]²                          (15)

**Iteration.** The weights need the fitted slope and vice versa, so `wfit`
iterates three times: minimize; evaluate the curve on 500 points spanning
[0, 300 nm], take `np.gradient`, interpolate the slope to each t_i; update
σ_eff and restart from the current best p. Three passes converge at the
printed precision. The null fit's converged σ_eff is frozen for the
comparison table.

**Optimizer.** `scipy.optimize.least_squares`, trust-region reflective,
bounds per section 4, `x_scale='jac'`, `ftol = xtol = 1e-12`. Parameters
are checked against bounds afterward (a pinned parameter would invalidate
Eq. 16); none pinned.

## 7. Parameter uncertainties

With J the residual Jacobian at the best fit,

    C = (Jᵀ J)⁻¹ ,  σ(p_j) = √C_jj ,  corr = C_jk/√(C_jj C_kk)           (16)

(look up: *Gauss-Newton approximation*). Residuals already contain σ_eff,
so no further scaling. Caution: linearized errors can understate how fast
χ² rises far from the minimum in near-degenerate directions (archived
report 07, section 10.2, shows a concrete case).

## 8. How the models are compared

**Absolute goodness of fit.** χ²/dof near 1 indicates a model consistent
with the data (look up: *reduced chi-squared*); the quantitative version
is the tail probability p = P(χ²_dof ≥ χ²_obs) (look up: *chi-squared
goodness-of-fit test*), which can reject a model with no alternative
invoked.

**Nested tests.** Each k = 4 model contains the null with one extra
parameter, so under the null the improvement Δχ² follows a χ² distribution
with 1 dof (look up: *likelihood-ratio test*, *Wilks' theorem*).

**Non-nested comparison.** The two k = 4 models are compared by
AIC = χ² + 2k (equal k, so ΔAIC = Δχ²; look up: *Akaike information
criterion*). For the table, all models are evaluated under the null fit's
converged σ_eff (Eq. 15) so the χ² values are mutually comparable:

    χ²_common(model) = Σ_i [ (M(t_i; p̂) − y_i) / σ_eff,i^(null) ]²        (17)

## 9. Results

Per-model fits (each under its own converged σ_eff):

| model | k | χ² | dof | χ²/dof | GoF p |
|---|---|---|---|---|---|
| Symmetric (null) | 3 | 48.2 | 12 | 4.02 | 2.9×10⁻⁶ |
| Symmetric (λ free) | 4 | 15.5 | 11 | 1.41 | 0.16 |
| Asymmetric | 4 | 14.9 | 11 | 1.35 | 0.19 |

Nested tests: null → symmetric-free Δχ² = 32.7; null → asymmetric
Δχ² = 33.3 (both 1 dof, both ≈ 5.7σ). Comparison under the common σ_eff
(Eq. 17): χ² = 48.2 / 19.0 / 17.5, AIC = 54.2 / 27.0 / 25.5 — the two
rescues are statistically indistinguishable (ΔAIC = 0.6 under each model's
own σ_eff, 1.5 under the common weighting; both far below any decision
threshold).

### Full fit parameters and correlations

All fits converged with no parameter on a bound. Physical kernel
convention throughout (C_fb = back/front collection-and-generation ratio).

**Symmetric (null)** — χ² = 48.2, dof = 12:

| parameter | value |
|---|---|
| C_o | 0.01104 ± 0.00044 |
| C_fb | 2.99 ± 0.16 |
| s | 0.701 ± 0.030 |

| corr | C_o | C_fb | s |
|---|---|---|---|
| C_o | 1 | −0.98 | −0.40 |
| C_fb | | 1 | +0.35 |
| s | | | 1 |

**Symmetric (λ free)** — χ² = 15.5, dof = 11:

| parameter | value |
|---|---|
| C_o | 0.0027 ± 0.0003 |
| C_fb | 3.12 ± 0.13 |
| s | 0.635 ± 0.032 |
| λ | 8.27 ± 0.98 nm |

| corr | C_o | C_fb | s | λ |
|---|---|---|---|---|
| C_o | 1 | −0.48 | +0.33 | −0.97 |
| C_fb | | 1 | +0.09 | +0.26 |
| s | | | 1 | −0.42 |
| λ | | | | 1 |

**Asymmetric** — χ² = 14.9, dof = 11:

| parameter | value |
|---|---|
| C_o | 0.01114 ± 0.00033 |
| C_fb | **0.649 ± 0.083** |
| λ_b | 9.63 ± 1.28 nm |
| s | 0.625 ± 0.034 |

Derived: λ_b − λ_esc = 7.63 ± 1.28 nm.

| corr | C_o | C_fb | λ_b | s |
|---|---|---|---|---|
| C_o | 1 | −0.37 | +0.07 | −0.31 |
| C_fb | | 1 | −0.95 | +0.54 |
| λ_b | | | 1 | −0.50 |
| s | | | | 1 |

The stretch lands at s ≈ 0.62–0.70 in all fits, consistent with the
campaign fits (report 05) and indicating a genuine EELS↔SRIM axis
systematic.

## 10. Reading: statistics finds two rescues; physics allows one

1. **The null fails absolutely**: GoF p = 2.9×10⁻⁶ — a fit this bad occurs
   ~3 times in a million if the model and error model are right — with the
   asymmetric model's p = 0.19 simultaneously certifying that the error
   model is sound.
2. **Statistics alone cannot pick the repair.** Freeing the shared λ or
   freeing the back-side λ_b each gains Δχ² ≈ 33 for one parameter, and
   the two k = 4 models are indistinguishable (ΔAIC ≤ 1.5).
3. **Physics picks.** The free-λ symmetric rescue requires
   λ = 8.3 ± 1.0 nm *on both surfaces* — ~4× the known SE escape depth —
   and retains C_fb = 3.12 ± 0.13, a 3× back-collection *advantage*; with
   equal escape depths, C_fb is directly the collection ratio, and the
   collection geometry forbids values above 1. Both of its fitted
   parameters are non-physical. The asymmetric model fits with every
   parameter physical: the front keeps the known 2 nm escape depth, the
   collection ratio is 0.649 ± 0.083 < 1 as geometry demands (an upper
   bound on the pure collection penalty, since it includes any exit-side
   generation enhancement), and the one non-trivial finding is
   λ_b = 9.6 ± 1.3 nm.
4. **Interpretation of λ_b is deliberately left open**: forward-directed
   delta-electron transport, or an effective correction for deposition
   physics SRIM omits (an explicit channelling model is observationally
   degenerate with this one — archived report 07, section 10.3).

## 11. What the figure shows

Top panel: the 15 pooled points with y error bars (σ_i) and x error bars
(0.10·t_i), plus the three fitted curves — "Symmetric" (dashed grey),
"Symmetric (λ free)" (red dash-dot), "Asymmetric" (dotted blue; dotted so
the near-coincident red curve underneath stays visible). The red and blue
curves nearly coincide, which is the visual statement of reading item 2. Bottom panel: normalized residuals per model,

    r_i = [ y_i − M(t_i; p̂) ] / σ_eff,i^(null)                           (18)

with the common σ_eff so the three point sets are comparable. The null's
residuals swing systematically (−4σ, +3.5σ, −2.5σ) — the signature of a
wrong model shape — while both k = 4 models scatter within ~±2.5σ.

**Figure specification** (for exact reproduction):

- Canvas 3.37 × 3.6 in (APL single column), two stacked panels sharing the
  t axis, height ratio 3:1, hspace 0.07.
- Style `config/dlister_r01.mplstyle` plus rc overrides:
  `figure.autolayout = False`, `font.size = 9`, `axes.labelsize = 10`,
  tick label size 8, `legend.fontsize = 7`.
- Data: black open circles (markersize 3.0, elinewidth 0.6), y errors σ_i,
  x errors 0.10·t_i, label "Data".
- Curves on 400 points spanning [0, 1.05·max t], linewidth 1.2:
  `#686868` dashed / `#cc0808` dash-dot / `#2e0acc` dotted (dotted so the
  red curve beneath it remains visible where the two coincide). The
  grey/red/blue trio with distinct linestyles passes colorblind-separation
  checks.
- Top ylabel `$I_n$`; x from 0; y from 0; legend `loc="lower left"`,
  `bbox_to_anchor=(0.10, 0.03)`; no title.
- Residual panel: markers 2.2 pt in model colors, grey zero line, y limits
  ±1.15 × max |r_i|; xlabel `$t$ (nm)`, ylabel `$\Delta I_n/\sigma$`.
- Output: vector PDF (`bbox_inches="tight"`) + 600 dpi PNG preview.

## 12. Assumptions and cross-checks

1. **λ_esc = 2 nm is an external physics input.** Its role is now explicit
   in the model set itself: the free-λ model shows exactly what happens
   without it. Sweep 1.5/2.0/2.5 nm for the final systematic (planned).
2. **D is the semi-infinite SRIM run.** Report 08 refits with the
   finite-thickness deposition D(z; t) of report 06: the null worsens
   (χ² 48 → 60; nested Δχ² 33 → 46) and the asymmetric parameters move
   < 1σ. The free-λ symmetric model has not yet been rerun under D(z; t);
   given the correction lowers thin-film deposition, its λ and C_fb are
   expected to move away from physicality, not toward it (unverified).
3. **SE source ∝ deposited energy** (Eq. 2). Standard; untested here.
4. **Error model**: y errors from Eq. (1), x errors via Eq. (15). GoF
   p = 0.16–0.19 for the k = 4 models supports it.
5. **Channelling is omitted from the figure**, not from the analysis:
   archived report 07 documents the explicit channelling model, its
   observational degeneracy with the asymmetric model (max family mismatch
   ≈ σ/5), and the C_fb transplant studies. Reproducible via
   `fit_and_plot_models_r04`/`_r05` and `scripts/thickness_aware_fits.py`.

## 13. How to reproduce

```
python scripts/paper_model_comparison.py
```

runs the three fits, prints every number in section 9 (with GoF p-values,
nested Δχ², and the physicality notes), and writes
`outputs/paper_model_comparison.pdf` plus the 600 dpi PNG preview.
Everything is deterministic: single-start bounded least squares from the
fixed initial guesses of section 4, no random numbers, so a rerun
reproduces every digit. Dependencies: NumPy, SciPy (`least_squares`,
`lfilter`, `interp1d`, `stats.chi2`), Matplotlib, pandas.

Code map: kernels and all three models — `escape_terms`, `sym_model`,
`null_model`, `split_model` inside `fit_models`; weighting/covariance —
`wfit`, `report`; figure — `create_paper_figure`. The free-λ discovery
path and the identifiability check are preserved as
`fit_and_plot_models_r06` in `scripts/david_model_exploration.py`; the
full exploration history is r01–r06 in that file.

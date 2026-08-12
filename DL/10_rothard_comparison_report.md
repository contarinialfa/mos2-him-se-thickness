# Literature comparison: Rothard et al. (1995) foil-emission model vs our fits

**Date:** 2026-07-30 · **Code:** `scripts/rothard_model_exploration.py` ·
**Paper:** H. Rothard et al., *Target-thickness-dependent electron emission
from carbon foils bombarded with swift highly charged heavy ions*, Phys.
Rev. A **51**, 3066 (1995) (PDF in `DL/`) · **Companions:**
`DL/09_paper_fit_report_v2.md` (our paper fits)

**Question.** Rothard et al. measured thickness-dependent electron yields
from *both* surfaces of thin foils and proposed semiempirical models for
them (their Eqs. 9–10). Are their findings consistent with ours, and how
does their model do on our data?

**Answer.** The mechanism agrees with ours, independently and from a very
different regime: the exit surface carries an extra, longer emission
length sourced by forward-transported δ electrons. Their model in its
literal form cannot fit our data — for a stated, structural reason (they
assume constant dE/dx; our flakes span the full He stopping curve) — but
once their two-channel exit emission is weighted by the deposition profile
it fits our data as well as our asymmetric model and is statistically
indistinguishable from it. Our single λ_b is the parsimonious effective
version of their two-channel structure.

## 1. Their experiment and terminology (mapped to ours)

13.6 MeV/u Ar^(16–18)+ through self-supporting carbon foils, 4–700 µg/cm²
(≈ 24–4200 nm at ρ = 1.65 g/cm³); low-energy (< 100 eV) electron yields
measured simultaneously at both surfaces with cylindrical Faraday cups.

| Rothard term | surface | our term |
|---|---|---|
| backward yield γ_B | beam entrance | front |
| forward yield γ_F | beam exit | back |

Their headline observations: γ_F > γ_B for all thicknesses; γ_B saturates
within ~15–20 µg/cm² (~100 nm); γ_F keeps rising until ~400–500 µg/cm²
(~2.5 µm) — the exit surface has a much longer thickness memory than the
entrance surface.

## 2. Their model (Eqs. 9–11)

Extended Sternglass picture: the projectile's energy loss splits into a
fraction β_S deposited in soft collisions (making slow electrons locally)
and β_δ = 1 − β_S carried by fast δ electrons (making slow electrons along
their forward transport, characteristic length λ_δ). Slow electrons then
diffuse to a surface with length λ_S. Assuming λ_δ ≫ λ_S and dE/dx
constant across the foil:

    γ_B(d) = Λ β_S (dE/dx) [1 − e^(−d/λ_S)]                              (their 9)

    γ_F(d) = Λ (dE/dx) [1 − β_S e^(−d/λ_S) − β_δ e^(−d/λ_δ)]             (their 10)

with Λ = P λ_S/⟨W⟩ (their 11). Sternglass equipartition sets β_S ≈ 0.5.
Note the structure: the entrance yield has *one* scale (λ_S); the exit
yield is a *sum of two saturating channels* (λ_S and λ_δ). For their
system λ_S ≈ 3 µg/cm² ≈ 18 nm and λ_δ is micrometre-scale.

## 3. Consistency of findings

1. **Same mechanism, independently proposed.** Their exit-side long scale
   is exactly the physics our asymmetric model encodes as λ_b > λ_f:
   forward-peaked δ electrons give the exit surface a deeper effective
   source region than the entrance surface. Our data demanded this
   (report 09); their higher-energy data demanded it first.
2. **Same asymmetry direction.** Their γ_F > γ_B everywhere ↔ our
   back-channel emission enhancement (which in our fits is partly masked
   by the collection penalty C_fb = 0.65; the *emission* asymmetry shows
   up as λ_b > λ_f).
3. **The scale ordering follows projectile velocity, as it should.** The
   maximum binary-encounter δ energy is E_BE = 2m_e v_p². For their
   13.6 MeV/u ions (v_p ≈ 23 a.u.): E_BE ≈ 29 keV → micrometre-scale λ_δ.
   For our 30 keV He: E_BE ≈ 16 eV → nanometre-scale δ transport. Their
   λ_δ/λ_S is orders of magnitude; ours (λ_b/λ_f ≈ 5, or
   λ_δ/λ_S ≈ 8.5 in the equipartition fit below) is a single order —
   the asymmetry survives at our energy but compressed, exactly as the
   velocity scaling predicts.
4. **Analogous bookkeeping parameters.** Their material parameter Λ and
   heavy-ion correction factors C_F, C_B (their Eqs. 14–16) play the same
   role as our C_o and C_fb: per-surface proportionality constants that
   absorb collection and material factors.
5. **A historical note:** the campaign's "Model-B" E5 form (archived
   report 07 §4; a fixed-2 nm saturating front plus a longer-scale back
   term) is recognizably this same Sternglass–Rothard structure.

## 4. Adapting their model to our data

**Why the literal Eqs. (9)–(10) cannot work here.** They assume dE/dx is
constant across the foil — valid in their regime, where even the thickest
foil is a small fraction of the ion range. Our flakes span the *entire*
stopping curve of 30 keV He (range ≈ 167 nm on the SRIM axis), which is
what makes I_n(t) rise, peak, and fall. Both of their yields are monotone
saturating functions of thickness; no parameter choice can produce a
decline.

**The D(z)-weighted generalization.** Keep their two-channel exit
structure but weight by the actual deposition profile (all conventions as
report 09; channel mixing uses unit-area kernels so each channel
saturates to its β fraction, mirroring their form — C_fb below is
therefore in the unit-area convention, not the physical collection
ratio):

    I(t) = C_o [ Ŷ_f(t; λ_S) + C_fb ( β Ŷ_b(t; λ_S) + (1−β) Ŷ_b(t; λ_δ) ) ]   (1)

with λ_S = λ_esc = 2 nm fixed, and either β = 0.5 fixed (their
equipartition; k = 4, same as our asymmetric model) or β free (k = 5;
Eq. 1 then nests both our asymmetric model at β = 0 and our null at
β = 1 or λ_δ = λ_S).

## 5. Results

Same data, same iterative effective-variance weighting as report 09; our
null and asymmetric models refitted in-script as anchors so all χ² come
from identical machinery.

| model | k | χ² | dof | χ²/ν (reduced) | GoF p | AIC |
|---|---|---|---|---|---|---|
| our null (anchor) | 3 | 48.2 | 12 | 4.02 | 2.9×10⁻⁶ | 54.2 |
| our asymmetric (anchor) | 4 | 15.0 | 11 | 1.36 | 0.18 | 23.0 |
| Rothard literal (const dE/dx) | 3 | 9916.5 | 12 | 826 | ~0 | 9922.5 |
| Rothard D(z), β = 0.5 | 4 | 15.9 | 11 | 1.45 | 0.14 | 23.9 |
| Rothard D(z), β free | 5 | 14.9 | 10 | 1.49 | 0.14 | 24.9 |

Reduced χ² in words: the D(z)-weighted Rothard model with their
equipartition fits at χ²/ν = 1.45, between our asymmetric model's 1.36
and our null's 4.02, and statistically equivalent to the former (both
GoF p ≥ 0.14). Freeing β *raises* the reduced χ² to 1.49 despite the
lower absolute χ², because the extra parameter costs a degree of freedom
for a gain of only Δχ² = 1.0.

![Rothard model comparison](rothard_comparison.png)

*Figure: the literal Rothard form (grey dashed) saturates and cannot
follow the stopping-curve decline; the D(z)-weighted equipartition form
(red dash-dot) and our asymmetric model (blue dotted) are nearly
coincident through the data. Residual panel: the two viable fits,
normalized by the asymmetric fit's converged σ_eff (the literal model's
residuals, at tens of σ, are omitted to keep the scale readable).*

Key parameters:

- Equipartition variant: λ_δ = 17.1 ± 4.4 nm (with the fixed λ_S = 2 nm
  channel carrying the other half of the exit emission), s = 0.633 ± 0.034.
- β free: β = 0.41 ± 0.30 — consistent with both equipartition (0.5) and
  our pure single-channel form (0) — and λ_δ = 14.2 ± 7.6 nm. The extra
  parameter buys Δχ² = 1.0: unsupported.
- Literal variant: λ_δ pins at its bound; χ²/ν ≈ 826. Regime mismatch,
  not a physics failure.

## 6. Reading

1. **Their model works on our data once it is given our deposition
   physics**, and then performs identically to our asymmetric model
   (ΔAIC = 0.9 at equal k). The data cannot distinguish a two-channel
   exit emission (their structure) from a single effective back-side
   escape depth (ours): our λ_b = 9.6 ± 1.3 nm is the effective
   single-scale summary of their 50/50 mix of 2 nm and 17.1 ± 4.4 nm.
2. **This is independent literature support for the paper's asymmetric
   model** — the same exit-side δ-transport structure was proposed and
   measured at 13.6 MeV/u, and scales to our energy in the expected
   direction. The paper can cite Rothard et al. as the higher-energy
   precedent for back/front emission asymmetry with a longer exit-side
   length scale.
3. **No paper conclusion changes.** Adopting their parameterization would
   relabel λ_b ≈ 9.6 nm as an equipartition mixture with λ_δ ≈ 17 nm;
   fit quality, the null rejection, and the physicality argument are
   untouched. We keep the single-λ_b form for parsimony.
4. **Caveats.** Their λ_S ≈ 18 nm (carbon, their electron-energy range)
   is not directly comparable to our 2 nm MoS₂ input; the comparison here
   is structural, not parameter-by-parameter. Their C_F/C_B heavy-ion
   reduction factors address charge-state physics irrelevant at our
   energy. And our test of "their model" is necessarily of its
   *transport structure* — their production physics (β_S from stopping
   partition) enters only through the equipartition value 0.5, which our
   data cannot check (β is unconstrained).

## 7. How to reproduce

```
python scripts/rothard_model_exploration.py
```

fits the two anchors and the three Rothard variants, prints every number
in section 5, and writes the figure to `DL/rothard_comparison.png`.
Deterministic single-start fits; the paper PDF is in `DL/`, and the model
equations are transcribed in section 2 above.

# Independent SRIM route-(i) reanalysis

This directory contains a read-only reconstruction of the completed campaign
`mos2_30keV_route_i_srim_axis_resolution_v1`. No manuscript or SRIM archive
file was modified.

## Reproduction

From the repository root in PowerShell:

```powershell
python 'analysis\srim_route_i_v1\analyze_archive.py'
```

The script streams every `TRANSMIT.txt` record and regenerates all CSV and JSON
outputs in this directory. Runtime on the audit machine was approximately
one minute.

## Verified observables

For incident ion count `N`, exit angle `theta`, and incident energy `E0=30 keV`:

- `T = N_transmitted/N`.
- `F20 = sum[theta>20 deg]/N`.
- `Q20 = sum[theta>20 deg] sqrt(E_exit/E0)/N`.
- `B4 = sum_transmitted sqrt(E_exit/E0) min(sec(theta),4)/N`.
- The frozen Model-B exit driver is `B4[1-exp(-u/15.53 nm)]`.

Only the two independent 100k seeds are pooled. The 10k reconnaissance run is
used outside the production grid; its 29 overlaps are exact prefixes of the
seed-13579 production stream and are therefore never pooled with it.

The parser verified 118 completed, warning-free runs, 8,560,000 incident ions,
and 5,080,084 transmitted records. Direction-cosine normalization agreed to a
maximum absolute error of `1.71e-7`. The secant cap affected 86,589 records,
1.7045% of transmitted ions.

## Decisive numerical results

The unweighted `F20` production-grid maximum is at raw SRIM `u=115 nm`; points
from 105 to 120 nm are statistically indistinguishable at an approximate 95%
pointwise level. A local quadratic gives `u=113.225 nm`.

The stopping-weighted `Q20` grid maximum is at `u=95 nm`; 90 and 95 nm are
statistically indistinguishable. A local quadratic gives `u=92.856 nm`, with
an approximately 0.29 nm SRIM-Monte-Carlo statistical uncertainty. This narrow
number excludes thickness-scale and proxy/model systematics.

Using frozen `s=0.7017`, these local peaks map to 79.45 nm (unweighted) and
65.16 nm (weighted), not to the manuscript's raw 125 and 80 nm values. Using
the new fit's `s=0.697655`, the weighted peak is 64.79 nm. Its local scale-fit
component is about 3.38 nm, and the manuscript's additional repeatability
component is about 6.50 nm. Therefore the weighted route-(i) peak is not far
enough beyond the observed 40--55 nm maximum to support the manuscript's
phrase "much greater thickness" from peak location alone. The fitted-scale
and repeatability terms also omit the larger acceptance-threshold,
chamber-geometry, and SRIM-model systematics relevant to an external-SE3
calculation.

The new `B4` table changes the Letter fit only slightly when the published
frozen-effective-variance WLS procedure is reproduced:

| Quantity | Frozen table | New campaign |
|---|---:|---:|
| `C_f` | 0.325915 | 0.327536 |
| `C_b` | 1.092148 | 1.093150 |
| raw `lambda_b` (nm) | 15.5332 | 15.7296 |
| `s` | 0.701742 | 0.697655 |
| `s lambda_b` (nm) | 10.9003 | 10.9738 |
| model peak (nm) | 41.222 | 41.036 |
| chi-square/dof | 12.666/11 | 13.450/11 |

The median absolute relative difference between old and new `B4` over the
production grid is 0.56%. The apparent 52% maximum relative difference occurs
only in the nearly zero tail; where old `B4>0.05`, the maximum relative change
is 1.72%.

Removing the `sec(theta)=4` cap gives `s lambda_b=10.492 nm`, model peak
42.385 nm, and chi-square 15.232/11. Thus the Model-B peak and effective length
remain broadly stable, although the cap is a material model sensitivity and
should be reported.

## Shape tests for external SE3

Using exactly the new Model-B frozen effective variances for a common-error
comparison:

| Model | Parameters | chi-square/dof | Delta AIC vs Model B |
|---|---:|---:|---:|
| New Model B | 4 | 13.450/11 | 0 |
| Route-(i)-only `C0+A Q20(t/s)` | 3 | 120.743/12 | +105.29 |
| Route-(i)+route-(ii), free `s` | 4 | 12.493/11 | -0.96 |
| Route-(i)+route-(ii), `s=0.697655` fixed | 3 | 13.375/12 | -2.07 |

The route-(i)-only proxy is strongly rejected by the full curve even though
its peak-location argument alone is weak. However, a positive combination

`I(t)=C0 + Ai Q_{>20}(t/s) + Aii Q_{<=20}(t/s)`

fits as well as Model B. With free `s`, the common-error fit gives
`C0=0.3341`, `Ai=1.8488`, `Aii=0.7528`, `s=0.6617`, and a 46.36 nm peak. With
`s=0.697655` fixed, it gives `C0=0.3265`, `Ai=1.8127`, `Aii=0.7980`, and a
48.16 nm peak. All amplitudes are positive.

This mixture test does not prove that external SE3 dominates. Its amplitudes
absorb unknown chamber-surface yield, intercepted fraction, detector visibility,
and collection efficiency. It does prove that separately rejecting the two
routes does not mathematically reject their positive mixture. Quantitative tape
calibration or another external constraint is required before claiming that the
curve shape limits SE3 to a minor contribution.

## Remaining caveats

- The simulation assumes 0-degree normal beam incidence; the experimental
  beam--membrane angle still needs confirmation.
- `sqrt(E/E0)` is a velocity-proportional stopping proxy, not a material-specific
  stopping calculation for the surface struck outside the membrane.
- The 20-degree route split is geometric and analysis-chosen; its sensitivity
  has not yet been profiled.
- SRIM uses an amorphous target and does not model crystallographic channeling.
- Pointwise Monte Carlo errors do not include SRIM model error, channeling,
  threshold choice, chamber geometry, or shared normalization covariance.
- Fixing or constraining `s` with Model B fitted to the same intensity data is
  only a sensitivity check, not independent evidence.

`analysis_results.json` contains full-precision results, covariance matrices,
fit weights, definitions, and all sensitivity fits. The CSV files contain the
per-run and pooled observables and the old/new driver comparisons.

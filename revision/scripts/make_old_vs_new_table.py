# -*- coding: utf-8 -*-
"""Format the side-by-side comparison table between the thick-target (300 nm) fits and the
thickness-dependent D(z; x) fits.

Reads results.json produced by revision/scripts/refit_thickness_dependent_D.py and writes
old_vs_new.md, tabulating chi2/dof, goodness-of-fit p-values, common-weighting chi2 and AIC,
fitted parameters, model peak positions, nested delta-chi2, derived ratios, entrance-surface
contributions, and sensitivity tests (thickness scale, point thickness error, Flake 3,
leave-one-out, and entrance escape depth).

Usage (from the repository root):
    python revision/scripts/make_old_vs_new_table.py
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "revision/outputs/refit_thickness_dependent_D/results.json"
parser = argparse.ArgumentParser(description="Thick-target vs D(z; x) comparison table.")
parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="results.json of the refit (default: revision/outputs/refit_thickness_dependent_D/results.json)")
parser.add_argument("--out", type=Path, default=None, help="output folder (default: the folder of --results)")
args = parser.parse_args()
OUT = (args.out if args.out is not None else args.results.parent).resolve()
OUT.mkdir(parents=True, exist_ok=True)

R = json.load(io.open(args.results, encoding="utf-8"))
L, A = R["long"], R["aware"]
rows = []


def add(q, a, b, fmt="%.2f"):
    rows.append((q, fmt % a if not isinstance(a, str) else a, fmt % b if not isinstance(b, str) else b))


def pm(v, f="%.2f"):
    return (f + " ± " + f) % (v[0], v[1])


for k, name in (("null", "Symmetric, 2 nm"), ("symfree", "Symmetric, free depth"), ("split", "Asymmetric")):
    l, a = L["baseline"][k], A["baseline"][k]
    add(name + ": chi2/dof (own weights)", "%.1f/%d = %.2f" % (l["chi2"], l["dof"], l["chi2"] / l["dof"]), "%.1f/%d = %.2f" % (a["chi2"], a["dof"], a["chi2"] / a["dof"]))
    add(name + ": GoF p", l["p_gof"], a["p_gof"], "%.2g")
    add(name + ": chi2 common weighting", l["chi2_common"], a["chi2_common"], "%.1f")
    add(name + ": AIC", l["aic_common"], a["aic_common"], "%.1f")
    add(name + ": C_b", pm(l["C_b"]), pm(a["C_b"]))
    add(name + ": s", pm(l["s"], "%.3f"), pm(a["s"], "%.3f"))
    if k == "symfree":
        add(name + ": shared depth (nm)", pm(l["lambda"]), pm(a["lambda"]))
    if k == "split":
        add(name + ": exit-side depth (nm)", pm(l["lambda_b"]), pm(a["lambda_b"]))
    add(name + ": model maximum (nm)", l["peak_nm"], a["peak_nm"], "%.1f")
    add(name + ": model maximum I_n", l["peak_I"], a["peak_I"], "%.3f")
    if k != "null":
        add(name + ": nested dchi2 vs symmetric (common)", l["dchi2_common_vs_null"], a["dchi2_common_vs_null"], "%.1f")
        add(name + ": sigma equivalent", l["sigma_equivalent"], a["sigma_equivalent"], "%.1f")
        add(name + ": dchi2 vs symmetric (own weights)", l["dchi2_own_vs_null"], a["dchi2_own_vs_null"], "%.1f")
l, a = L["derived"], A["derived"]
add("C_b*lambda_b/lambda_front (asym)", l["Cb_lamb_over_lamf"], a["Cb_lamb_over_lamf"])
add("exit/blocked ratio at model maximum", l["exit_to_blocked_ratio_at_model_max"], a["exit_to_blocked_ratio_at_model_max"])
add("total/blocked ratio at model maximum", l["total_to_blocked_ratio_at_model_max"], a["total_to_blocked_ratio_at_model_max"])
add("exit-side depth ± with 15% scale (nm)", l["lambda_b_with_scale_15pct"], a["lambda_b_with_scale_15pct"])
add("exit-side depth ± with 20% scale (nm)", l["lambda_b_with_scale_20pct"], a["lambda_b_with_scale_20pct"])
for x in ("14.1", "54.1", "100.0", "212.0"):
    add("modelled entrance term at %s nm" % x, l["entrance_term_at_t_nm"][x], a["entrance_term_at_t_nm"][x], "%.4f")
add("Flake 3 share of symmetric chi2 (common)", "%.1f of %.1f" % (l["null_chi2_common_by_dataset"]["Dataset 3"], L["baseline"]["null"]["chi2_common"]), "%.1f of %.1f" % (a["null_chi2_common_by_dataset"]["Dataset 3"], A["baseline"]["null"]["chi2_common"]))
for kap in ("0.80", "0.85", "1.15", "1.20"):
    l, a = L["scale"][kap], A["scale"][kap]
    add("scale x%s: asym chi2 / depth / max (nm)" % kap, "%.1f / %.1f / %.1f" % (l["split_chi2"], l["lambda_b"][0], l["peak_nm"]), "%.1f / %.1f / %.1f" % (a["split_chi2"], a["lambda_b"][0], a["peak_nm"]))
for tfe in ("0.05", "0.15", "0.20"):
    l, a = L["point_thickness_error"][tfe], A["point_thickness_error"][tfe]
    add("point error %s: sym chi2 | asym chi2 | depth" % tfe, "%.1f | %.1f | %s" % (l["null_chi2"], l["split_chi2"], pm(l["lambda_b"], "%.1f")), "%.1f | %.1f | %s" % (a["null_chi2"], a["split_chi2"], pm(a["lambda_b"], "%.1f")))
    add("point error %s: sym s" % tfe, l["null_s"], a["null_s"], "%.3f")
for k3 in ("0.80", "1.20"):
    l, a = L["flake3_rescaled"][k3], A["flake3_rescaled"][k3]
    add("Flake 3 x%s: sym chi2 | depth (nm)" % k3, "%.1f | %.1f" % (l["null_chi2"], l["lambda_b"][0]), "%.1f | %.1f" % (a["null_chi2"], a["lambda_b"][0]))
l, a = L["without_flake3"], A["without_flake3"]
add("without Flake 3: sym chi2/dof, p", "%.2f/%d, %.2f" % (l["null_chi2"], l["dof"], l["p_gof"]), "%.2f/%d, %.2f" % (a["null_chi2"], a["dof"], a["p_gof"]))
for nm, r in (("long", L), ("aware", A)):
    lo = r["leave_one_out_null"]
    r["_loo"] = "%.1f-%.1f, max p %.1g" % (min(x["chi2"] for x in lo), max(x["chi2"] for x in lo), max(x["p_gof"] for x in lo))
add("leave one point out: sym chi2 range", L["_loo"], A["_loo"])
for le in ("1.5", "2.5"):
    l, a = L["entrance_scale"][le], A["entrance_scale"][le]
    add("entrance scale %s nm: C_b | depth | s" % le, "%.2f | %.2f | %.3f" % (l["C_b"][0], l["lambda_b"][0], l["s"][0]), "%.2f | %.2f | %.3f" % (a["C_b"][0], a["lambda_b"][0], a["s"][0]))
out = ["| Quantity | Long target (submitted) | D(z; x) |", "|---|---|---|"] + ["| %s | %s | %s |" % r for r in rows]
io.open(OUT / "old_vs_new.md", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out))

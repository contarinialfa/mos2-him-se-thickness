# -*- coding: utf-8 -*-
"""Quadrature, weighting, and deposition-profile validation for the D(z; x) model fits.

Cross-checks the three SE-escape model fits of revision/scripts/refit_thickness_dependent_D.py
using direct trapezoidal integration on fine per-thickness grids and central-difference
Jacobians, and validates the finite-thickness correction against the nine raw SRIM thickness-series
runs (SI Sec. SVIII):
  - verifies grid endpoint placement within the membrane for all 15 thicknesses;
  - tests trapezoidal grid convergence (n = 2001 / 8001 / 64001) and effective-variance
    iteration settings (3 vs 12 passes; 500-point grid slope vs direct central differences);
  - recomputes derived quantities (common-weighting chi2, model maxima, peak-to-blocked ratios,
    scale-inflated exit-depth uncertainties, and entrance/exit terms);
  - compares D(z; x) against the nine finite-thickness SRIM runs bin-by-bin and via
    escape-weighted sums;
  - tabulates the positions of the 15 data points on the SRIM depth scale (x / s) relative to
    the 10-200 nm sweep runs and the 300 nm target.

Outputs (in --out): verify_results.json, verify_stdout.txt.

Usage (from the repository root):
    python revision/scripts/verify_thickness_dependent_profile.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.dont_write_bytecode = True
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import CubicSpline, RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
IONIZ = ROOT / 'data/SRIM 30 keV - Monolayer Long/IONIZ.txt'
TRANSMISSION = ROOT / 'data/reference_summaries/transmission_bayesian_summary.csv'
SWEEP_DIR = ROOT / 'data/SRIM-Thickness-Series'
parser = argparse.ArgumentParser(description='Validation of the D(z; x) refit and deposition profile.')
parser.add_argument('--results', type=Path, default=ROOT / 'revision/outputs/refit_thickness_dependent_D/results.json',
                    help='results.json of the refit (default: revision/outputs/refit_thickness_dependent_D/results.json)')
parser.add_argument('--out', type=Path, default=ROOT / 'revision/outputs/verify_thickness_dependent_profile',
                    help='output folder (default: revision/outputs/verify_thickness_dependent_profile)')
args = parser.parse_args()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts'))
import thickness_sweep_ionization as tsi  # noqa: E402

REF = json.loads(args.results.read_text(encoding='utf-8'))
LAM_ESC, TFE, BLOCKED = 2.0, 0.10, 0.2361
log_lines = []


def say(text=''):
    print(text, flush=True)
    log_lines.append(str(text))


# ---------------------------------------------------------------- inputs
depth, ion = tsi.load_ionization(IONIZ)
frame = pd.read_csv(TRANSMISSION)
frame['err'] = 0.5 * (frame.normalized_err_lower_68 + frame.normalized_err_upper_68)
frame = frame.sort_values('thickness_nm').reset_index(drop=True)
T, Y, E = (frame.thickness_nm.to_numpy(float), frame.normalized_median.to_numpy(float), frame.err.to_numpy(float))
sweep = [(t, *tsi.load_ionization(p)) for t, p in tsi.discover_sweep(SWEEP_DIR)]
with contextlib.redirect_stdout(io.StringIO()):
    d_aware_tsi, info = tsi.build_ionization_model(depth, ion, sweep)
theta, T_LONG = info['theta'], float(info['t_long'])
assert info['kernel_name'] == 'exponential-zlam'
say('long SRIM grid: %d bins, %.4f .. %.4f nm; D first %.4f, last five %s eV/A'
    % (len(depth), depth[0], depth[-1], ion[0], np.array2string(ion[-5:], precision=4)))

# ---------------------------------------------------------------- deposition evaluation from theta
_cs = CubicSpline(depth, ion)


def dlong(z):
    return _cs(np.clip(z, depth[0], depth[-1]))


def ratio_eval(z, t):
    amp = theta[0] + theta[1] * z + theta[2] * z**2
    lam = theta[3] * np.exp(theta[4] * z)
    return (1 - amp * np.exp(-(t - z) / lam)) / (1 - amp * np.exp(-(T_LONG - z) / lam))


def dep_long(z, t_raw):
    return dlong(z)


def dep_aware(z, t_raw):
    ts = np.clip(t_raw, 10.0, T_LONG)
    return ratio_eval(z * (ts / t_raw), ts) * dlong(z)


# Model-free alternative: linear interpolation of the raw ratio rows (thickness x fractional depth)
_rat = info['ratios'].copy()
for row in _rat:
    for j in range(1, len(row)):
        if row[j] < 0.8:
            row[j] = row[j - 1]
_fg = np.concatenate(([0.0], info['f_grid'], [1.0]))
_rat = np.column_stack((_rat[:, 0], _rat, _rat[:, -1]))
_rgi = RegularGridInterpolator((info['t_nodes'], _fg), _rat, method='linear')


def dep_rawinterp(z, t_raw):
    ts = np.clip(t_raw, info['t_nodes'][0], info['t_nodes'][-1]) * np.ones_like(z)
    f = np.clip(z / t_raw, 0.0, 1.0)
    r = _rgi(np.column_stack((ts.ravel(), f.ravel()))).reshape(z.shape)
    return r * dlong(z)


DEP = {'long': dep_long, 'aware': dep_aware, 'rawinterp': dep_rawinterp}

# Direct kernel evaluation against build_ionization_model
worst = 0.0
for t_raw in (8.0, 14.1 / 0.63, 37.0, 86.0, 159.0, 240.0, 299.0, 337.0):
    z = np.linspace(0, t_raw, 4001)
    a, b = d_aware_tsi(z, t_raw), dep_aware(z, t_raw)
    worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))))
    assert a[-1] != 0.0
say('direct D(z; x) evaluation vs build_ionization_model: largest relative difference %.2e' % worst)
# Verify the refit grid endpoint lies inside the film (ionization() returns 0 for z > thickness)
for x in T:
    for s in (0.6278114655835976, 0.6390433037833285, 0.7336995079321014):
        tt = np.linspace(0, x, 2048)
        assert d_aware_tsi(tt / s, x / s)[-1] > 0.0
say('refit grid end point is inside the film for all 15 thicknesses (no zeroed last node)')


# ---------------------------------------------------------------- integration: trapezoid
def integrals(dep, x, s, lam_f, lam_b, n=8001, chunk=400):
    x = np.atleast_1d(np.asarray(x, float))
    front, back = np.zeros_like(x), np.zeros_like(x)
    idx = np.flatnonzero(x > 0)
    u = np.linspace(0.0, 1.0, n)[None, :]
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        xp = x[sel][:, None]
        zz = xp * u
        d = dep(zz / s, xp / s)
        h = xp[:, 0] / (n - 1)
        ff = d * np.exp(-zz / lam_f)
        fb = d * np.exp(-(xp - zz) / lam_b)
        front[sel] = h * (ff.sum(axis=1) - 0.5 * (ff[:, 0] + ff[:, -1]))
        back[sel] = h * (fb.sum(axis=1) - 0.5 * (fb[:, 0] + fb[:, -1]))
    return front, back


def make(kind, dep, lam_esc=LAM_ESC, n=8001):
    if kind == 'null':        # p = C_o, C_b, s
        def f(x, p):
            a, b = integrals(dep, x, p[2], lam_esc, lam_esc, n)
            return p[0] * (a + p[1] * b)
    elif kind == 'symfree':   # p = C_o, C_b, s, lambda (one depth for both faces)
        def f(x, p):
            a, b = integrals(dep, x, p[2], p[3], p[3], n)
            return p[0] * (a + p[1] * b)
    else:                     # split: p = C_o, C_b, lambda_b, s
        def f(x, p):
            a, b = integrals(dep, x, p[3], lam_esc, p[2], n)
            return p[0] * (a + p[1] * b)
    return f


AXIS = np.linspace(0, float(depth.max()), 500)
BOUNDS = {'null': ([0, 0, 0.2], [np.inf, np.inf, 3]), 'symfree': ([0, 0, 0.2, 0.2], [np.inf, np.inf, 3, 50]),
          'split': ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3])}
NAMES = {'null': ['C_o', 'C_b', 's'], 'symfree': ['C_o', 'C_b', 's', 'lambda'], 'split': ['C_o', 'C_b', 'lambda_b', 's']}


def slope_at(f, p, mode):
    if mode == 'axis':
        return np.interp(T, AXIS, np.gradient(f(AXIS, p), AXIS))
    h = 0.05
    return (f(T + h, p) - f(T - h, p)) / (2 * h)


def fit(kind, dep, p0, n_iter=3, mode='axis', tfe=TFE):
    f = make(kind, dep)
    sig, hist = E.copy(), []
    for _ in range(n_iter):
        sig_used = sig
        sol = optimize.least_squares(lambda p, w=sig_used: (f(T, p) - Y) / w, p0, bounds=BOUNDS[kind], x_scale='jac',
                                     ftol=1e-12, xtol=1e-12, gtol=1e-12)
        p0 = sol.x
        sig = np.sqrt(E**2 + (slope_at(f, p0, mode) * tfe * T)**2)
        hist.append((float(np.sum(((f(T, p0) - Y) / sig_used)**2)), float(np.sum(((f(T, p0) - Y) / sig)**2))))
    jac = np.zeros((len(T), len(p0)))
    for k in range(len(p0)):          # central-difference Jacobian at converged weights
        dp = np.zeros(len(p0))
        dp[k] = 1e-5 * abs(p0[k])
        jac[:, k] = (f(T, p0 + dp) - f(T, p0 - dp)) / (2 * dp[k]) / sig_used
    err = np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)))
    return {'p': p0, 'e': err, 'chi2': hist[-1][1], 'chi2_fit_weights': hist[-1][0], 'sig': sig, 'hist': hist, 'f': f}


def ref_p(dep_name, kind):
    return np.array([REF[dep_name]['baseline'][kind][n][0] for n in NAMES[kind]])


def peak_of(f, p):
    grid = np.arange(5.0, 100.0, 0.02)
    curve = f(grid, p)
    return float(grid[np.argmax(curve)]), float(np.max(curve))


out = {}
# ---------------------------------------------------------------- trapezoidal fits, all three models
say('\n=== trapezoidal-quadrature fits (3 weighting passes, slope on 500-point axis) ===')
trap_fits = {}
for dep_name in ('long', 'aware', 'rawinterp'):
    r_null = fit('null', DEP[dep_name], [0.01, 3.0, 0.68])
    r_sym = fit('symfree', DEP[dep_name], [*r_null['p'], LAM_ESC])
    r_split = fit('split', DEP[dep_name], [*r_null['p'][:2], LAM_ESC, r_null['p'][2]])
    trap_fits[dep_name] = {'null': r_null, 'symfree': r_sym, 'split': r_split}
    out[dep_name] = {}
    for kind, r in trap_fits[dep_name].items():
        common = float(np.sum(((r['f'](T, r['p']) - Y) / r_null['sig'])**2))
        r['chi2_common'] = common
        out[dep_name][kind] = {'p': dict(zip(NAMES[kind], map(float, r['p']))), 'e': dict(zip(NAMES[kind], map(float, r['e']))),
                               'chi2': r['chi2'], 'chi2_fit_weights': r['chi2_fit_weights'], 'chi2_common': common}
        line = '%-9s %-7s chi2 %.4f (fit weights %.4f) common %.4f | ' % (dep_name, kind, r['chi2'], r['chi2_fit_weights'], common)
        line += ', '.join('%s %.5g +/- %.3g' % (n, v, e) for n, v, e in zip(NAMES[kind], r['p'], r['e']))
        say(line)
        if dep_name in REF:
            ref = REF[dep_name]['baseline'][kind]
            line = '%-9s %-7s results.json: chi2 %.4f common %.4f | ' % ('', '', ref['chi2'], ref['chi2_common'])
            line += ', '.join('%s %.5g +/- %.3g' % (n, ref[n][0], ref[n][1]) for n in NAMES[kind])
            say(line)
    for kind in ('symfree', 'split'):
        d = trap_fits[dep_name]['null']['chi2_common'] - trap_fits[dep_name][kind]['chi2_common']
        out[dep_name][kind]['dchi2_common_vs_null'] = d
        say('%-9s %-7s nested dchi2 (common) %.3f, sqrt %.3f%s' % (dep_name, kind, d, np.sqrt(d), (
            '   results.json %.3f' % REF[dep_name]['baseline'][kind]['dchi2_common_vs_null']) if dep_name in REF else ''))

# Trapezoid grid convergence at the fitted aware symfree parameters
p = trap_fits['aware']['symfree']['p']
vals = [float(np.sum(((make('symfree', dep_aware, n=n)(T, p) - Y) / trap_fits['aware']['symfree']['sig'])**2)) for n in (2001, 8001, 64001)]
say('trapezoid grid convergence, aware symfree chi2 at n = 2001 / 8001 / 64001: %.5f / %.5f / %.5f' % tuple(vals))
out['trapezoid_convergence_symfree_aware'] = vals

# Weighting: more passes, and a direct central-difference slope at the data instead of the 0.6 nm axis
say('\n=== weighting sensitivity (aware) ===')
out['weighting'] = {}
for label, kw in (('12 passes, axis slope', {'n_iter': 12}), ('3 passes, direct slope', {'mode': 'direct'}),
                  ('12 passes, direct slope', {'n_iter': 12, 'mode': 'direct'})):
    rn = fit('null', dep_aware, [0.01, 3.0, 0.68], **kw)
    rs = fit('symfree', dep_aware, [*rn['p'], LAM_ESC], **kw)
    rp = fit('split', dep_aware, [*rn['p'][:2], LAM_ESC, rn['p'][2]], **kw)
    cs_ = float(np.sum(((rs['f'](T, rs['p']) - Y) / rn['sig'])**2))
    cp_ = float(np.sum(((rp['f'](T, rp['p']) - Y) / rn['sig'])**2))
    out['weighting'][label] = {'null_chi2': rn['chi2'], 'null_chi2_fit_weights': rn['chi2_fit_weights'], 'symfree_chi2': rs['chi2'],
                               'symfree_lambda': float(rs['p'][3]), 'split_chi2': rp['chi2'], 'split_lambda_b': float(rp['p'][2]),
                               'split_lambda_b_err': float(rp['e'][2]), 'dchi2_common_symfree': rn['chi2'] - cs_,
                               'dchi2_common_split': rn['chi2'] - cp_}
    say('%-24s null %.3f (fit weights %.3f) | symfree %.3f lambda %.3f | split %.3f lambda_b %.3f +/- %.3f | dchi2 common sym %.2f split %.2f'
        % (label, rn['chi2'], rn['chi2_fit_weights'], rs['chi2'], rs['p'][3], rp['chi2'], rp['p'][2], rp['e'][2], rn['chi2'] - cs_, rn['chi2'] - cp_))

# ---------------------------------------------------------------- derived quantities from results.json parameters
say('\n=== derived quantities recomputed by trapezoidal integration ===')
out['derived'] = {}
for dep_name in ('long', 'aware'):
    fs = {k: make(k, DEP[dep_name]) for k in NAMES}
    pn = ref_p(dep_name, 'null')
    sig_null = np.sqrt(E**2 + (slope_at(fs['null'], pn, 'axis') * TFE * T)**2)
    common = {k: float(np.sum(((fs[k](T, ref_p(dep_name, k)) - Y) / sig_null)**2)) for k in NAMES}
    d = {'chi2_common': common}
    for k in ('symfree', 'split'):
        d['dchi2_common_' + k] = common['null'] - common[k]
        say('%-5s %-7s chi2_common trap %.4f refit %.4f | dchi2 trap %.4f refit %.4f'
            % (dep_name, k, common[k], REF[dep_name]['baseline'][k]['chi2_common'], d['dchi2_common_' + k],
               REF[dep_name]['baseline'][k]['dchi2_common_vs_null']))
    say('%-5s null    chi2_common trap %.4f refit %.4f' % (dep_name, common['null'], REF[dep_name]['baseline']['null']['chi2_common']))
    for k in NAMES:
        nm, pi = peak_of(make(k, DEP[dep_name], n=3001), ref_p(dep_name, k))
        d['peak_' + k] = [nm, pi]
        say('%-5s %-7s model maximum trap %.2f nm, I %.5f | refit %.2f nm, I %.5f' % (
            dep_name, k, nm, pi, REF[dep_name]['baseline'][k]['peak_nm'], REF[dep_name]['baseline'][k]['peak_I']))
    pi = d['peak_split'][1]
    d['total_over_blocked'], d['exit_over_blocked'] = pi / BLOCKED, pi / BLOCKED - 1
    say('%-5s maximum / blocked mean 0.2361: trap %.4f (minus 1: %.4f) | refit %.4f (%.4f)' % (
        dep_name, pi / BLOCKED, pi / BLOCKED - 1, REF[dep_name]['derived']['total_to_blocked_ratio_at_model_max'],
        REF[dep_name]['derived']['exit_to_blocked_ratio_at_model_max']))
    lam_b, e_b = REF[dep_name]['baseline']['split']['lambda_b']
    d['lambda_b_scale15'], d['lambda_b_scale20'] = float(np.hypot(e_b, 0.15 * lam_b)), float(np.hypot(e_b, 0.20 * lam_b))
    say('%-5s lambda_b %.4f +/- %.4f; with 15%% scale %.4f (refit %.4f); with 20%% scale %.4f (refit %.4f)' % (
        dep_name, lam_b, e_b, d['lambda_b_scale15'], REF[dep_name]['derived']['lambda_b_with_scale_15pct'],
        d['lambda_b_scale20'], REF[dep_name]['derived']['lambda_b_with_scale_20pct']))
    ps = ref_p(dep_name, 'split')
    ent = {x: float(ps[0] * integrals(DEP[dep_name], x, ps[3], LAM_ESC, ps[2])[0][0]) for x in (14.1, 54.1, 100.0, 212.0)}
    ext = {x: float(ps[0] * ps[1] * integrals(DEP[dep_name], x, ps[3], LAM_ESC, ps[2])[1][0]) for x in (14.1, 54.1, 100.0, 191.2, 212.0)}
    d['entrance_term'], d['exit_term'] = ent, ext
    say('%-5s entrance term trap %s | refit %s' % (dep_name, {k: round(v, 5) for k, v in ent.items()},
                                                   {k: round(v, 5) for k, v in REF[dep_name]['derived']['entrance_term_at_t_nm'].items()}))
    say('%-5s exit term trap %s | refit at 212 nm %.6f' % (dep_name, {k: round(v, 5) for k, v in ext.items()},
                                                           REF[dep_name]['derived']['exit_term_at_212nm']))
    out['derived'][dep_name] = d

# ---------------------------------------------------------------- the deposition model itself
say('\n=== D(z; x) kernel validation (validate_model) ===')
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tsi.validate_model(info)
say(buf.getvalue().rstrip())

say('\n=== D(z; x) against the raw finite-thickness SRIM runs ===')
say('ratio R = D_run / D_long per bin (last bin excluded); "exit zone" = last 10 % of the film')
say(' t_raw   rms(model-raw)  mean(model-raw) exit zone   max|dev|   raw R last kept bin   model R there   raw R of dropped last bin')
out['runs'] = {}
S_SPLIT = REF['aware']['baseline']['split']['s'][0]
lam_raw = {'entrance/null 2 nm': LAM_ESC / S_SPLIT, 'symfree %.2f nm' % REF['aware']['baseline']['symfree']['lambda'][0]:
           REF['aware']['baseline']['symfree']['lambda'][0] / REF['aware']['baseline']['symfree']['s'][0],
           'split exit %.2f nm' % REF['aware']['baseline']['split']['lambda_b'][0]: REF['aware']['baseline']['split']['lambda_b'][0] / S_SPLIT}
for t_run, z_run, d_run in sweep:
    keep = np.ones(len(z_run), bool)
    keep[-1] = False
    raw = d_run / dlong(z_run)
    while raw[keep][-1] < 0.8:
        keep[np.flatnonzero(keep)[-1]] = False
    z, r_raw = z_run[keep], raw[keep]
    r_mod = ratio_eval(z, t_run)
    dev = r_mod - r_raw
    zone = z >= 0.9 * t_run
    width = t_run / 100.0
    rec = {'rms': float(np.sqrt(np.mean(dev**2))), 'mean_dev_exit_zone': float(np.mean(dev[zone])), 'max_abs_dev': float(np.max(np.abs(dev))),
           'raw_last_kept': float(r_raw[-1]), 'model_last_kept': float(r_mod[-1]), 'raw_dropped_last_bin': float(raw[-1]), 'integrals': {}}
    say('%6g   %.4f          %+.4f                     %.4f     %.4f                %.4f          %.4f'
        % (t_run, rec['rms'], rec['mean_dev_exit_zone'], rec['max_abs_dev'], rec['raw_last_kept'], rec['model_last_kept'], rec['raw_dropped_last_bin']))
    for label, lam in lam_raw.items():
        w_exit = np.exp(-(z[-1] - z) / lam) * width
        w_ent = np.exp(-z / lam) * width
        e_raw, e_mod, e_long = (float(np.sum(w_exit * v)) for v in (d_run[keep], r_mod * dlong(z), dlong(z)))
        f_raw, f_mod, f_long = (float(np.sum(w_ent * v)) for v in (d_run[keep], r_mod * dlong(z), dlong(z)))
        rec['integrals'][label] = {'lam_raw_nm': lam, 'exit_model_over_raw': e_mod / e_raw, 'exit_long_over_raw': e_long / e_raw,
                                   'entrance_model_over_raw': f_mod / f_raw, 'entrance_long_over_raw': f_long / f_raw}
    out['runs']['%g' % t_run] = rec
say('\nescape-weighted sums over the SRIM bins of each run: (model D)/(raw run) - 1 and (long D)/(raw run) - 1, in percent')
for label, lam in lam_raw.items():
    say('  weight %s (%.2f nm on the SRIM scale)' % (label, lam))
    say('   t_raw   exit: model   exit: long   entrance: model   entrance: long')
    for t_run, _, _ in sweep:
        q = out['runs']['%g' % t_run]['integrals'][label]
        say('  %6g   %+7.2f       %+7.2f      %+7.2f           %+7.2f' % (t_run, 100 * (q['exit_model_over_raw'] - 1), 100 * (q['exit_long_over_raw'] - 1),
                                                                         100 * (q['entrance_model_over_raw'] - 1), 100 * (q['entrance_long_over_raw'] - 1)))

say('\n=== where the 15 data points sit on the SRIM scale (x / s) ===')
out['coverage'] = {}
for kind in ('split', 'symfree', 'null'):
    s = REF['aware']['baseline'][kind]['s'][0]
    xr = T / s
    cover = {'s': s, 'x_over_s': [float(v) for v in xr], 'n_within_sweep_10_200': int(np.sum((xr >= 10) & (xr <= 200))),
             'n_between_200_and_300': int(np.sum((xr > 200) & (xr <= T_LONG))), 'n_beyond_300': int(np.sum(xr > T_LONG)),
             'measured_nm_at_200': 200 * s, 'measured_nm_at_300': T_LONG * s}
    out['coverage'][kind] = cover
    say('%-7s s = %.4f: x/s = %s' % (kind, s, np.array2string(xr, precision=1)))
    say('        inside the sweep (10-200 nm): %d points; between the 200 nm run and the 300 nm target: %d (measured %s nm); beyond 300 nm: %d (measured %s nm)'
        % (cover['n_within_sweep_10_200'], cover['n_between_200_and_300'], T[(xr > 200) & (xr <= T_LONG)], cover['n_beyond_300'], T[xr > T_LONG]))
say('exit-face reduction applied by the model at each data point, D(x/s; x/s) / D_long(x/s), split s:')
face = [float(dep_aware(np.array([[x / S_SPLIT]]), np.array([[x / S_SPLIT]]))[0, 0] / dlong(x / S_SPLIT)) for x in T]
out['exit_face_ratio_at_data'] = dict(zip(map(str, T), face))
say('   ' + ', '.join('%g nm: %.3f' % (x, v) for x, v in zip(T, face)))
say('deficit amplitude A(z) and kernel length lam(z) of the fitted model: ' + ', '.join(
    'z=%g: A %.3f, lam %.1f nm' % (z, theta[0] + theta[1] * z + theta[2] * z**2, theta[3] * np.exp(theta[4] * z)) for z in (20, 100, 200, 250, 300)))

(OUT / 'verify_results.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8', newline='\n')
(OUT / 'verify_stdout.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8', newline='\n')

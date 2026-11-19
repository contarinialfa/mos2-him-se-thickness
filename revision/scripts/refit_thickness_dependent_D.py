# -*- coding: utf-8 -*-
"""Fit the three SE-escape models using the thickness-dependent SRIM deposition D(z; x).

For a flake of measured thickness x, the entrance and exit escape integrals are evaluated on the
deposition profile of a target of that thickness, D(z/s; x/s), rather than on the single
thick-target (300 nm) profile D_long(z/s) used in the initial submission.

Every fit is evaluated for both deposition models through the same per-point quadrature:
  - "long": single 300 nm SRIM profile across all thicknesses;
  - "aware": finite-thickness deposition model of scripts/thickness_sweep_ionization.py,
    fitted to the nine SRIM runs (10-200 nm) in data/SRIM-Thickness-Series/.
The "long" baseline is also compared against the stored reference table from v1.0.0
(revision/inputs/long_target_error_budget.csv).

Fit procedure: exponential escape with exact per-cell kernels, effective-variance weights combining
the 68% intensity errors and the 10% fractional thickness error via the local model slope (three
iterations), and common weighting from the symmetric fit for model comparison. Each thickness is
integrated on a 2048-point grid.

Outputs (in --out): results.json, run_log.txt, deposition_model_stdout.txt.

Usage (from the repository root):
    python revision/scripts/refit_thickness_dependent_D.py
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, signal, stats

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
IONIZ = ROOT / 'data/SRIM 30 keV - Monolayer Long/IONIZ.txt'
TRANSMISSION = ROOT / 'data/reference_summaries/transmission_bayesian_summary.csv'
SWEEP_DIR = ROOT / 'data/SRIM-Thickness-Series'
LONG_TARGET = ROOT / 'revision/inputs/long_target_error_budget.csv'

parser = argparse.ArgumentParser(description='Refit of the SE-escape models with thickness-dependent D(z; x).')
parser.add_argument('--out', type=Path, default=ROOT / 'revision/outputs/refit_thickness_dependent_D',
                    help='output folder (default: revision/outputs/refit_thickness_dependent_D)')
OUT = parser.parse_args().out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
LAMBDA_ESC, TFE, N_GRID, BLOCKED_MEAN = 2.0, 0.10, 2048, 0.2361

sys.path.insert(0, str(ROOT / 'scripts'))
import thickness_sweep_ionization as tsi  # noqa: E402

depth, ion = tsi.load_ionization(IONIZ)
frame = pd.read_csv(TRANSMISSION)
frame['err'] = 0.5 * (frame.normalized_err_lower_68 + frame.normalized_err_upper_68)
frame = frame.sort_values('thickness_nm').reset_index(drop=True)
T0, Y0, E0 = frame.thickness_nm.to_numpy(float), frame.normalized_median.to_numpy(float), frame.err.to_numpy(float)
IS_F3 = (frame.dataset == 'Dataset 3').to_numpy()
assert len(T0) == 15 and IS_F3.sum() == 4

d_long = interpolate.interp1d(depth, ion, kind='cubic', fill_value=(ion[0], ion[-1]), bounds_error=False)
sweep_files = tsi.discover_sweep(SWEEP_DIR)
sweep = [(thickness, *tsi.load_ionization(path)) for thickness, path in sweep_files]
with contextlib.redirect_stdout(io.StringIO()) as captured:
    d_aware, model_info = tsi.build_ionization_model(depth, ion, sweep)
(OUT / 'deposition_model_stdout.txt').write_text(captured.getvalue(), encoding='utf-8', newline='\n')
DEPOSITION = {'long': lambda z, x_raw: d_long(z), 'aware': d_aware}
AXIS = np.linspace(0, float(np.max(depth)), 500)


def escape_terms(s, t, dt, lam):
    a = dt / lam
    e, phi = np.exp(-a), -np.expm1(-a) / a
    c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
    back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
    psi = (phi - e) / a
    cell = np.exp(-t[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
    return float(np.sum(cell)), float(back[-1])


def terms(dep, x, stretch, lam_front, lam_back):
    """Entrance and exit escape integrals for one measured thickness x."""
    if x <= 0:
        return 0.0, 0.0
    t, dt = np.linspace(0, x, N_GRID, retstep=True)
    s = dep(t / stretch, x / stretch)
    front = escape_terms(s, t, dt, lam_front)[0]
    back = escape_terms(s, t, dt, lam_back)[1]
    return front, back


def make_models(dep, lam_esc):
    def general(x_array, C_o, C_fb, stretch, lam_front, lam_back):
        x_array = np.atleast_1d(np.asarray(x_array, float))
        out = np.zeros(x_array.shape)
        for i, x in enumerate(x_array):
            front, back = terms(dep, x, stretch, lam_front, lam_back)
            out[i] = C_o * (front + C_fb * back)
        return out
    return {'null': lambda x, C_o, C_fb, s: general(x, C_o, C_fb, s, lam_esc, lam_esc),
            'symfree': lambda x, C_o, C_fb, s, lam: general(x, C_o, C_fb, s, lam, lam),
            'split': lambda x, C_o, C_fb, lam_b, s: general(x, C_o, C_fb, s, lam_esc, lam_b)}


def wfit(model, p0, bounds, t, y, err, tfe):
    sig = err.copy()
    for _ in range(3):
        resid = lambda p: (model(t, *p) - y) / sig  # noqa: E731
        fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac', ftol=1e-12, xtol=1e-12)
        slope = np.interp(t, AXIS, np.gradient(model(AXIS, *fit.x), AXIS))
        sig = np.sqrt(err**2 + (slope * tfe * t)**2)
        p0 = fit.x
    cov = np.linalg.inv(fit.jac.T @ fit.jac)
    return fit.x, np.sqrt(np.diag(cov)), float(np.sum(resid(fit.x)**2)), sig


def run(dep_name, t=T0, y=Y0, err=E0, tfe=TFE, lam_esc=LAMBDA_ESC, which=('null', 'symfree', 'split'), peak=False):
    models = make_models(DEPOSITION[dep_name], lam_esc)
    n, res = len(t), {}
    opt, perr, chi2, sig_null = wfit(models['null'], [0.01, 3.0, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]), t, y, err, tfe)
    res['null'] = {'p': opt, 'e': perr, 'chi2': chi2, 'dof': n - 3, 'names': ['C_o', 'C_b', 's']}
    null_opt = opt
    if 'symfree' in which:
        opt, perr, chi2, _ = wfit(models['symfree'], [*null_opt, lam_esc], ([0, 0, 0.2, 0.2], [np.inf, np.inf, 3, 50]), t, y, err, tfe)
        res['symfree'] = {'p': opt, 'e': perr, 'chi2': chi2, 'dof': n - 4, 'names': ['C_o', 'C_b', 's', 'lambda']}
    if 'split' in which:
        opt, perr, chi2, _ = wfit(models['split'], [*null_opt[:2], lam_esc, null_opt[2]], ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]), t, y, err, tfe)
        res['split'] = {'p': opt, 'e': perr, 'chi2': chi2, 'dof': n - 4, 'names': ['C_o', 'C_b', 'lambda_b', 's']}
    for key, r in res.items():
        common = ((models[key](t, *r['p']) - y) / sig_null)**2
        r['chi2_common'] = float(np.sum(common))
        r['aic_common'] = r['chi2_common'] + 2 * len(r['p'])
        r['p_gof'] = float(stats.chi2.sf(r['chi2'], r['dof']))
        r['common_terms'] = common
        if peak:
            grid = np.arange(5.0, 100.0, 0.02)
            curve = models[key](grid, *r['p'])
            r['peak_nm'], r['peak_I'] = float(grid[np.argmax(curve)]), float(np.max(curve))
    res['_models'] = models
    return res


def flat(r):
    return {name: [float(v), float(e)] for name, v, e in zip(r['names'], r['p'], r['e'])}


results, lines = {}, []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


for dep_name in ('long', 'aware'):
    R = {}
    base = run(dep_name, peak=True)
    models = base['_models']
    R['baseline'] = {k: {**flat(base[k]), 'chi2': base[k]['chi2'], 'dof': base[k]['dof'], 'chi2_common': base[k]['chi2_common'],
                         'aic_common': base[k]['aic_common'], 'p_gof': base[k]['p_gof'], 'peak_nm': base[k]['peak_nm'],
                         'peak_I': base[k]['peak_I']} for k in ('null', 'symfree', 'split')}
    for k in ('symfree', 'split'):
        d = base['null']['chi2_common'] - base[k]['chi2_common']
        R['baseline'][k]['dchi2_common_vs_null'] = d
        R['baseline'][k]['sigma_equivalent'] = float(np.sqrt(d))
        R['baseline'][k]['dchi2_own_vs_null'] = base['null']['chi2'] - base[k]['chi2']
    sp = base['split']
    C_o, C_b, lam_b, s = sp['p']
    R['derived'] = {
        'Cb_lamb_over_lamf': float(C_b * lam_b / LAMBDA_ESC),
        'symfree_Cb_lam_over_lam': float(base['symfree']['p'][1]),
        'exit_to_blocked_ratio_at_model_max': sp['peak_I'] / BLOCKED_MEAN - 1,
        'total_to_blocked_ratio_at_model_max': sp['peak_I'] / BLOCKED_MEAN,
        'entrance_term_at_t_nm': {str(x): float(C_o * terms(DEPOSITION[dep_name], x, s, LAMBDA_ESC, lam_b)[0]) for x in (14.1, 54.1, 100.0, 212.0)},
        'exit_term_at_212nm': float(C_o * C_b * terms(DEPOSITION[dep_name], 212.0, s, LAMBDA_ESC, lam_b)[1]),
        'lambda_b_with_scale_15pct': float(np.hypot(sp['e'][2], 0.15 * lam_b)),
        'lambda_b_with_scale_20pct': float(np.hypot(sp['e'][2], 0.20 * lam_b)),
        'null_chi2_common_by_dataset': {name: float(np.sum(base['null']['common_terms'][(frame.dataset == name).to_numpy()]))
                                        for name in ('Dataset 1', 'Dataset 2', 'Dataset 3')},
    }
    say('%s: baseline done (null %.2f/12, symfree %.2f/11, split %.2f/11, lambda_b %.3f +/- %.3f)'
        % (dep_name, base['null']['chi2'], base['symfree']['chi2'], sp['chi2'], lam_b, sp['e'][2]))
    R['scale'] = {}
    for kappa in (0.80, 0.85, 1.15, 1.20):
        r = run(dep_name, t=kappa * T0, which=('null', 'split'), peak=True)
        R['scale']['%.2f' % kappa] = {'null_chi2': r['null']['chi2'], 'split_chi2': r['split']['chi2'], **flat(r['split']),
                                      'peak_nm': r['split']['peak_nm']}
    say('%s: scale sweep done' % dep_name)
    R['point_thickness_error'] = {}
    for tfe in (0.05, 0.15, 0.20):
        r = run(dep_name, tfe=tfe, which=('null', 'split'))
        R['point_thickness_error']['%.2f' % tfe] = {'null_chi2': r['null']['chi2'], 'null_s': float(r['null']['p'][2]),
                                                    'split_chi2': r['split']['chi2'], **flat(r['split'])}
    say('%s: point thickness error done' % dep_name)
    R['flake3_rescaled'] = {}
    for k3 in (0.80, 1.20):
        t_mod = T0.copy()
        t_mod[IS_F3] *= k3
        order = np.argsort(t_mod)
        r = run(dep_name, t=t_mod[order], y=Y0[order], err=E0[order], which=('null', 'split'))
        R['flake3_rescaled']['%.2f' % k3] = {'null_chi2': r['null']['chi2'], 'split_chi2': r['split']['chi2'], **flat(r['split'])}
    keep = ~IS_F3
    r = run(dep_name, t=T0[keep], y=Y0[keep], err=E0[keep], which=('null',))
    R['without_flake3'] = {'null_chi2': r['null']['chi2'], 'dof': r['null']['dof'], 'p_gof': r['null']['p_gof']}
    R['leave_one_out_null'] = []
    for i in range(len(T0)):
        keep = np.arange(len(T0)) != i
        r = run(dep_name, t=T0[keep], y=Y0[keep], err=E0[keep], which=('null',))
        R['leave_one_out_null'].append({'dropped_nm': float(T0[i]), 'chi2': r['null']['chi2'], 'p_gof': r['null']['p_gof']})
    say('%s: Flake 3 tests done' % dep_name)
    R['entrance_scale'] = {}
    for lam_esc in (1.5, 2.5):
        r = run(dep_name, lam_esc=lam_esc, which=('null', 'split'))
        R['entrance_scale']['%.1f' % lam_esc] = {'split_chi2': r['split']['chi2'], **flat(r['split'])}
    say('%s: entrance scale done' % dep_name)
    results[dep_name] = R

# Compare the "long" baseline against the stored v1.0 long-target fit values
pub = pd.read_csv(LONG_TARGET)
key = {'null': 'null_lambda2', 'symfree': 'symfree', 'split': 'split'}
check = []
for k, mk in key.items():
    row = pub[(pub.variant == 'a_baseline') & (pub.model_key == mk)].iloc[0]
    cur = results['long']['baseline'][k]
    check.append((k + ' chi2', cur['chi2'], row.chi2))
    check.append((k + ' chi2_common', cur['chi2_common'], row.chi2_common_sigma))
    check.append((k + ' C_b', cur['C_b'][0], row.C_b))
    check.append((k + ' s', cur['s'][0], row.s))
    check.append((k + ' peak_nm', cur['peak_nm'], row.model_peak_nm))
check.append(('split lambda_b', results['long']['baseline']['split']['lambda_b'][0], pub[(pub.variant == 'a_baseline') & (pub.model_key == 'split')].iloc[0].lambda_b_nm))
check.append(('split lambda_b_err', results['long']['baseline']['split']['lambda_b'][1], pub[(pub.variant == 'a_baseline') & (pub.model_key == 'split')].iloc[0].lambda_b_err_nm))
for variant, group, sub in (('b_scale_k0.80', 'scale', '0.80'), ('b_scale_k1.20', 'scale', '1.20'), ('c_tfe05', 'point_thickness_error', '0.05'),
                            ('c_tfe20', 'point_thickness_error', '0.20')):
    row_n = pub[(pub.variant == variant) & (pub.model_key == 'null_lambda2')].iloc[0]
    row_s = pub[(pub.variant == variant) & (pub.model_key == 'split')].iloc[0]
    cur = results['long'][group][sub]
    check.append((variant + ' null chi2', cur['null_chi2'], row_n.chi2))
    check.append((variant + ' split chi2', cur['split_chi2'], row_s.chi2))
    check.append((variant + ' lambda_b', cur['lambda_b'][0], row_s.lambda_b_nm))
say()
say('Comparison of long-target baseline against v1.0 reference values:')
worst = 0.0
for name, cur_val, published in check:
    rel = abs(cur_val - published) / max(abs(published), 1e-12)
    worst = max(worst, rel)
    say('   %-34s current %10.4f   v1.0 %10.4f   rel. diff %.1e' % (name, cur_val, published, rel))
say('   largest relative difference: %.1e' % worst)
results['reproduction_of_published_largest_relative_difference'] = worst
results['inputs'] = {'deposition_kernel': model_info['kernel_name']}
(OUT / 'results.json').write_text(json.dumps(results, indent=1, default=float), encoding='utf-8', newline='\n')
(OUT / 'run_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

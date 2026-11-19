# -*- coding: utf-8 -*-
"""Supplementary fits and sensitivity checks of the SE emission models (SI Secs. SVII, SVIII, and SIX).

Evaluates each fit under both SRIM deposition profiles:
  - "long": single 300 nm SRIM target, D_300(z);
  - "aware": finite-thickness profile D(z; x), constructed from the 300 nm run and the
    nine-run SRIM thickness series via scripts/thickness_sweep_ionization.py.

Computes:
  1. within-image correlations of the posterior ratio samples (SI Sec. SVII);
  2. chi2 and strongest parameter correlations for the symmetric (null), symmetric
     free-lambda, and asymmetric fits (SI Sec. SVIII);
  3. five-parameter fit with both escape depths and C_b free (SI Sec. SVIII);
  4. asymmetric fit by generalized least squares under the within-image posterior
     covariance (SI Sec. SVIII, item ii);
  5. forward-warp deposition fit and its maximum deviation from the asymmetric family
     (SI Sec. SVIII);
  6. slope of the asymmetric curve along the falling branch (SI Sec. SIX).

Usage (from the repository root):
    python revision/scripts/refit_supplement.py [--out FOLDER]
"""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, signal, stats  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
IONIZ = ROOT / 'data' / 'SRIM 30 keV - Monolayer Long' / 'IONIZ.txt'
TRANSMISSION = ROOT / 'data' / 'reference_summaries' / 'transmission_bayesian_summary.csv'
SWEEP_DIR = ROOT / 'data' / 'SRIM-Thickness-Series'
COV_SCRIPT = HERE / 'within_image_covariance.py'
DEFAULT_OUT = ROOT / 'revision' / 'outputs' / 'refit_supplement'

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='output folder (default: %(default)s)')
OUT = parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)

LAMBDA_ESC, TFE, N_GRID = 2.0, 0.10, 2048
sys.path.insert(0, str(ROOT / 'scripts'))
import thickness_sweep_ionization as tsi  # noqa: E402

depth, ion = tsi.load_ionization(IONIZ)
frame = pd.read_csv(TRANSMISSION)
frame['err'] = 0.5 * (frame.normalized_err_lower_68 + frame.normalized_err_upper_68)
frame = frame.sort_values('thickness_nm').reset_index(drop=True)
T, Y, E = frame.thickness_nm.to_numpy(float), frame.normalized_median.to_numpy(float), frame.err.to_numpy(float)
d_long = interpolate.interp1d(depth, ion, kind='cubic', fill_value=(ion[0], ion[-1]), bounds_error=False)
sweep = [(thickness, *tsi.load_ionization(path)) for thickness, path in tsi.discover_sweep(SWEEP_DIR)]
with contextlib.redirect_stdout(io.StringIO()):
    d_aware, _ = tsi.build_ionization_model(depth, ion, sweep)
DEPOSITION = {'long': lambda z, x_raw: d_long(z), 'aware': d_aware}
AXIS = np.linspace(0, float(np.max(depth)), 500)
lines, results = [], {}


def say(text=''):
    print(text, flush=True)
    lines.append(text)


def escape_terms(s, grid, dt, lam):
    """Front (entrance-surface) and back (exit-surface) escape integrals of the source s on the grid."""
    a = dt / lam
    e, phi = np.exp(-a), -np.expm1(-a) / a
    c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
    back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
    psi = (phi - e) / a
    cell = np.exp(-grid[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
    return float(np.sum(cell)), float(back[-1])


def general(dep, x_array, C_o, C_b, stretch, lam_front, lam_back, warp=None):
    x_array = np.atleast_1d(np.asarray(x_array, float))
    out = np.zeros(x_array.shape)
    for i, x in enumerate(x_array):
        if x <= 0:
            continue
        grid, dt = np.linspace(0, x, N_GRID, retstep=True)
        z = grid / stretch
        if warp is None:
            s = dep(z, x / stretch)
        else:  # forward warp of the profile, as d1p_model in scripts/thickness_aware_fits.py
            delta_inf, rho = warp
            decay = np.exp(-z / (delta_inf / rho))
            s = (1 - rho * decay) * dep(z - delta_inf * (1 - decay), x / stretch)
        out[i] = C_o * (escape_terms(s, grid, dt, lam_front)[0] + C_b * escape_terms(s, grid, dt, lam_back)[1])
    return out


def wfit(model, p0, bounds):
    """Effective-variance fit: three passes of bounded least squares with the thickness term of sigma_eff."""
    sig = E.copy()
    for _ in range(3):
        resid = lambda p: (model(T, *p) - Y) / sig  # noqa: E731
        fit = optimize.least_squares(resid, p0, bounds=bounds, x_scale='jac', ftol=1e-12, xtol=1e-12)
        slope = np.interp(T, AXIS, np.gradient(model(AXIS, *fit.x), AXIS))
        sig = np.sqrt(E**2 + (slope * TFE * T)**2)
        p0 = fit.x
    cov = np.linalg.inv(fit.jac.T @ fit.jac)
    err = np.sqrt(np.diag(cov))
    return fit.x, err, cov / np.outer(err, err), float(np.sum(resid(fit.x)**2)), sig


def strongest(corr, names):
    c = corr - np.eye(len(names))
    i, j = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    return '%s-%s' % (names[min(i, j)], names[max(i, j)]), float(c[i, j])


# 1. within-image correlations of the posterior ratio samples
spec = importlib.util.spec_from_file_location('within_image_covariance', COV_SCRIPT)
cov_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov_module)
correlation, _ = cov_module.build_correlation(frame)
off = np.concatenate([correlation[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
                      for idx in (np.flatnonzero((frame.dataset == d).to_numpy()) for d in ('Dataset 1', 'Dataset 2', 'Dataset 3'))])
results['within_image_correlations'] = {'min': float(off.min()), 'max': float(off.max()), 'median': float(np.median(off))}
say('within-image correlations of the ratio samples: %.2f to %.2f, median %.2f'
    % (off.min(), off.max(), np.median(off)))

for name in ('long', 'aware'):
    dep = DEPOSITION[name]
    R = {}
    null = lambda x, C_o, C_b, s: general(dep, x, C_o, C_b, s, LAMBDA_ESC, LAMBDA_ESC)  # noqa: E731
    symfree = lambda x, C_o, C_b, s, lam: general(dep, x, C_o, C_b, s, lam, lam)  # noqa: E731
    split = lambda x, C_o, C_b, lam_b, s: general(dep, x, C_o, C_b, s, LAMBDA_ESC, lam_b)  # noqa: E731
    five = lambda x, C_o, C_b, lam_b, s, lam_f: general(dep, x, C_o, C_b, s, lam_f, lam_b)  # noqa: E731
    d1p = lambda x, A, s, delta_inf, rho: general(dep, x, A, 1.0, s, LAMBDA_ESC, LAMBDA_ESC, warp=(delta_inf, rho))  # noqa: E731

    # 2. the three model fits: chi2 and strongest parameter correlation
    n_opt, n_err, n_corr, n_chi, sig_null = wfit(null, [0.01, 3.0, 0.68], ([0, 0, 0.2], [np.inf, np.inf, 3]))
    f_opt, f_err, f_corr, f_chi, _ = wfit(symfree, [*n_opt, LAMBDA_ESC], ([0, 0, 0.2, 0.2], [np.inf, np.inf, 3, 50]))
    s_opt, s_err, s_corr, s_chi, sig_split = wfit(split, [*n_opt[:2], LAMBDA_ESC, n_opt[2]], ([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]))
    R['correlations'] = {'null': strongest(n_corr, ['C_o', 'C_b', 's']), 'symfree': strongest(f_corr, ['C_o', 'C_b', 's', 'lambda']),
                         'split': strongest(s_corr, ['C_o', 'C_b', 'lambda_back', 's'])}
    R['chi2'] = {'null': n_chi, 'symfree': f_chi, 'split': s_chi}
    say('%s: chi2 %.2f / %.2f / %.2f; strongest correlations %s' % (name, n_chi, f_chi, s_chi, R['correlations']))

    # 3. five parameters: both escape depths and C_b free
    p_opt, p_err, _, p_chi, _ = wfit(five, [*s_opt, LAMBDA_ESC], ([0, 0, 0.2, 0.2, 0.2], [np.inf, np.inf, 50, 3, 50]))
    R['five_parameter'] = {'chi2': p_chi, 'C_b': [float(p_opt[1]), float(p_err[1])], 'lambda_back': [float(p_opt[2]), float(p_err[2])],
                           's': [float(p_opt[3]), float(p_err[3])], 'lambda_front': [float(p_opt[4]), float(p_err[4])]}
    say('%s: five-parameter fit chi2 %.2f, C_b %.1f +/- %.1f, lambda_front %.0f +/- %.0f nm, lambda_back %.1f +/- %.1f nm'
        % (name, p_chi, p_opt[1], p_err[1], p_opt[4], p_err[4], p_opt[2], p_err[2]))

    # 4. asymmetric fit under the within-image posterior covariance (frozen-slope effective covariance)
    def gls(covariance, start):
        chol = np.linalg.cholesky(covariance)
        fit = optimize.least_squares(lambda p: np.linalg.solve(chol, split(T, *p) - Y), start,
                                     bounds=([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]), x_scale='jac', ftol=1e-12, xtol=1e-12)
        err = np.sqrt(np.diag(np.linalg.inv(fit.jac.T @ fit.jac)))
        return fit.x, err, float(np.sum(fit.fun**2))
    vertical = correlation * np.outer(E, E)
    pre, _, _ = gls(vertical, s_opt)
    h = np.maximum(1e-3, 1e-4 * T)
    slope = (split(T + h, *pre) - split(T - h, *pre)) / (2 * h)
    g_opt, g_err, g_chi = gls(vertical + np.diag((slope * TFE * T)**2), pre)
    R['posterior_covariance'] = {'chi2': g_chi, 'C_b': [float(g_opt[1]), float(g_err[1])], 'lambda_back': [float(g_opt[2]), float(g_err[2])],
                                 's': [float(g_opt[3]), float(g_err[3])],
                                 'shift_in_sigma_of_diagonal_fit': {'C_b': float((g_opt[1] - s_opt[1]) / s_err[1]),
                                                                    'lambda_back': float((g_opt[2] - s_opt[2]) / s_err[2]),
                                                                    's': float((g_opt[3] - s_opt[3]) / s_err[3])}}
    say('%s: posterior-covariance fit chi2 %.2f, C_b %.2f +/- %.2f, lambda_back %.1f +/- %.1f nm, s %.2f +/- %.2f'
        % (name, g_chi, g_opt[1], g_err[1], g_opt[2], g_err[2], g_opt[3], g_err[3]))

    # 5. forward warp and its mismatch with the asymmetric family (least-squares projection of the
    #    forward-warp curve onto the asymmetric model)
    w_opt, w_err, _, w_chi, _ = wfit(d1p, [0.03, n_opt[2], 6.0, 0.7], ([0, 0.2, 0, 0.05], [np.inf, 3, 30, 0.95]))
    synth = d1p(T, *w_opt)
    proj = optimize.least_squares(lambda p: (split(T, *p) - synth) / E, [*n_opt[:2], LAMBDA_ESC, n_opt[2]],
                                  bounds=([0, 0, 0.2, 0.2], [np.inf, np.inf, 50, 3]), x_scale='jac', ftol=1e-12, xtol=1e-12)
    gap = split(T, *proj.x) - synth
    R['forward_warp'] = {'chi2': w_chi, 'chi2_common': float(np.sum(((synth - Y) / sig_null)**2)),
                         'delta_inf': [float(w_opt[2]), float(w_err[2])], 'rho': [float(w_opt[3]), float(w_err[3])],
                         'max_family_mismatch_I_n': float(np.max(np.abs(gap))), 'median_data_sigma': float(np.median(E)),
                         'mismatch_over_median_sigma': float(np.max(np.abs(gap)) / np.median(E)),
                         'projected_lambda_back': float(proj.x[2])}
    say('%s: forward warp chi2 %.2f; max mismatch with the asymmetric family %.4f in I_n = sigma/%.1f (median sigma %.4f)'
        % (name, w_chi, np.max(np.abs(gap)), np.median(E) / np.max(np.abs(gap)), np.median(E)))

    # 6. slope of the asymmetric curve on the falling branch
    grid = np.arange(60.0, 190.1, 1.0)
    curve = split(grid, *s_opt)
    local = -np.gradient(curve, grid)
    R['falling_branch'] = {'mean_slope_65_to_185nm': float((np.interp(65, grid, curve) - np.interp(185, grid, curve)) / 120),
                           'max_slope': float(local.max()), 'at_nm': float(grid[np.argmax(local)]),
                           'slope_at_nm': {str(v): float(np.interp(v, grid, local)) for v in (80, 100, 120, 140, 160)}}
    say('%s: falling branch mean slope 65-185 nm %.4f per nm; steepest %.4f at %.0f nm' % (name, R['falling_branch']['mean_slope_65_to_185nm'], local.max(), grid[np.argmax(local)]))
    R['stretch'] = {'null': float(n_opt[2]), 'symfree': float(f_opt[2]), 'split': float(s_opt[3]), 'posterior_covariance': float(g_opt[3])}
    results[name] = R

(OUT / 'results.json').write_text(json.dumps(results, indent=1, default=float), encoding='utf-8', newline='\n')
(OUT / 'run_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

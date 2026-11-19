# -*- coding: utf-8 -*-
"""Generate Fig. 3 of the revised manuscript: SE intensity versus thickness with the three SE-escape
model curves, common-weighting normalized residuals, and the exit-surface SRIM ionization
D(t/s; t/s)/D_max on the right-hand axis.

The model curves and residuals are evaluated from the thickness-dependent SRIM profile D(z; x)
using the fitted parameters in revision/outputs/refit_thickness_dependent_D/results.json
(produced by revision/scripts/refit_thickness_dependent_D.py). The right-hand axis plots the
exit-surface ionization D(x; x) normalized to its maximum and mapped onto the measured thickness
axis using the stretch factor s of the asymmetric fit.

Outputs (in --out):
    model_comparison.pdf, model_comparison.png, plotted_curves.csv,
    plotted_residuals.csv, plotted_Dxx_profile.csv, verification.json.

Usage (from the repository root):
    python revision/scripts/fig3_model_comparison.py
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
IONIZ = ROOT / 'data/SRIM 30 keV - Monolayer Long/IONIZ.txt'
TRANSMISSION = ROOT / 'data/reference_summaries/transmission_bayesian_summary.csv'
BLOCKED = ROOT / 'data/reference_summaries/backside_blocked_bayesian_summary.csv'
STYLE = ROOT / 'config/dlister_r01.mplstyle'
SWEEP_DIR = ROOT / 'data/SRIM-Thickness-Series'

parser = argparse.ArgumentParser(description='Fig. 3: model comparison with D(z; x) fits.')
parser.add_argument('--results', type=Path, default=ROOT / 'revision/outputs/refit_thickness_dependent_D/results.json',
                    help='results.json of the refit (default: revision/outputs/refit_thickness_dependent_D/results.json)')
parser.add_argument('--out', type=Path, default=ROOT / 'revision/outputs/fig3_model_comparison',
                    help='output folder (default: revision/outputs/fig3_model_comparison)')
args = parser.parse_args()
REFIT = args.results
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp(prefix='mplconfig_')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy import signal

sys.path.insert(0, str(ROOT / 'scripts'))
import thickness_sweep_ionization as tsi  # noqa: E402

depth, ion = tsi.load_ionization(IONIZ)
sweep = [(thickness, *tsi.load_ionization(path)) for thickness, path in tsi.discover_sweep(SWEEP_DIR)]
with contextlib.redirect_stdout(io.StringIO()):
    deposition, _ = tsi.build_ionization_model(depth, ion, sweep)  # D(z, x) on the SRIM scale

frame = pd.read_csv(TRANSMISSION)
frame['error_sym'] = 0.5 * (frame.normalized_err_lower_68 + frame.normalized_err_upper_68)
ordered_frame = frame.sort_values('thickness_nm')
t, y, error = (ordered_frame[c].to_numpy(float) for c in ('thickness_nm', 'normalized_median', 'error_sym'))
blocked = pd.read_csv(BLOCKED).sort_values('thickness_nm')
t_blk, y_blk = blocked.thickness_nm.to_numpy(float), blocked.normalized_median.to_numpy(float)
error_blk = 0.5 * (blocked.normalized_err_lower_68 + blocked.normalized_err_upper_68).to_numpy(float)
assert len(t) == 15 and len(t_blk) == 11

fit = json.loads(REFIT.read_text(encoding='utf-8'))['aware']['baseline']
LAMBDA_ESC, TFE, N_GRID = 2.0, 0.10, 2048


def escape_terms(s, grid, dt, lam):
    a = dt / lam
    e, phi = np.exp(-a), -np.expm1(-a) / a
    c1, c0 = dt * (1 - phi) / a, dt * (phi - (1 - phi) / a)
    back, _ = signal.lfilter([c1, c0], [1, -e], s, zi=[-c1 * s[0]])
    psi = (phi - e) / a
    cell = np.exp(-grid[:-1] / lam) * dt * ((phi - psi) * s[:-1] + psi * s[1:])
    return float(np.sum(cell)), float(back[-1])


def general(x_array, C_o, C_b, stretch, lam_front, lam_back):
    out = np.zeros(len(x_array))
    for i, x in enumerate(x_array):
        if x <= 0:
            continue
        grid, dt = np.linspace(0, x, N_GRID, retstep=True)
        s = deposition(grid / stretch, x / stretch)
        out[i] = C_o * (escape_terms(s, grid, dt, lam_front)[0] + C_b * escape_terms(s, grid, dt, lam_back)[1])
    return out


P = {k: {name: fit[k][name][0] for name in fit[k] if isinstance(fit[k][name], list)} for k in ('null', 'symfree', 'split')}
MODELS = {
    'null': lambda x: general(x, P['null']['C_o'], P['null']['C_b'], P['null']['s'], LAMBDA_ESC, LAMBDA_ESC),
    'symfree': lambda x: general(x, P['symfree']['C_o'], P['symfree']['C_b'], P['symfree']['s'], P['symfree']['lambda'], P['symfree']['lambda']),
    'split': lambda x: general(x, P['split']['C_o'], P['split']['C_b'], P['split']['s'], LAMBDA_ESC, P['split']['lambda_b']),
}
STYLES = {'null': ('Symmetric', '#686868', '--'), 'symfree': (r'Symmetric ($\lambda$ free)', '#cc0808', '-.'),
          'split': ('Asymmetric', '#2e0acc', ':')}
# Common weighting: effective errors from the converged symmetric (null) fit
axis = np.linspace(0, float(np.max(depth)), 500)
slope = np.interp(t, axis, np.gradient(MODELS['null'](axis), axis))
sig_eff = np.sqrt(error**2 + (slope * TFE * t)**2)
residuals = {k: (y - MODELS[k](t)) / sig_eff for k in MODELS}
chi2_common = {k: float(np.sum(residuals[k]**2)) for k in MODELS}
for k in MODELS:
    assert abs(chi2_common[k] - fit[k]['chi2_common']) < 2e-3, k

S = P['split']['s']  # depth scale of the right-hand curve: the asymmetric fit
x_srim = np.arange(depth[0], depth[-1] + 1e-9, 1.0)
D_exit = np.array([float(deposition(np.array([x]), float(x))[0]) for x in x_srim])
t_D, D_norm = S * x_srim, D_exit / D_exit.max()
curve_axis = np.linspace(0.0, 1.05 * np.max(t), 400)
curves = {k: MODELS[k](curve_axis) for k in MODELS}
pd.DataFrame({'t_nm': curve_axis, **{'I_n_' + k: curves[k] for k in MODELS}}).to_csv(OUT / 'plotted_curves.csv', index=False, lineterminator='\n')
pd.DataFrame({'t_nm': t, 'I_n': y, 'sigma_eff': sig_eff, **{'residual_' + k: residuals[k] for k in MODELS}}).to_csv(OUT / 'plotted_residuals.csv', index=False, lineterminator='\n')
pd.DataFrame({'x_srim_nm': x_srim, 't_nm': t_D, 'D_exit_over_its_max': D_norm}).to_csv(OUT / 'plotted_Dxx_profile.csv', index=False, lineterminator='\n')

COLUMN_RC = {'figure.autolayout': False, 'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 8,
             'ytick.labelsize': 8, 'legend.fontsize': 8.0}
FLAKES = [('Dataset 1', 'Flake 1', 'o', 'black'), ('Dataset 2', 'Flake 2', 's', '#007739'), ('Dataset 3', 'Flake 3', '^', '#cc7218')]
OPEN = {'markerfacecolor': 'none', 'markersize': 3.0, 'elinewidth': 0.6, 'linestyle': 'none', 'zorder': 3}
CAPS = {'capsize': 1.5, 'capthick': 0.6}
LEGEND = {'loc': 'lower center', 'bbox_to_anchor': (0.5, 1.01), 'ncol': 2, 'columnspacing': 0.8,
          'handlelength': 1.8, 'handletextpad': 0.5, 'borderpad': 0.35, 'labelspacing': 0.3}
ORDER = ['Flake 1', 'Flake 2', 'Flake 3', 'Back blocked', 'Symmetric', r'Symmetric ($\lambda$ free)', 'Asymmetric', 'SRIM ionization']
with plt.rc_context():
    matplotlib.style.use(str(STYLE))
    plt.rcParams.update(COLUMN_RC)
    plt.rcParams['path.simplify'] = False
    fig, (ax, res) = plt.subplots(2, 1, figsize=(3.37, 3.6), sharex=True, gridspec_kw={'height_ratios': (3, 1), 'hspace': 0.07})
    for dataset, label, marker, colour in FLAKES:
        f = frame[frame.dataset == dataset]
        ax.errorbar(f.thickness_nm, f.normalized_median, xerr=TFE * f.thickness_nm, yerr=f.error_sym, fmt=marker,
                    color=colour, label=label, **OPEN, **CAPS)
    ax.errorbar(t_blk, y_blk, xerr=TFE * t_blk, yerr=error_blk, fmt='D', color='black', markersize=2.6, elinewidth=0.6,
                linestyle='none', zorder=3, label='Back blocked', **CAPS)
    extent = 0.0
    for k, (label, colour, style) in STYLES.items():
        ax.plot(curve_axis, curves[k], color=colour, linestyle=style, linewidth=1.2, zorder=2, label=label)
        res.plot(t, residuals[k], marker='o', markersize=2.2, linestyle='none', color=colour, zorder=2)
        extent = max(extent, float(np.max(np.abs(residuals[k]))))
    res.axhline(0.0, color='#686868', linewidth=0.8, zorder=1)
    ax.set_ylabel(r'$I_n$')
    ax.set_xlim(0.0, curve_axis[-1])
    ax.set_ylim(0.0, 1.5110914216988326)  # match single-column intensity axis height
    res.set_xlabel(r'$t$ (nm)')
    res.set_ylabel(r'$\Delta I_n/\sigma$')
    res.set_ylim(-1.15 * extent, 1.15 * extent)
    handles, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    d_line, = ax2.plot(t_D, D_norm, color='black', linestyle='-', linewidth=0.7, label='SRIM ionization', zorder=1)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([0, .2, .4, .6, .8, 1])
    ax2.yaxis.set_minor_locator(FixedLocator([v / 20 for v in range(1, 20) if v % 4]))
    ax2.set_ylabel(r'$D(t/s;\,t/s)/D_\mathrm{max}$')
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', which='both', right=False)
    by_label = dict(zip(labels + ['SRIM ionization'], handles + [d_line]))
    ax.legend([by_label[name] for name in ORDER], ORDER, **LEGEND)
    fig.subplots_adjust(right=0.7726)
    fig.canvas.draw()
    fig.savefig(OUT / 'model_comparison.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    fig.savefig(OUT / 'model_comparison.png', dpi=600, bbox_inches='tight')

info = {'chi2_common': chi2_common, 'parameters': P, 'right_axis_stretch': S,
        'residual_limit': 1.15 * extent,
        'largest_residual': {k: [float(t[np.argmax(np.abs(residuals[k]))]), float(residuals[k][np.argmax(np.abs(residuals[k]))])] for k in MODELS},
        'model_maximum_nm': {k: float(curve_axis[np.argmax(curves[k])]) for k in MODELS}}
(OUT / 'verification.json').write_text(json.dumps(info, indent=1), encoding='utf-8', newline='\n')
print(json.dumps(info, indent=1))

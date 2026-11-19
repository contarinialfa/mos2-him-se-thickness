"""SRIM transmitted fraction and transmitted-flux shares on the measured thickness axis.

Reads the processed SRIM route-(i) campaign tables in
analysis/srim_campaign/route_i_metrics/ (per_run_metrics.csv and
threshold_14p2_pooled_curves.csv) and evaluates, using linear interpolation on
the pooled-curve grid and the stretch u = t/s between the measured EELS
thickness t and the SRIM depth axis u:

* run archive composition: number of runs, seed families, incident and
  transmitted histories, production and reconnaissance thicknesses and overlap
  (SI Sec. SVII);
* depths u at which the transmitted fraction T falls to 90% and 10%, and T at
  selected measured thicknesses t for s = 0.70, 0.6246, and 0.62 (main text,
  Fig. 3 discussion; SI Sec. SVII);
* shares of transmitted ions and stopping-weighted flux sqrt(E_exit/E0) exiting
  beyond the 14.2 deg aperture cut (local-impact-capable population) at
  t = 40, 55, 42, and 56 nm, along with the grid peak of Q_{>14.2} (SI Sec. SVII);
* lateral flake-to-tape distance d = 15 mm * tan(4-7 deg) (SI Sec. SII).

Output (default): revision/outputs/srim_transmitted_fraction/stdout.txt.

Usage (from the repository root):
    python revision/scripts/srim_transmitted_fraction.py
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'analysis' / 'srim_campaign' / 'route_i_metrics'
DEFAULT_OUT = ROOT / 'revision' / 'outputs' / 'srim_transmitted_fraction'
OUTPUT_NAME = 'stdout.txt'

_lines = []


def emit(*args) -> None:
    """Print to stdout and record the line for the output file."""
    line = ' '.join(str(a) for a in args)
    print(line)
    _lines.append(line)


def interp(x, y, x0):
    return float(np.interp(x0, x, y))


def inv(x, y, y0):
    # first crossing of decreasing y through y0
    for i in range(len(x) - 1):
        if (y[i] - y0) * (y[i + 1] - y0) <= 0 and y[i] != y[i + 1]:
            return x[i] + (y0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='output folder (default: revision/outputs/srim_transmitted_fraction)')
    args = parser.parse_args()
    out_file = args.out / OUTPUT_NAME

    runs_csv = P / 'per_run_metrics.csv'
    curves_csv = P / 'threshold_14p2_pooled_curves.csv'
    for path in (runs_csv, curves_csv):
        emit('input:', path.relative_to(ROOT).as_posix())

    # Run archive composition (SI Sec. SVII).
    runs = list(csv.DictReader(runs_csv.open(encoding='utf-8')))
    emit('runs:', len(runs))
    fam = {}
    for r in runs:
        fam.setdefault((r['family'], r['seed']), []).append(r)
    for k, v in fam.items():
        us = sorted(float(r['u_nm']) for r in v)
        n = {int(r['n_incident']) for r in v}
        emit(k, 'n runs', len(v), 'u range', us[0], us[-1], 'n_incident', n, 'distinct u', len(set(us)))
        emit('    u:', us)
    allu = sorted({float(r['u_nm']) for r in runs})
    emit('unique thicknesses:', len(allu), allu[0], allu[-1])
    emit('total incident:', sum(int(r['n_incident']) for r in runs),
         'total transmitted:', sum(int(r['n_transmitted']) for r in runs))
    prod_u = sorted({float(r['u_nm']) for r in runs if int(r['n_incident']) == 100000})
    rec_u = sorted({float(r['u_nm']) for r in runs if int(r['n_incident']) == 10000})
    emit('production u:', len(prod_u), prod_u)
    emit('recon u:', len(rec_u), rec_u)
    emit('recon u overlapping production:', len([u for u in rec_u if u in prod_u]),
         '; outside:', [u for u in rec_u if u not in prod_u])

    # Pooled transmitted-flux curves at the 14.2 degree cut.
    cur = list(csv.DictReader(curves_csv.open(encoding='utf-8')))
    U = np.array([float(r['u_nm']) for r in cur])
    T = np.array([float(r['transmission']) for r in cur])
    Fg = np.array([float(r['flux_gt14p2']) for r in cur])
    Fl = np.array([float(r['flux_le14p2']) for r in cur])  # read for completeness; not used below
    Qg = np.array([float(r['sqrtE_gt14p2']) for r in cur])
    Qa = np.array([float(r['sqrtE_all']) for r in cur])
    src = [r['source'] for r in cur]
    emit('\ncurve grid:', list(zip(U.tolist(), src)))

    # Transmitted fraction (main text, Fig. 3 discussion; SI Sec. SVII thickness-axis convention).
    emit('\nTransmission: u at T=0.90: %.1f ; u at T=0.10: %.1f' % (inv(U, T, 0.90), inv(U, T, 0.10)))
    emit('Transmitted fraction T at u = t/s (s = 0.70 is the reference stretch; '
         's = 0.6246 and 0.62 lie at the lower end of the fitted range 0.62-0.73):')
    for s in (0.70, 0.6246, 0.62):
        emit('  s=%.4f' % s)
        for t in (186.0, 185.7, 191.2, 200.0, 212.0):
            u = t / s
            emit('    t=%.1f -> u=%.1f : T=%.4f' % (t, u, interp(U, T, u)))
    emit('  T at u=100: %.4f ; u=255: %.4f ; u=287: %.4f ; u=300: %.4f'
         % (interp(U, T, 100), interp(U, T, 255), interp(U, T, 287), interp(U, T, 300)))

    # Local-impact-capable shares (SI Sec. SVII chamber-geometry classification).
    emit('\nLocal-impact-capable shares (s=0.70):')
    for t in (40.0, 55.0, 42.0, 56.0):
        u = t / 0.70
        Tt = interp(U, T, u); fg = interp(U, Fg, u); qg = interp(U, Qg, u); qa = interp(U, Qa, u)
        emit('  t=%.0f (u=%.1f): T=%.3f ; ions >14.2deg share of transmitted = %.3f ; weighted share = %.3f ; '
             'transmitted per 2e5 incident = %.0f' % (t, u, Tt, fg / Tt, qg / qa, Tt * 2e5))
    emit('same with s=0.62:')
    for t in (40.0, 55.0):
        u = t / 0.62
        Tt = interp(U, T, u); fg = interp(U, Fg, u); qg = interp(U, Qg, u); qa = interp(U, Qa, u)
        emit('  t=%.0f (u=%.1f): ions share %.3f ; weighted share %.3f' % (t, u, fg / Tt, qg / qa))
    i = int(np.argmax(Qg))
    neighbours = [(float(U[j]), round(float(Qg[j]), 4)) for j in range(max(0, i - 3), min(len(U), i + 4))]
    emit('\nQ_{>14.2} grid maximum at u = %.1f (value %.4f); neighbours: %s' % (U[i], Qg[i], neighbours))
    emit('grid band 52-56 nm on measured axis corresponds to u = %.1f - %.1f (s=0.70)' % (52 / 0.70, 56 / 0.70))
    emit('u=80.3 -> t at s=0.70: %.2f ; at s=0.6977: %.2f ; at s=0.6246: %.2f'
         % (80.3 * 0.70, 80.3 * 0.6977, 80.3 * 0.6246))
    emit('tan(14.2 deg) = %.4f' % math.tan(math.radians(14.2)))

    # Lateral flake-to-tape distance for minimum exit angles 4-7 deg (SI Sec. SII).
    emit('d = 15 tan(4,7 deg) = %.3f, %.3f ; mid = %.3f ; 15*tan(5.5deg) = %.3f'
         % (15 * math.tan(math.radians(4)), 15 * math.tan(math.radians(7)),
            (15 * math.tan(math.radians(4)) + 15 * math.tan(math.radians(7))) / 2,
            15 * math.tan(math.radians(5.5))))

    args.out.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(_lines) + '\n')
    try:
        shown = out_file.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = out_file.name
    print(f'wrote {shown}')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()

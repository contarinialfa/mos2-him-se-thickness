"""Evaluate the recoil-ionization share in the semi-infinite SRIM ionization profile.

The SE-escape models use the "ionization by ions" column of the SRIM IONIZ.txt table
from the semi-infinite (300 nm) 30 keV He -> MoS2 run, omitting the "ionization by
recoils" column. This script computes recoil / (ion + recoil):

* integrated over 0-60 nm and over 0-300 nm (piecewise-linear trapezoidal integral
  on the SRIM depth axis, with the first tabulated value extended to zero depth);
* point by point over the tabulated depths 200-300 nm (minimum and maximum).

See SI Sec. SVIII (deposition profile).

Input (default): data/SRIM 30 keV - Monolayer Long/IONIZ.txt.
Output (default): revision/outputs/recoil_ionization_share/recoil_share.json.

Usage (from the repository root):
    python revision/scripts/recoil_ionization_share.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'data' / 'SRIM 30 keV - Monolayer Long' / 'IONIZ.txt'
DEFAULT_OUT = ROOT / 'revision' / 'outputs' / 'recoil_ionization_share'
OUTPUT_NAME = 'recoil_share.json'

# np.trapezoid (NumPy >= 2.0) and np.trapz (older NumPy) are identical.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def display_path(path: Path) -> str:
    """Repository-relative POSIX path if inside the repository, else the file name."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='SRIM IONIZ.txt table (default: the semi-infinite run in data/)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='output folder (default: revision/outputs/recoil_ionization_share)')
    args = parser.parse_args()

    out_file = args.out / OUTPUT_NAME

    rows = []
    for line in args.input.read_text(encoding='latin-1').splitlines():
        tokens = line.split()
        if len(tokens) == 3:
            try:
                rows.append([float(v) for v in tokens])
            except ValueError:
                pass
    data = np.asarray(rows)
    depth = np.r_[0.0, data[:, 0] / 10.0]
    ions = np.r_[data[0, 1], data[:, 1]]
    recoils = np.r_[data[0, 2], data[:, 2]]
    summaries = []
    for endpoint in [60.0, 300.0]:
        axis = np.r_[depth[depth < endpoint], endpoint]
        ion_integral = _trapezoid(np.interp(axis, depth, ions), axis)
        recoil_integral = _trapezoid(np.interp(axis, depth, recoils), axis)
        summaries.append({'depth_nm': [0, endpoint],
                          'recoil_fraction_of_total': float(recoil_integral / (ion_integral + recoil_integral)),
                          'recoil_fraction_of_ions': float(recoil_integral / ion_integral)})
    tail = (data[:, 0] / 10 >= 200) & (data[:, 0] / 10 <= 300.01)
    fractions = data[tail, 2] / (data[tail, 1] + data[tail, 2])
    result = {
        'input': display_path(args.input),
        'rows': len(rows),
        'method': 'piecewise-linear trapezoidal integral on SRIM depth axis; constant extension to zero; recoil/(ion+recoil)',
        'integrals': summaries,
        'local_200_to_300nm_fraction_min': float(fractions.min()),
        'local_200_to_300nm_fraction_max': float(fractions.max()),
        'scope': 'Share of the SRIM recoil-ionization column omitted from the SE-escape models (SI Sec. SVIII).',
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

    for item in summaries:
        lo, hi = item['depth_nm']
        print(f"recoil/(ion+recoil) over {lo}-{hi:g} nm: {item['recoil_fraction_of_total']:.6f} "
              f"(recoil/ion {item['recoil_fraction_of_ions']:.6f})")
    print(f"local recoil/(ion+recoil) at 200-300 nm: {result['local_200_to_300nm_fraction_min']:.6f}"
          f" to {result['local_200_to_300nm_fraction_max']:.6f}")
    print(f"input {result['input']} ({result['rows']} rows)")
    print(f"wrote {display_path(out_file)}")


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()

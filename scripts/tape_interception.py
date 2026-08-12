"""Geometric tape-interception fraction from SRIM transmitted-ion records.

Geometry (see the supplementary material, Sec. SII):
- The ion exits the membrane at the origin, travelling downward (SRIM X = depth axis).
- The converter tape is a vertical strip attached to the underside of the stage,
  its plane at horizontal distance d from the flake (toward +Y), width w
  (transverse, Z), extending L = 15 mm below the stage plane.
- The recorded minimum exit angles phi_min = 4-7 deg imply
  d = L * tan(phi_min) = 1.05-1.84 mm.
- Hit condition for a transmitted-ion direction (cx = down, cy, cz), cy > 0:
    the ray reaches the tape plane at scale t = d / cy
    transverse position Z = d * cz / cy   -> |Z| <= w/2
    depth below stage    X = d * cx / cy  -> X <= L

Usage:
    python scripts/tape_interception.py <path-to-SRIM-campaign-root>

where the campaign root contains the production run directories
(production_100k_seed*/NNN_*_uXXXnm_*/artifacts/SRIM Outputs/TRANSMIT.txt).
The full 1.2 GB campaign archive is not tracked in this repository; see the
README for availability.
"""
import glob
import math
import os
import re
import sys

import numpy as np

E0 = 30000.0
L = 15.0                      # mm, tape length below the stage plane
D_LIST = [1.05, 1.44, 1.84]   # mm  (phi_min = 4, 5.5, 7 deg)
W_LIST = [6.35, 12.7]         # mm  (small tape, large tape)


def parse_transmit(path):
    cx, cy, cz, energy = [], [], [], []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("T"):
                continue
            tokens = line.split()
            try:
                e = float(tokens[3])
                c1, c2, c3 = (float(tokens[-3]), float(tokens[-2]), float(tokens[-1]))
            except (ValueError, IndexError):
                continue
            energy.append(e)
            cx.append(c1)
            cy.append(c2)
            cz.append(c3)
    return (np.array(cx), np.array(cy), np.array(cz), np.array(energy))


def intercept_frac(cx, cy, cz, d, w, weights=None):
    ok = cy > 1e-12
    z_at = np.where(ok, d * cz / np.where(ok, cy, 1.0), np.inf)
    x_at = np.where(ok, d * cx / np.where(ok, cy, 1.0), np.inf)
    hit = ok & (np.abs(z_at) <= w / 2.0) & (x_at <= L)
    if weights is None:
        return hit.mean()
    return (weights * hit).sum() / weights.sum()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = sys.argv[1]

    runs = {}
    pattern = os.path.join(base, "production_100k_seed*", "*", "artifacts", "SRIM Outputs", "TRANSMIT.txt")
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"u(\d+)nm", path)
        if not match:
            continue
        u = int(match.group(1))
        arrays = parse_transmit(path)
        if u not in runs:
            runs[u] = list(arrays)
        else:
            runs[u] = [np.concatenate([a, b]) for a, b in zip(runs[u], arrays)]

    header = f"{'u(nm)':>6}{'t(nm)':>7}{'Ntr':>8}{'th>14.2%':>9}"
    for w in W_LIST:
        for d in D_LIST:
            header += f"  w{w:g}/d{d:g}"
    header += "   [sqrtE-weighted, w=6.35, d=1.44]"
    print(header)

    for u in sorted(runs):
        cx, cy, cz, energy = runs[u]
        n = len(cx)
        if n == 0:
            continue
        theta = np.degrees(np.arccos(np.clip(cx, -1, 1)))
        weights = np.sqrt(np.clip(energy, 0, None) / E0)
        row = f"{u:>6}{0.70 * u:>7.1f}{n:>8}{100 * (theta > 14.2).mean():>8.1f}%"
        for w in W_LIST:
            for d in D_LIST:
                row += f"{100 * intercept_frac(cx, cy, cz, d, w):>9.1f}"
        row += f"{100 * intercept_frac(cx, cy, cz, 1.44, 6.35, weights=weights):>12.1f}"
        print(row)

    # Azimuthal-symmetry check: a tape toward +Z must give the same statistics.
    u_test = sorted(runs)[0]
    cx, cy, cz, _ = runs[u_test]
    f_y = intercept_frac(cx, cy, cz, 1.44, 6.35)
    f_z = intercept_frac(cx, cz, cy, 1.44, 6.35)
    print(f"\nazimuthal-symmetry check @u{u_test}: +Y {100 * f_y:.2f}%  vs +Z {100 * f_z:.2f}%")


if __name__ == "__main__":
    main()

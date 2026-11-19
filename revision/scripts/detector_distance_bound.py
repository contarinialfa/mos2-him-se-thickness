"""Upper bound on the sample-to-detector distance variation across the Fig. 3 flakes.

Evaluates the change in sample-to-detector distance resulting from stage translations between
the three Fig. 3 flakes for an arbitrary detector azimuth (SI Sec. SI, "Imaging positions").

In the laboratory frame, the sample translates by an in-plane step h on the grid and a vertical
step dWD. For a detector at elevation e = 16 deg and azimuth phi, the first-order distance
change is the projection onto the detector direction:
    |dD| = |h cos(e) cos(phi - psi) + dWD sin(e)|,
with maximum h cos(e) + |dWD| sin(e) over phi, and vertical-only component |dWD| sin(e).

Input: revision/outputs/grid_positions/pairs_fig3.csv.
Constants (main text, Methods): detector distance 9 mm, elevation 16 deg.

Usage (from the repository root):
    python -B revision/scripts/detector_distance_bound.py [--out DIR]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "outputs" / "grid_positions" / "pairs_fig3.csv"
OUT_DEFAULT = ROOT / "revision" / "outputs" / "detector_distance_bound"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
out = ap.parse_args().out

D0, ELEV = 9000.0, math.radians(16.0)
with open(SRC, encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

lines = [
    "Change in sample-to-detector distance between the Fig. 3 flakes, any detector azimuth",
    "input: %s" % SRC.relative_to(ROOT).as_posix(),
    "constants: detector distance 9 mm, elevation 16 deg (main text, Methods)",
    "",
    "pair | in-plane h (um) | dWD (um) | 3-D travel (um) | vertical-only dD (um) | max dD any azimuth (um, % of 9 mm)",
]
worst = 0.0
travel_of_worst = 0.0
for r in rows:
    h, dwd, trav = float(r["inplane_on_grid_um"]), float(r["dWD_um"]), float(r["stage_travel_3d_um"])
    vert = abs(dwd) * math.sin(ELEV)
    mx = h * math.cos(ELEV) + abs(dwd) * math.sin(ELEV)
    if mx > trav:
        raise SystemExit("bound exceeds the 3-D stage travel of pair %s -> %s" % (r["a"], r["b"]))
    if mx > worst:
        worst, travel_of_worst = mx, trav
    lines.append("%s -> %s | %.1f | %.1f | %.1f | %.1f | %.1f (%.2f%%)" % (r["a"], r["b"], h, dwd, trav, vert, mx, 100 * mx / D0))
lines += [
    "",
    "largest change for any azimuth: %.0f um = %.1f%% of 9 mm (upper bound; the vertical-only figure is the lower end)" % (worst, 100 * worst / D0),
    "check: never larger than the 3-D stage travel (%.0f um)." % travel_of_worst,
]
out.mkdir(parents=True, exist_ok=True)
with open(out / "summary.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

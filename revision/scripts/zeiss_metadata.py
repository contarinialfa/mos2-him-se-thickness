"""Read the acquisition metadata of a Zeiss ORION helium-ion-microscope TIFF.

The ORION software stores the instrument state of every frame as XML in TIFF tag 65000.
read_fields() returns it as {field: (value, units)}; nested groups are joined with dots
(for example "GFIS.WorkingDistance", "Detector.Scintillator"). Values and units are the
strings recorded in the file. Note that the beam-shift and beam-tilt fields (McXShift,
McYShift, McXTilt, McYTilt) carry the label "Micro Radians" although their numerical
values are recorded in radians.

Used by imaging_settings_and_positions.py and grid_positions.py to extract acquisition
parameters and stage coordinates for the main text (Methods) and SI Sec. SI.

Usage (from the repository root):
    python -B revision/scripts/zeiss_metadata.py data/raw_images/transmission/flake_1_loc4.tif
"""
import sys
import xml.etree.ElementTree as ET

from PIL import Image


def read_fields(path):
    xml = Image.open(path).tag_v2.get(65000)
    if isinstance(xml, bytes):
        xml = xml.decode("utf-8", "replace")
    root = ET.fromstring(xml)
    out = {}

    def walk(node, prefix):
        val = node.find("Value")
        if val is not None:
            unit = node.find("Units")
            out[prefix] = ((val.text or "").strip(), (unit.text or "").strip() if unit is not None else "")
            return
        for child in node:
            walk(child, f"{prefix}.{child.tag}" if prefix else child.tag)

    walk(root, "")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B revision/scripts/zeiss_metadata.py <frame.tif>")
    for k, (v, u) in read_fields(sys.argv[1]).items():
        print(f"{k:55} {v} {u}")

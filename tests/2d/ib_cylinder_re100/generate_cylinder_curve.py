#!/usr/bin/env python3
"""Generate the curve used by inputs.ib_cylinder_re100."""

import math
from pathlib import Path


XC = 2.0
YC = 0.0
DIAMETER = 1.0
RADIUS = 0.5 * DIAMETER

# With finest dx = D / 20, this gives ds ~= 2.6 dx. The current
# unpreconditioned coupled IB solve is sensitive to over-dense markers.
N_SEGMENTS = 24


def main():
    points = []
    for i in range(N_SEGMENTS):
        theta = 2.0 * math.pi * i / N_SEGMENTS
        points.append((XC + RADIUS * math.cos(theta), YC + RADIUS * math.sin(theta)))

    out = Path(__file__).with_name("cylinder.curve")
    with out.open("w", encoding="ascii") as fh:
        fh.write(f"{N_SEGMENTS} {N_SEGMENTS}\n")
        for x, y in points:
            fh.write(f"{x:.17g} {y:.17g}\n")
        for i in range(N_SEGMENTS):
            fh.write(f"{i} {(i + 1) % N_SEGMENTS}\n")

    ds = 2.0 * RADIUS * math.sin(math.pi / N_SEGMENTS)
    print(f"wrote {out}")
    print(f"segments: {N_SEGMENTS}")
    print(f"segment chord: {ds:.12g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the STL used by inputs.ib_cylinder_channel."""

import math
from pathlib import Path


XC = 1.0
YC = 0.5
RADIUS = 0.125
Z_LO = 0.0
Z_HI = 0.25

# The channel grid in inputs.ib_cylinder_channel has dx = 1/16.
# These counts give circumferential and axial panel sizes close to
# 1.5 * dx, which is intentionally coarser than the fluid mesh.
N_THETA = 8
N_Z = 3


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag == 0.0:
        return 0.0, 0.0, 0.0
    return nx / mag, ny / mag, nz / mag


def main():
    rings = []
    for k in range(N_Z + 1):
        z = Z_LO + (Z_HI - Z_LO) * k / N_Z
        ring = []
        for i in range(N_THETA):
            theta = 2.0 * math.pi * i / N_THETA
            ring.append(
                (
                    XC + RADIUS * math.cos(theta),
                    YC + RADIUS * math.sin(theta),
                    z,
                )
            )
        rings.append(ring)

    triangles = []
    for k in range(N_Z):
        for i in range(N_THETA):
            j = (i + 1) % N_THETA
            a = rings[k][i]
            b = rings[k][j]
            c = rings[k + 1][j]
            d = rings[k + 1][i]
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    out = Path(__file__).with_name("cylinder.stl")
    with out.open("w", encoding="ascii") as fh:
        fh.write("solid cylinder_channel_r125_8x3\n")
        for tri in triangles:
            nx, ny, nz = normal(*tri)
            fh.write(f"  facet normal {nx:.17g} {ny:.17g} {nz:.17g}\n")
            fh.write("    outer loop\n")
            for x, y, z in tri:
                fh.write(f"      vertex {x:.17g} {y:.17g} {z:.17g}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write("endsolid cylinder_channel_r125_8x3\n")

    chord = 2.0 * RADIUS * math.sin(math.pi / N_THETA)
    dz = (Z_HI - Z_LO) / N_Z
    print(f"wrote {out}")
    print(f"triangles: {len(triangles)}")
    print(f"circumferential chord: {chord:.12g}")
    print(f"axial panel size:      {dz:.12g}")


if __name__ == "__main__":
    main()

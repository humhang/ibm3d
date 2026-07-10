#!/usr/bin/env python3
"""Generate the STL used by inputs.ib_sphere_re100."""

import math
from pathlib import Path


XC = 0.0
YC = 0.0
ZC = 0.0
DIAMETER = 1.0
RADIUS = 0.5 * DIAMETER

# With finest dx = D / 80, equatorial and meridional panel lengths are
# about 5.2 finest cells. Coupled BiCGStab still lacks an in-iteration IB
# block preconditioner and is sensitive to over-dense markers.
N_THETA = 48
N_PHI = 24


def point(theta, phi):
    sin_phi = math.sin(phi)
    return (
        XC + RADIUS * sin_phi * math.cos(theta),
        YC + RADIUS * sin_phi * math.sin(theta),
        ZC + RADIUS * math.cos(phi),
    )


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


def orient_outward(a, b, c):
    nx, ny, nz = normal(a, b, c)
    cx = (a[0] + b[0] + c[0]) / 3.0 - XC
    cy = (a[1] + b[1] + c[1]) / 3.0 - YC
    cz = (a[2] + b[2] + c[2]) / 3.0 - ZC
    if nx * cx + ny * cy + nz * cz < 0.0:
        return a, c, b
    return a, b, c


def main():
    north = (XC, YC, ZC + RADIUS)
    south = (XC, YC, ZC - RADIUS)
    rings = []
    for k in range(1, N_PHI):
        phi = math.pi * k / N_PHI
        rings.append([point(2.0 * math.pi * i / N_THETA, phi) for i in range(N_THETA)])

    triangles = []
    first = rings[0]
    for i in range(N_THETA):
        triangles.append(orient_outward(north, first[(i + 1) % N_THETA], first[i]))

    for k in range(len(rings) - 1):
        lo = rings[k]
        hi = rings[k + 1]
        for i in range(N_THETA):
            j = (i + 1) % N_THETA
            triangles.append(orient_outward(lo[i], lo[j], hi[j]))
            triangles.append(orient_outward(lo[i], hi[j], hi[i]))

    last = rings[-1]
    for i in range(N_THETA):
        triangles.append(orient_outward(last[i], last[(i + 1) % N_THETA], south))

    out = Path(__file__).with_name("sphere.stl")
    with out.open("w", encoding="ascii") as fh:
        fh.write("solid sphere_re100_d1_48x24\n")
        for tri in triangles:
            nx, ny, nz = normal(*tri)
            fh.write(f"  facet normal {nx:.17g} {ny:.17g} {nz:.17g}\n")
            fh.write("    outer loop\n")
            for x, y, z in tri:
                fh.write(f"      vertex {x:.17g} {y:.17g} {z:.17g}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write("endsolid sphere_re100_d1_48x24\n")

    equator_chord = 2.0 * RADIUS * math.sin(math.pi / N_THETA)
    meridional = math.pi * RADIUS / N_PHI
    print(f"wrote {out}")
    print(f"triangles: {len(triangles)}")
    print(f"equator chord:    {equator_chord:.12g}")
    print(f"meridional panel: {meridional:.12g}")


if __name__ == "__main__":
    main()

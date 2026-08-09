#!/usr/bin/env python3
"""
Command-line entry point.

    python -m dircurv volume.nii.gz --mask brain.nii.gz --sigma 2e-7 --out maps/

Runs the reliability analysis on a measured 2D or 3D field and writes the maps
to disk. Dimensionality is taken from the array.

For complex data, run each part separately -- the expansion is linear in the
field, so it applies component-wise, while the modulus is nonlinear and outside
the theory:

    python -m dircurv u.mat --component real --out maps_real/
    python -m dircurv u.mat --component imag --out maps_imag/

`sigma` is the noise standard deviation in the SAME UNITS as the field values,
not a relative figure. If you know only a relative level, multiply it by the
field amplitude yourself; `--sigma-relative` does that for you.

Start with `--coverage-only`. Interpolation needs a complete local neighbourhood,
so it reports what fraction of the masked region is measurable at all before any
time is spent on the full run.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m dircurv",
        description="Directional curvature measurement with reliability "
                    "reporting, on a measured 2D or 3D field.")
    p.add_argument("field", help=".nii, .nii.gz, .mat, .npy or .npz")
    p.add_argument("--mask", help="validity mask, same shape as the field")
    p.add_argument("--out", default="dircurv_maps", help="output directory")
    p.add_argument("--spacing", type=float, nargs="+",
                   help="voxel spacing, in the axis order of the loaded array; "
                        "one value means isotropic. Required unless the file has "
                        "a header. No unit conversion is applied.")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="noise standard deviation in the field's own units")
    p.add_argument("--sigma-relative", type=float,
                   help="noise as a fraction of the field amplitude; converted "
                        "for you and overrides --sigma")
    p.add_argument("--component", choices=["real", "imag", "abs"],
                   help="which part of a complex field to use; 'abs' is "
                        "nonlinear and outside the theory")
    p.add_argument("--m", type=int, default=None,
                   help="number of probe directions (default 16 in 2D, 24 in 3D; "
                        "C3 needs more than 10 in 2D and more than 20 in 3D)")
    p.add_argument("--step", type=int, default=4,
                   help="subsample stride; 1 measures every voxel and is slow")
    p.add_argument("--key", help="variable name inside a .mat or .npz")
    p.add_argument("--axis-order", choices=["reverse", "keep"], default="reverse",
                   help="NIfTI and MATLAB store [x,y,z]; the modules want "
                        "[z,y,x], so the default reverses the axes")
    p.add_argument("--coverage-only", action="store_true",
                   help="report the measurable fraction and stop")
    p.add_argument("--h-cap", type=float,
                   help="largest span to consider, in spacing units")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .io import load_field, load_mask, save_maps, describe

    field, spacing = load_field(args.field, spacing=args.spacing, key=args.key,
                                axis_order=args.axis_order,
                                component=args.component)
    mask = (load_mask(args.mask, field.shape, key=args.key,
                      axis_order=args.axis_order) if args.mask else None)

    sigma = args.sigma
    if args.sigma_relative is not None:
        amp = float(np.abs(field).max())
        sigma = args.sigma_relative * amp
        print(f"sigma set from --sigma-relative: {args.sigma_relative:g} x "
              f"{amp:g} = {sigma:g} (field units)")

    ndim = field.ndim
    if ndim == 3:
        from .grid3d import VolumeField, reliability_volumes, coverage_fraction
        from .grid3d import VERDICT_CODE
        fld = VolumeField(field, spacing=spacing, mask=mask)
        cov = coverage_fraction(fld, step=max(args.step, 2))
        print(f"coverage: {100*cov:.1f}% of masked voxels have a complete "
              "interpolation neighbourhood")
        if cov < 0.05:
            print("  this is very low. The interpolation stencil needs a locally "
                  "complete neighbourhood, so a mask that excludes scattered "
                  "individual voxels leaves little measurable. A contiguous "
                  "anatomical mask works directly.")
        if args.coverage_only:
            return 0
        m = args.m if args.m else 24
        print(f"measuring with m={m} directions, step={args.step} ...")
        maps = reliability_volumes(fld, m=m, h_cap=args.h_cap, sigma=sigma,
                                   step=args.step, progress=True)
        codes = VERDICT_CODE
    elif ndim == 2:
        from .grid2d import GridField, reliability_maps, VERDICT_CODE
        fld = GridField(field, spacing=spacing, mask=mask)
        if args.coverage_only:
            ok = tot = 0
            for i in range(fld.ny):
                for j in range(fld.nx):
                    if fld.mask[i, j]:
                        tot += 1
                        ok += fld.valid(np.array([i, j], float))
            print(f"coverage: {100*ok/max(tot,1):.1f}% of masked pixels")
            return 0
        m = args.m if args.m else 16
        print(f"measuring with m={m} directions, step={args.step} ...")
        maps = reliability_maps(fld, m=m, h_cap=args.h_cap, sigma=sigma,
                                step=args.step, progress=True)
        codes = VERDICT_CODE
    else:
        print(f"error: expected a 2D or 3D array, got {ndim}D", file=sys.stderr)
        return 2

    print()
    describe(maps, verdict_codes=codes)
    print()
    save_maps(maps, args.out, spacing=spacing, axis_order=args.axis_order,
              template=args.field if args.field.lower().endswith(
                  (".nii", ".nii.gz")) else None)
    print()
    print("Reading the output: 'verdict' is the summary map -- "
          + ", ".join(f"{v}={k}" for k, v in sorted(codes.items(),
                                                    key=lambda t: t[1]))
          + ". 'kappa' is the conditioning of the direction geometry available "
            "at each voxel, 'rho' the predicted error relative to the recovered "
            "curvature, and H11..H33 the Hessian components.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

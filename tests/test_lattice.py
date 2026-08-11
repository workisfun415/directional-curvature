"""Lattice mode: interpolation-free measurement on grid nodes.

This path exists because the interpolating path, measured on a masked disc and a
masked sphere, could be applied at NO pixel or voxel where central differences
could not -- its 4x4 and 4x4x4 support made it strictly more demanding than a
3x3 or 3x3x3 stencil, cancelling the one-sided advantage the theory predicts.
Restricting probe points to grid nodes removes the interpolation support
entirely.

Two silent index-order bugs occurred while this module was written, so the
convention tests below use ANISOTROPIC spacing and distinct diagonal Hessian
entries: any axis permutation or spacing mix-up changes the answer and fails.
"""
import numpy as np
import pytest

from dircurv.lattice import (measure_lattice, feasible_lattice,
                             lattice_directions, max_steps)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _implB_lattice import measure_B


# ------------------------------------------------------------------ fields
def quad2(n=41, spacing=(0.005, 0.002),
          H=np.array([[3.0, -1.0], [-1.0, 7.0]])):
    yg, xg = np.mgrid[0:n, 0:n]
    Y, X = (yg - n//2)*spacing[0], (xg - n//2)*spacing[1]
    F = (0.5*(H[0, 0]*X**2 + 2*H[0, 1]*X*Y + H[1, 1]*Y**2)
         + 0.7*X - 0.4*Y + 1.3)
    return F, H, spacing


def quad3(n=25, spacing=(0.005, 0.003, 0.002),
          H=np.array([[3.0, -1.0, 0.5], [-1.0, 7.0, 0.2], [0.5, 0.2, 11.0]])):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    Z, Y, X = (zz - n//2)*spacing[0], (yy - n//2)*spacing[1], (xx - n//2)*spacing[2]
    F = 0.5*(H[0, 0]*X**2 + H[1, 1]*Y**2 + H[2, 2]*Z**2
             + 2*H[0, 1]*X*Y + 2*H[0, 2]*X*Z + 2*H[1, 2]*Y*Z)
    return F, H, spacing


# ---------------------------------------------------------------- exactness
def test_quadratic_exact_2d_anisotropic():
    F, H, sp = quad2()
    r = measure_lattice(F, (20, 20), spacing=sp, order=2, s_cap=1)
    assert np.abs(r.H_hat - H).max() < 1e-10


def test_quadratic_exact_3d_anisotropic():
    F, H, sp = quad3()
    r = measure_lattice(F, (12, 12, 12), spacing=sp, order=1, s_cap=1)
    assert np.abs(r.H_hat - H).max() < 1e-10


def test_axis_convention_is_physical_not_index():
    """H11 belongs to x, the FASTEST array axis. Distinct diagonal entries and
    anisotropic spacing mean a swap cannot pass."""
    F, H, sp = quad2()
    r = measure_lattice(F, (20, 20), spacing=sp, order=2, s_cap=1)
    assert abs(r.H_hat[0, 0] - 3.0) < 1e-9
    assert abs(r.H_hat[1, 1] - 7.0) < 1e-9
    F3, H3, sp3 = quad3()
    r3 = measure_lattice(F3, (12, 12, 12), spacing=sp3, order=1, s_cap=1)
    assert np.abs(np.diag(r3.H_hat) - np.array([3.0, 7.0, 11.0])).max() < 1e-9


# ------------------------------------------------- independent implementation
@pytest.mark.parametrize("dim", [2, 3])
def test_agrees_with_independent_implementation(dim):
    if dim == 2:
        F, H, sp = quad2(); idx, order = (20, 20), 2
    else:
        F, H, sp = quad3(); idx, order = (12, 12, 12), 1
    rA = measure_lattice(F, idx, spacing=sp, order=order, s_cap=1)
    HB, kB, _ = measure_B(F, idx, spacing=sp, order=order, s=1)
    assert np.abs(rA.H_hat - HB).max() < 1e-10
    assert abs(rA.kappa - kB) < 1e-8


# -------------------------------------------------------------- feasibility
def _central_ok(mask, idx):
    off = [-1, 0, 1]
    if mask.ndim == 2:
        i, j = idx
        return all(mask[i+a, j+b] for a in off for b in off)
    i, j, k = idx
    return all(mask[i+a, j+b, k+c] for a in off for b in off for c in off)


def test_reaches_voxels_central_differences_cannot():
    """The point of this module. Lattice mode must be applicable strictly more
    widely than a central-difference stencil, not less."""
    n, sp = 33, 0.002
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    X, Y, Z = (xx-n//2)*sp, (yy-n//2)*sp, (zz-n//2)*sp
    mask = np.sqrt(X**2 + Y**2 + Z**2) < 0.026
    only_lat = only_cent = 0
    for i in range(1, n-1):
        for j in range(1, n-1):
            for k in range(1, n-1):
                if not mask[i, j, k]:
                    continue
                lat = len(feasible_lattice(mask, (i, j, k), order=1, s_cap=2)) >= 6
                cen = _central_ok(mask, (i, j, k))
                only_lat += (lat and not cen)
                only_cent += (cen and not lat)
    assert only_lat > 100, only_lat
    assert only_cent == 0, only_cent


def test_outside_mask_is_unusable():
    F, H, sp = quad3()
    mask = np.zeros_like(F, bool)
    mask[10:15, 10:15, 10:15] = True
    r = measure_lattice(F, (0, 0, 0), spacing=sp, mask=mask, order=1)
    assert r.verdict == "UNUSABLE"
    assert r.H_hat is None


def test_too_few_directions_is_unusable():
    """A one-voxel-thick sheet cannot span the space of symmetric matrices."""
    F, H, sp = quad3()
    mask = np.zeros_like(F, bool)
    mask[12, :, :] = True
    r = measure_lattice(F, (12, 12, 12), spacing=sp, mask=mask, order=1)
    assert r.verdict == "UNUSABLE"


def test_probe_points_never_leave_the_mask():
    """No interpolation means the support is exactly the visited nodes."""
    n = 25
    mask = np.zeros((n, n), bool)
    mask[5:20, 5:20] = True
    for vec, s in feasible_lattice(mask, (6, 6), order=2, s_cap=3):
        v = np.asarray(vec, int)
        for step in (s, 2*s):
            p = np.array([6, 6]) + step*v
            assert mask[tuple(p)], (vec, s, p)


# -------------------------------------------------------------------- order
def test_antipodal_pairs_are_present_in_the_open_interior():
    F, H, sp = quad3()
    r = measure_lattice(F, (12, 12, 12), spacing=sp, order=1, s_cap=1)
    assert r.antipodal_sampled == 1.0
    assert r.expected_order == 2


def test_lattice_sets_are_antipodally_closed():
    for ndim in (2, 3):
        for order in (1, 2):
            D = set(lattice_directions(ndim, order))
            for v in D:
                assert tuple(-np.array(v)) in D, (ndim, order, v)


def test_max_steps_respects_the_mask():
    mask = np.zeros((21, 21), bool)
    mask[10, 10:16] = True
    assert max_steps(mask, (10, 10), (0, 1), 5) == 2      # nodes at 2 and 4
    assert max_steps(mask, (10, 10), (0, -1), 5) == 0     # nothing to the left


# --------------------------------------------------------------- batch + CLI
def test_reliability_lattice_shapes_and_masking():
    from dircurv.lattice import reliability_lattice, VERDICT_CODE
    F, H, sp = quad3(n=21)
    zz, yy, xx = np.mgrid[0:21, 0:21, 0:21]
    mask = (np.sqrt((xx-10)**2 + (yy-10)**2 + (zz-10)**2) < 7)
    maps = reliability_lattice(F, sp, mask=mask, order=1, s_cap=2, step=3)
    for k in ("kappa", "verdict", "order", "H11", "H33"):
        assert maps[k].shape == F.shape
    assert np.all(maps["verdict"][~mask] == 0)
    scored = maps["verdict"] > 0
    assert scored.sum() > 20
    # inside a ball the quadratic must be recovered exactly
    good = scored & np.isfinite(maps["H11"])
    assert np.nanmax(np.abs(maps["H11"][good] - H[0, 0])) < 1e-8
    assert np.nanmax(np.abs(maps["H33"][good] - H[2, 2])) < 1e-8


def test_cli_defaults_to_the_lattice_path(tmp_path, capsys):
    """The interpolating path has no boundary advantage, so it must not be the
    default: on a masked sphere it reached 62.9% of voxels against 100% here."""
    nib = pytest.importorskip("nibabel")
    from dircurv.__main__ import main
    n, sp = 21, 0.002
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    X, Y, Z = (xx-n//2)*sp, (yy-n//2)*sp, (zz-n//2)*sp
    F = np.exp(X + 0.5*Y)*np.cos(2*Z)
    mask = np.sqrt(X**2 + Y**2 + Z**2) < 0.014
    aff = np.diag([sp*1000]*3 + [1.0])
    pv = str(tmp_path / "u.nii.gz")
    pm = str(tmp_path / "m.nii.gz")
    nib.save(nib.Nifti1Image(np.transpose(F, (2, 1, 0)), aff), pv)
    nib.save(nib.Nifti1Image(np.transpose(mask, (2, 1, 0)).astype(float), aff), pm)
    rc = main([pv, "--mask", pm, "--spacing", str(sp), "--coverage-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lattice path" in out and "no interpolation" in out

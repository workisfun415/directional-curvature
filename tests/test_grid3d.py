"""3D gridded interface: exactness, masking and the C3 direction requirement."""
import numpy as np
import pytest
from dircurv.grid3d import (VolumeField, MaskSupportError3D, measure_voxel,
                            reliability_volumes, coverage_fraction,
                            KAPPA_FULL_SPHERE, _poly_basis_3d,
                            _fibonacci_hemisphere, antipodal_sampled_fraction_3d,
                            local_directions_3d)

N, SP = 25, 0.04
zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
X, Y, Z = (xx - N // 2) * SP, (yy - N // 2) * SP, (zz - N // 2) * SP
RAD = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
HT = np.array([[3.0, -1.0, 0.5], [-1.0, 2.0, 0.2], [0.5, 0.2, 1.5]])
FQ = (0.5 * (HT[0, 0] * X ** 2 + HT[1, 1] * Y ** 2 + HT[2, 2] * Z ** 2
             + 2 * HT[0, 1] * X * Y + 2 * HT[0, 2] * X * Z
             + 2 * HT[1, 2] * Y * Z) + 0.7 * X - 0.4 * Y + 0.2 * Z + 1.3)
C = (N // 2, N // 2, N // 2)


def test_tricubic_recovers_a_quadratic_exactly():
    """Tricubic reproduces cubics, so a quadratic must come back exactly.
    A trilinear interpolant would inject error of the same size as the signal."""
    r = measure_voxel(VolumeField(FQ, spacing=SP), C, m=24)
    assert np.abs(r.H_hat - HT).max() < 1e-10


def test_complex_input_is_refused():
    with pytest.raises(TypeError):
        VolumeField(FQ + 1j * FQ, spacing=SP)


def test_kappa_matches_the_icosahedral_reference():
    r = measure_voxel(VolumeField(FQ, spacing=SP), C, m=12)
    assert abs(r.kappa - KAPPA_FULL_SPHERE) < 1e-6


def test_antipodal_pairs_give_second_order():
    v = VolumeField(FQ, spacing=SP)
    dirs, _ = local_directions_3d(v, C, m=24)
    assert antipodal_sampled_fraction_3d(dirs) == 1.0
    assert measure_voxel(v, C, m=24).expected_order == 2


def test_c3_needs_more_than_twenty_directions():
    """The degree-3 basis on the sphere has 20 columns."""
    B, _ = _poly_basis_3d(_fibonacci_hemisphere(30), 3)
    assert B.shape[1] == 20
    v = VolumeField(FQ, spacing=SP)
    assert measure_voxel(v, C, m=12).C3_status == "UNDER-DETERMINED"


def test_no_extrapolation_at_the_mask_edge():
    mk = RAD < 0.24
    v = VolumeField(FQ, spacing=SP, mask=mk)
    edge = [(iz, iy, ix) for iz in range(N) for iy in range(N) for ix in range(N)
            if mk[iz, iy, ix] and not v.valid(np.array([iz, iy, ix], float))]
    assert len(edge) > 10
    for idx in edge[::max(1, len(edge) // 8)]:
        assert measure_voxel(v, idx, m=12).verdict == "UNUSABLE"


def test_strict_sampling_raises_outside_the_mask():
    mk = RAD < 0.24
    v = VolumeField(FQ, spacing=SP, mask=mk)
    edge = next((iz, iy, ix) for iz in range(N) for iy in range(N)
                for ix in range(N)
                if mk[iz, iy, ix] and not v.valid(np.array([iz, iy, ix], float)))
    with pytest.raises(MaskSupportError3D):
        v.at(np.array(edge, float), strict=True)


def test_coverage_falls_with_scattered_dropout():
    """Tricubic needs 64 valid neighbours, so 0.9**64 is about 0.1 percent."""
    full = coverage_fraction(VolumeField(FQ, spacing=SP), step=3)
    drop = np.random.default_rng(1).random(FQ.shape) > 0.10
    scat = coverage_fraction(VolumeField(FQ, spacing=SP, mask=drop), step=3)
    assert full > 0.4
    assert scat < 0.02


def test_volumes_shapes_and_invalid_marking():
    mk = RAD < 0.20
    v = VolumeField(FQ, spacing=SP, mask=mk)
    vols = reliability_volumes(v, m=12, sigma=1e-4, step=6)
    for k in ("solid_angle", "kappa", "verdict", "order", "c3_status"):
        assert vols[k].shape == FQ.shape
    assert np.all(vols["verdict"][~mk] == 0)

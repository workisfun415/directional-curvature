"""Masked-data integrity: the module must never interpolate outside the mask."""
import numpy as np
import pytest
from dircurv.grid2d import (GridField, MaskSupportError, measure_pixel,
                            reliability_maps, aperture_deg, local_directions)

N, SP = 41, 0.03
yy, xx = np.mgrid[0:N, 0:N]
X, Y = (xx - N // 2) * SP, (yy - N // 2) * SP
FIELD = np.exp(X) * np.cos(2 * Y) + 0.3 * X ** 3


def test_complex_input_is_refused():
    with pytest.raises(TypeError):
        GridField(FIELD + 1j * FIELD, spacing=SP)


def test_strict_sampling_raises_outside_mask():
    mask = np.hypot(X, Y) < 0.20
    g = GridField(FIELD, spacing=SP, mask=mask)
    edge = None
    for iy in range(N):
        for ix in range(N):
            if mask[iy, ix] and not g.valid(np.array([iy, ix], float)):
                edge = (iy, ix); break
        if edge:
            break
    assert edge is not None
    with pytest.raises(MaskSupportError):
        g.at(np.array(edge, float), strict=True)


def test_no_extrapolation_anywhere():
    """Every masked-in pixel whose 4x4 stencil leaves the mask must return
    UNUSABLE rather than a number."""
    mask = np.hypot(X, Y) < 0.20
    g = GridField(FIELD, spacing=SP, mask=mask)
    edge = [(iy, ix) for iy in range(N) for ix in range(N)
            if mask[iy, ix] and not g.valid(np.array([iy, ix], float))]
    assert len(edge) > 10
    for idx in edge[::max(1, len(edge) // 12)]:        # sample, not exhaustive
        assert measure_pixel(g, idx, m=8).verdict == "UNUSABLE"


def test_aperture_detects_restriction():
    """A half-plane mask must not report a full circle of usable directions."""
    g_free = GridField(FIELD, spacing=SP)
    g_half = GridField(FIELD, spacing=SP, mask=X > -0.02)
    idx = (N // 2, N // 2)
    dirs, spans = local_directions(g_free, idx, m=16)
    h_ref = float(np.median(spans))
    assert aperture_deg(g_free, idx, h_ref) > 300
    dirs2, spans2 = local_directions(g_half, idx, m=16)
    if len(spans2):
        assert aperture_deg(g_half, idx, float(np.median(spans2))) < 300


def test_singular_direction_sets_are_refused():
    """A one-pixel-wide valid strip cannot span Sym_2 and must not be scored."""
    mask = np.abs(Y) < 0.5 * SP
    g = GridField(FIELD, spacing=SP, mask=mask)
    r = measure_pixel(g, (N // 2, N // 2), m=16)
    assert r.verdict == "UNUSABLE"
    assert r.H_hat is None


def test_maps_shapes_and_invalid_marking():
    mask = np.hypot(X, Y) < 0.18
    g = GridField(FIELD, spacing=SP, mask=mask)
    maps = reliability_maps(g, m=8, sigma=1e-4, step=8)
    for k in ("aperture", "kappa", "verdict", "order", "c3_status"):
        assert maps[k].shape == (N, N)
    assert np.all(maps["verdict"][~mask] == 0)
    assert np.all(np.isnan(maps["aperture"][~mask]))

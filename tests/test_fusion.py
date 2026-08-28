"""Fusion module: an experimental convex combination of two Hessian estimates.

The evidence and its limits are in the module docstring. These tests pin the
arithmetic and the endpoint behaviour, not the scientific claim -- the reported
weight is not universally optimal and no test here should be read as endorsing it.
"""
import numpy as np
import pytest
from dircurv.fusion import fuse_hessians, sweep_weights, W_REPORTED


H1 = np.array([[3.0, -1.0, 0.5], [-1.0, 7.0, 0.2], [0.5, 0.2, 11.0]])
H2 = np.array([[2.0, 0.5, -0.3], [0.5, 5.0, 0.1], [-0.3, 0.1, 9.0]])


def test_endpoints_return_the_inputs():
    assert np.allclose(fuse_hessians(H1, H2, 0.0), H2)
    assert np.allclose(fuse_hessians(H1, H2, 1.0), H1)


def test_midpoint_is_the_average():
    assert np.allclose(fuse_hessians(H1, H2, 0.5), 0.5*(H1 + H2))


def test_result_is_symmetric():
    for w in (0.0, 0.25, 0.65, 1.0):
        F = fuse_hessians(H1, H2, w)
        assert np.allclose(F, F.T)


def test_complex_inputs_are_supported():
    """Real and imaginary parts of a complex field are processed separately and
    recombined, so the fused Hessian may be complex."""
    A = H1 + 1j*H2
    B = H2 + 1j*H1
    F = fuse_hessians(A, B, 0.65)
    assert np.iscomplexobj(F)
    assert np.allclose(F, 0.65*A + 0.35*B)


def test_missing_estimate_returns_the_other():
    assert np.allclose(fuse_hessians(None, H2, 0.65), H2)
    assert np.allclose(fuse_hessians(H1, None, 0.65), H1)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_weight_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        fuse_hessians(H1, H2, bad)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        fuse_hessians(H1, np.eye(2), 0.5)


def test_sweep_covers_both_endpoints():
    out = sweep_weights(H1, H2)
    assert 0.0 in out and 1.0 in out
    assert np.allclose(out[0.0], H2)
    assert np.allclose(out[1.0], H1)


def test_reported_weight_is_documented_as_not_universal():
    """W_REPORTED is a reported value, not a recommended default. The docstring
    must say so, because the per-acquisition optimum ranged from 0.25 to 0.75."""
    from dircurv import fusion
    assert abs(W_REPORTED - 0.65) < 1e-12
    doc = fusion.__doc__
    assert "NOT UNIVERSALLY OPTIMAL" in doc
    assert "not be used clinically" in doc
    assert "60 Hz" in doc

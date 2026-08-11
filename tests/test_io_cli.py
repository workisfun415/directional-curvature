"""File I/O and the command line: the paths a user actually takes."""
import os
import numpy as np
import pytest
from dircurv.io import load_field, load_mask, save_maps

nib = pytest.importorskip("nibabel")

N, SP = 21, 0.002


def _volume():
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    Z, Y, X = (zz - N//2)*SP, (yy - N//2)*SP, (xx - N//2)*SP
    return np.exp(X + 0.5*Y)*np.cos(2*Z), np.sqrt(X**2 + Y**2 + Z**2) < 0.014


def _write_nifti(tmp_path, arr, name):
    # store as [x, y, z], the NIfTI convention
    p = str(tmp_path / name)
    aff = np.diag([SP*1000]*3 + [1.0])
    nib.save(nib.Nifti1Image(np.transpose(arr, (2, 1, 0)).astype(float), aff), p)
    return p


def test_axis_order_round_trips(tmp_path):
    """The default reverses [x,y,z] to [z,y,x]. Getting this wrong silently
    transposes the Hessian, so it is pinned here."""
    vol, _ = _volume()
    p = _write_nifti(tmp_path, vol, "u.nii.gz")
    got, sp = load_field(p, spacing=SP, verbose=False)
    assert got.shape == vol.shape
    assert np.abs(got - vol).max() < 1e-9
    kept, _ = load_field(p, spacing=SP, axis_order="keep", verbose=False)
    assert np.abs(kept - np.transpose(vol, (2, 1, 0))).max() < 1e-9


def test_header_spacing_is_used(tmp_path):
    vol, _ = _volume()
    p = _write_nifti(tmp_path, vol, "u.nii.gz")
    _, sp = load_field(p, verbose=False)
    assert all(abs(v - SP*1000) < 1e-6 for v in sp)      # header is in mm


def test_complex_needs_an_explicit_component(tmp_path):
    vol, _ = _volume()
    p = str(tmp_path / "c.npy")
    np.save(p, vol + 1j*vol*0.5)
    with pytest.raises(ValueError, match="complex"):
        load_field(p, spacing=SP, verbose=False)
    re, _ = load_field(p, spacing=SP, component="real", verbose=False)
    im, _ = load_field(p, spacing=SP, component="imag", verbose=False)
    assert np.abs(re - np.transpose(vol, (2, 1, 0))).max() < 1e-9
    assert np.abs(im - 0.5*np.transpose(vol, (2, 1, 0))).max() < 1e-9


def test_missing_spacing_is_an_error(tmp_path):
    vol, _ = _volume()
    p = str(tmp_path / "u.npy")
    np.save(p, vol)
    with pytest.raises(ValueError, match="spacing"):
        load_field(p, verbose=False)


def test_mask_shape_mismatch_is_an_error(tmp_path):
    vol, mask = _volume()
    p = _write_nifti(tmp_path, mask.astype(float), "m.nii.gz")
    with pytest.raises(ValueError, match="does not match"):
        load_mask(p, (5, 5, 5), verbose=False)


def test_save_maps_writes_and_restores_axis_order(tmp_path):
    a = np.arange(N**3, dtype=float).reshape(N, N, N)
    written = save_maps({"kappa": a}, str(tmp_path / "out"), spacing=(SP,)*3,
                        verbose=False)
    assert written == ["kappa.nii.gz"]
    back = np.asarray(nib.load(str(tmp_path / "out" / "kappa.nii.gz")).dataobj)
    assert np.abs(back - np.transpose(a, (2, 1, 0))).max() < 1e-9


def test_cli_coverage_only_runs(tmp_path, capsys):
    from dircurv.__main__ import main
    vol, mask = _volume()
    pv = _write_nifti(tmp_path, vol, "u.nii.gz")
    pm = _write_nifti(tmp_path, mask.astype(float), "m.nii.gz")
    rc = main([pv, "--mask", pm, "--spacing", str(SP), "--coverage-only"])
    assert rc == 0
    assert "coverage" in capsys.readouterr().out


def test_cli_full_run_writes_maps(tmp_path):
    """The default path is the lattice one, which has no C3 pilot and therefore
    no rho map."""
    from dircurv.__main__ import main
    vol, mask = _volume()
    pv = _write_nifti(tmp_path, vol, "u.nii.gz")
    pm = _write_nifti(tmp_path, mask.astype(float), "m.nii.gz")
    out = str(tmp_path / "maps")
    rc = main([pv, "--mask", pm, "--spacing", str(SP),
               "--sigma-relative", "0.01", "--out", out, "--step", "6"])
    assert rc == 0
    for name in ("verdict", "kappa", "order", "H11", "H33", "n_directions"):
        assert os.path.exists(os.path.join(out, name + ".nii.gz"))


def test_cli_interpolate_flag_uses_the_other_path(tmp_path):
    """--interpolate reaches the tricubic path, which does run a C3 pilot and
    therefore does write rho."""
    from dircurv.__main__ import main
    vol, mask = _volume()
    pv = _write_nifti(tmp_path, vol, "u.nii.gz")
    pm = _write_nifti(tmp_path, mask.astype(float), "m.nii.gz")
    out = str(tmp_path / "maps_interp")
    rc = main([pv, "--mask", pm, "--spacing", str(SP), "--interpolate",
               "--sigma-relative", "0.01", "--out", out, "--step", "8",
               "--m", "24"])
    assert rc == 0
    for name in ("verdict", "kappa", "rho", "H11", "H33"):
        assert os.path.exists(os.path.join(out, name + ".nii.gz"))

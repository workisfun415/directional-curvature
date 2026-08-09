#!/usr/bin/env python3
"""
io.py
=====
Reading measured fields and writing reliability maps, so the package can be run
on a real dataset without writing any glue code.

Supported inputs
    .nii, .nii.gz    NIfTI, via nibabel; voxel spacing is taken from the header
    .mat             MATLAB v5-v7, via scipy.io; v7.3 needs h5py
    .npy, .npz       numpy

Two conventions that are easy to get wrong, and are handled explicitly here.

**Axis order.** The measurement modules index arrays as ``[z, y, x]`` -- slowest
axis first, matching how a stack of slices is usually held in Python. NIfTI and
MATLAB conventionally store ``[x, y, z]``. `load_field` therefore reverses the
axes by default, and `axis_order` lets you override that. Getting this wrong does
not raise an error; it silently transposes the Hessian, so the loaded shape and
spacing are always reported back.

**Units.** No conversion is performed. Spacing is used exactly as given, so the
recovered Hessian carries units of (field unit) per (spacing unit) squared. If
spacing is in millimetres the Hessian is per mm^2. `sigma` must be in the same
units as the field values, not relative. NIfTI headers usually give millimetres;
this is reported when a file is read so it can be checked.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

__all__ = ["load_field", "load_mask", "save_maps", "describe"]


# =====================================================================
# Reading
# =====================================================================

def _read_nifti(path):
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("reading NIfTI needs nibabel: pip install nibabel") from e
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    zooms = img.header.get_zooms()[:arr.ndim]
    units = img.header.get_xyzt_units()[0] or "unknown"
    return arr, tuple(float(z) for z in zooms), f"NIfTI, header units: {units}"


def _read_mat(path, key=None):
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError("reading .mat needs scipy: pip install scipy") from e
    try:
        d = loadmat(path)
    except NotImplementedError as e:
        raise NotImplementedError(
            "this looks like a MATLAB v7.3 file, which is HDF5. Either save it "
            "with '-v7' in MATLAB, or read it with h5py and pass the array "
            "directly to VolumeField.") from e
    keys = [k for k in d if not k.startswith("__")]
    if key is None:
        arrs = [k for k in keys if isinstance(d[k], np.ndarray) and d[k].ndim >= 2]
        if len(arrs) != 1:
            raise ValueError(f"{path} holds several arrays {arrs}; pass key= to "
                             "choose one")
        key = arrs[0]
    return np.asarray(d[key]), None, f"MATLAB .mat, variable '{key}'"


def _read_npy(path, key=None):
    if path.endswith(".npz"):
        d = np.load(path)
        keys = list(d.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError(f"{path} holds {keys}; pass key= to choose one")
            key = keys[0]
        return np.asarray(d[key]), None, f"npz, key '{key}'"
    return np.load(path), None, "npy"


def _read(path, key=None):
    low = path.lower()
    if low.endswith((".nii", ".nii.gz")):
        return _read_nifti(path)
    if low.endswith(".mat"):
        return _read_mat(path, key)
    if low.endswith((".npy", ".npz")):
        return _read_npy(path, key)
    raise ValueError(f"unrecognised extension: {path}")


def load_field(path, spacing=None, key=None, axis_order="reverse",
               component=None, verbose=True):
    """Load a measured field.

    Parameters
    ----------
    path : str
        .nii, .nii.gz, .mat, .npy or .npz
    spacing : float or sequence, optional
        Voxel spacing in the array's own axis order AFTER reordering. Required
        for formats that carry no header; for NIfTI it overrides the header.
    key : str, optional
        Variable name inside a .mat or .npz holding several arrays.
    axis_order : {"reverse", "keep"}
        NIfTI and MATLAB store [x, y, z]; the measurement modules want
        [z, y, x]. "reverse" (the default) transposes; "keep" does not.
    component : {"real", "imag", "abs", None}
        For complex data. "real" and "imag" are correct: the expansion is linear
        in the field, so it applies to each part separately. "abs" is accepted
        only if you ask for it explicitly and is NOT covered by the theory --
        the modulus is a nonlinear function of the displacement.
    """
    arr, hdr_spacing, what = _read(path, key)
    if axis_order == "reverse":
        arr = np.transpose(arr, tuple(reversed(range(arr.ndim))))
        if hdr_spacing is not None:
            hdr_spacing = tuple(reversed(hdr_spacing))
    elif axis_order != "keep":
        raise ValueError("axis_order must be 'reverse' or 'keep'")

    if np.iscomplexobj(arr):
        if component is None:
            raise ValueError(
                "the field is complex. Pass component='real' or 'imag' and run "
                "each separately: the operator is linear in the field, so it "
                "applies component-wise. component='abs' is available but the "
                "modulus is nonlinear and outside the theory.")
        if component == "real":
            arr = arr.real
        elif component == "imag":
            arr = arr.imag
        elif component == "abs":
            arr = np.abs(arr)
        else:
            raise ValueError("component must be 'real', 'imag' or 'abs'")

    arr = np.ascontiguousarray(arr, dtype=float)
    sp = spacing if spacing is not None else hdr_spacing
    if sp is None:
        raise ValueError(f"{path} carries no spacing in its header; pass "
                         "spacing=, in the units you want the Hessian expressed "
                         "in (Hessian is field-unit per spacing-unit squared)")
    sp = np.atleast_1d(np.asarray(sp, float))
    if sp.size == 1:
        sp = np.repeat(sp, arr.ndim)
    if sp.size != arr.ndim:
        raise ValueError(f"spacing has {sp.size} entries for a {arr.ndim}D array")

    if verbose:
        print(f"loaded {os.path.basename(path)}  [{what}]")
        print(f"  shape {arr.shape}  (axis order "
              f"{'reversed to [z,y,x]' if axis_order == 'reverse' else 'kept'})")
        print(f"  spacing {tuple(round(float(v), 6) for v in sp)}  "
              "-- units are yours; no conversion is applied")
        if component:
            print(f"  complex input, using the {component} part")
    return arr, tuple(float(v) for v in sp)


def load_mask(path, shape, key=None, axis_order="reverse", verbose=True):
    """Load a validity mask and check it against the field shape."""
    arr, _, what = _read(path, key)
    if axis_order == "reverse":
        arr = np.transpose(arr, tuple(reversed(range(arr.ndim))))
    m = np.asarray(arr) > 0
    if m.shape != tuple(shape):
        raise ValueError(f"mask shape {m.shape} does not match field {tuple(shape)}")
    if verbose:
        frac = 100.0 * m.mean()
        print(f"loaded mask {os.path.basename(path)} [{what}]  "
              f"{frac:.1f}% of voxels valid")
    return m


# =====================================================================
# Writing
# =====================================================================

def save_maps(maps, outdir, spacing=None, axis_order="reverse", template=None,
              verbose=True):
    """Write each map to disk, as NIfTI when nibabel is available and .npy
    otherwise. Arrays are transposed back to the input axis order so they
    overlay on the original data."""
    os.makedirs(outdir, exist_ok=True)
    try:
        import nibabel as nib
        have_nib = True
    except ImportError:
        have_nib = False
    written = []
    for name, arr in maps.items():
        a = np.asarray(arr, float)
        if axis_order == "reverse":
            a = np.transpose(a, tuple(reversed(range(a.ndim))))
        if have_nib and a.ndim == 3:
            if template is not None:
                aff = nib.load(template).affine
            else:
                sp = ([1.0] * 3 if spacing is None
                      else list(reversed(list(spacing)))[:3])
                aff = np.diag(list(sp) + [1.0])
            nib.save(nib.Nifti1Image(a, aff), os.path.join(outdir, name + ".nii.gz"))
            written.append(name + ".nii.gz")
        else:
            np.save(os.path.join(outdir, name + ".npy"), a)
            written.append(name + ".npy")
    if verbose:
        print(f"wrote {len(written)} maps to {outdir}/")
        print("  " + ", ".join(sorted(written)))
    return written


def describe(maps, verdict_codes=None):
    """Print a short summary of a reliability result."""
    from .grid3d import VERDICT_CODE as V3
    codes = verdict_codes or V3
    names = {v: k for k, v in codes.items()}
    v = maps.get("verdict")
    if v is not None:
        scored = int((v > 0).sum())
        print(f"scored voxels: {scored}")
        for c in sorted(set(np.asarray(v)[np.asarray(v) > 0].ravel().tolist())):
            n = int((v == c).sum())
            print(f"  {names.get(c, c):<10} {n:>7}  ({100*n/max(scored,1):.1f}%)")
    for k in ("kappa", "rho", "solid_angle", "aperture"):
        if k in maps:
            a = np.asarray(maps[k], float)
            a = a[np.isfinite(a)]
            if a.size:
                print(f"{k:<12} min {a.min():.4g}  median {np.median(a):.4g}  "
                      f"max {a.max():.4g}")

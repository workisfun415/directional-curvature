#!/usr/bin/env python3
"""
fusion.py -- EXPERIMENTAL
=========================
Convex fusion of a directional Hessian with a one-sided differencing Hessian.

    H_fused = w * H_directional + (1 - w) * H_onesided

Read this before using it.

WHAT THE EVIDENCE IS
--------------------
Evaluated on the public multifrequency MRE dataset of Feng and colleagues
(Sci Data 2025, doi:10.1038/s41597-025-05968-9): an agar phantom with hard and
soft inclusions, a human brain and a human liver, at 30, 40, 50 and 60 Hz.

On the phantom, where a piecewise-constant structure gives an error measure that
uses no reference reconstruction, w = 0.65 was chosen on one spatial half and
evaluated on the other. It beat one-sided differencing at all four frequencies on
BOTH scatter and bias.

With that weight then FROZEN and applied to brain and liver -- organs it was
never fitted on -- it reduced departure from the reference reconstruction in
7 of 8 acquisitions. Restricted to 30-50 Hz: 6 of 6, mean paired difference
0.220 (95% CI 0.165 to 0.274), sign test p = 0.031, median reduction 33%.

The gain is not simply variance reduction. Averaging a one-sided estimate with a
NEIGHBOURING one-sided estimate improved scatter from 0.339 to 0.302; averaging
with the directional estimate at the SAME voxel gave 0.181. Tensor fusion also
beat averaging the two scalar moduli at 30, 40 and 50 Hz in both organs.

WHAT THE EVIDENCE IS NOT
------------------------
w = 0.65 IS NOT UNIVERSALLY OPTIMAL. The per-acquisition optimum ranged from
0.25 to 0.75. At 60 Hz fusion was worse than one-sided differencing in liver and
gave almost no gain in brain; a plausible but unproven reason is that the two
estimators converge there, leaving no complementary information to exploit.

For brain and liver the error measure is departure from a reference
RECONSTRUCTION, not from ground truth. Only the phantom supports a claim about
physical error.

This is eight acquisitions of two organs and one phantom from a single study. It
is not a validated reconstruction method and must not be used clinically.

RECOMMENDED USE
---------------
Report the fused estimate alongside both endpoints, not instead of them, and
inspect the weight curve on your own data before trusting any single value.
`sweep_weights` is provided for exactly that.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fuse_hessians", "sweep_weights", "W_REPORTED", "evidence"]

W_REPORTED = 0.65
"""The weight evaluated in the study described in the module docstring. It is a
reported value, not a recommended default, and it is not optimal at 60 Hz."""


def fuse_hessians(H_directional, H_onesided, weight=W_REPORTED):
    """Convex combination of two Hessian estimates at the same location.

    Parameters
    ----------
    H_directional, H_onesided : (n, n) array
        Two estimates of the same Hessian. Both may be complex, as when the real
        and imaginary parts of a complex displacement have been processed
        separately and recombined.
    weight : float in [0, 1]
        Fraction given to the directional estimate. 0 returns the one-sided
        estimate unchanged, 1 returns the directional one.

    Returns
    -------
    (n, n) array, symmetrised.

    Notes
    -----
    No weight is chosen for you beyond the reported value, and the reported value
    is not optimal everywhere -- see the module docstring. If either estimate is
    None the other is returned, since a convex combination with a missing
    estimate is not defined.
    """
    if H_directional is None:
        return H_onesided
    if H_onesided is None:
        return H_directional
    w = float(weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {w}")
    A = np.asarray(H_directional)
    B = np.asarray(H_onesided)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    H = w * A + (1.0 - w) * B
    return 0.5 * (H + H.T)          # both inputs are symmetric; enforce it


def sweep_weights(H_directional, H_onesided, weights=None):
    """Fused estimates across a range of weights, for inspecting the curve.

    Because the optimal weight varied from 0.25 to 0.75 across acquisitions in
    the study, looking at the whole curve on your own data is more informative
    than accepting any single value.

    Returns a dict mapping weight to fused Hessian.
    """
    if weights is None:
        weights = [0.0, 0.25, 0.5, 0.65, 0.75, 1.0]
    return {w: fuse_hessians(H_directional, H_onesided, w) for w in weights}


def evidence():
    """Print the evidence and its limits. Deliberately unavoidable."""
    print(__doc__)

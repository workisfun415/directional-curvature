"""Bias: Lemma 6.2 coefficients (numerical) and Remark 6.3 (central symmetry)."""
from __future__ import annotations

import math

import numpy as np

from .. import common as C
from ..impl_a import factorized as A
from ..impl_b import physical as B


def numeric_bias_coefficients(n, stencil, thetas, h, rng, m=80):
    """Noise-free fitted bias / h^p compared with Lemma 6.2, for both implementations."""
    nu = stencil.p + 2
    f = C.make_test_function(np.eye(n), [nu], n, rng)
    pred = C.sector_bias_formula(f.tensors[nu], stencil.beta)
    des = C.random_filled_design(m, n - 1, rng)
    rows = []
    for th in thetas:
        q = B.curvature_data(f, des, th, stencil, h)
        HA = A.fit_H(des, th, q)
        HB = B.fit_H(des, th, q)
        EA = (HA - f.H) / h**stencil.p
        d = n - 1
        rows.append({
            "theta": th,
            "axial_err": abs(EA[-1, -1] - pred["B0"]),
            "mixed_err": float(np.abs(EA[:d, -1] - pred["B1_plain"]).max()),
            "tangential_err": float(np.abs(EA[:d, :d] - pred["B2"]).max()),
            "A_vs_B_max": float(np.abs(HA - HB).max()),
        })
    return rows


def parity_slopes(thetas, rng):
    """Tangential O(theta) vs O(theta^2) bias correction (n=3, cubic), per design."""
    n, nu = 3, 3
    T3 = C.symmetric_tensor(nu, n, rng)
    pred = C.sector_bias_formula(T3, 1.0)["B2"]
    designs = {
        "axis+pentagon": C.axis_plus_pentagon(),
        "centrally symmetric": C.centrally_symmetric_design(),
        "random filled": C.random_filled_design(40, 2, rng),
    }
    out = {}
    for name, des in designs.items():
        errs = []
        for th in thetas:
            q = np.array([C.contract(T3, C.ray(v, th)) for v in des.V])  # exact bias data per unit beta h^p
            Hb = B.fit_H(des, th, q)
            errs.append(float(np.linalg.norm(Hb[:2, :2] - pred)))
        slope = float(np.polyfit(np.log(thetas), np.log(errs), 1)[0])
        out[name] = {"errors": errs, "slope": slope}
    return out

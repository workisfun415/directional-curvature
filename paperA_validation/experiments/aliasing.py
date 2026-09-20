"""Section 7 (radial aliasing) and Section 11 (stencil vs regression) experiments."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .. import common as C
from ..impl_a import factorized as A
from ..impl_a import regression as RA
from ..impl_b import physical as B
from ..impl_b import regression_physical as RB

THETAS = [0.1, 0.05, 0.025, 0.0125]


def unit_tensor(idx, n=3):
    T = np.zeros((n,) * 3)
    for p in set(itertools.permutations(idx)):
        T[p] = 1.0
    return T


def planar_disk(m, rng, d=2):
    V = []
    while len(V) < m:
        x = rng.uniform(-1, 1, d)
        if x @ x <= 1:
            V.append(x)
    return np.array(V)


def cubic_fvals(T3, pts):
    return np.array([C.contract(T3, y) / 6 for y in pts])


def sector_norms(Hm):
    """Physical sector norms: axial |H_nn|, mixed ||b~|| = sqrt2 ||b||, tangential ||T||_F."""
    n = Hm.shape[0]
    d = n - 1
    return np.array([abs(Hm[-1, -1]), math.sqrt(2) * np.linalg.norm(Hm[:d, -1]), np.linalg.norm(Hm[:d, :d])])


def ab_agreement(des, T3, theta, h):
    pts = des.points(theta, h)
    f = cubic_fvals(T3, pts)
    HA = RA.fit_H(des, theta, h, f)
    HB = RB.fit_H(pts, f)
    return float(np.abs(HA - HB).max())


PATHS = {  # name: (tensor index, cubic sector j, Hessian sector k)
    "f333->mixed": ((2, 2, 2), 0, 1),
    "f333->tangential": ((2, 2, 2), 0, 2),
    "f133->tangential": ((0, 2, 2), 1, 2),
    "f133->mixed (non-divergent)": ((0, 2, 2), 1, 1),
}


def alias_paths(designs: dict, h=0.05):
    """For each design and path: bias slope in theta, and the limit theta^{k-j} Bias_k / h
    from regression (implementation B) compared with the moment formula (implementation A)."""
    out = {}
    for dname, des in designs.items():
        out[dname] = {}
        for pname, (idx, j, k) in PATHS.items():
            T3 = unit_tensor(idx)
            vals = []
            for th in THETAS:
                pts = des.points(th, h)
                Hb = RB.fit_H(pts, cubic_fvals(T3, pts))
                vals.append(sector_norms(Hb)[k] / h)
            vals = np.array(vals)
            slope = float(np.polyfit(np.log(THETAS), np.log(np.maximum(vals, 1e-300)), 1)[0])
            limit_obs = float(vals[-1] * THETAS[-1] ** (k - j))
            limit_mom = float(RA.alias_limit_moment(des, T3, j)[k])
            out[dname][pname] = {"j": j, "k": k, "bias_over_h": vals.tolist(), "slope": slope,
                                 "limit_regression": limit_obs, "limit_moment_formula": limit_mom}
    return out


def stencil_vs_regression(V, radii_common, h, sigma, thetas, rng, n_mc=4000):
    """Same physical points (product design + base point): 3-point stencil + LS vs quadratic regression.
    Reports noise-free tangential bias (cubic test tensor) and exact tangential variance for both,
    and a Monte Carlo check of the regression variance."""
    assert tuple(radii_common) == (1.0, 2.0), "3-point stencil (1,-2,1) uses radii h and 2h"
    T3 = C.symmetric_tensor(3, 3, rng)
    des = RA.RegressionDesign(V, [tuple(radii_common)] * len(V))
    sd = C.Design("rays", V, np.full(len(V), 1 / len(V)))
    st = C.STENCILS["3pt"]
    rows = []
    for th in thetas:
        pts = des.points(th, h)
        f = cubic_fvals(T3, pts)
        H_reg = RB.fit_H(pts, f)
        q = np.array([(C.contract(T3, 2 * h * C.ray(v, th)) / 6 - 2 * C.contract(T3, h * C.ray(v, th)) / 6)
                      / h**2 for v in V])
        H_st = B.fit_H(sd, th, q)
        bias_reg = sector_norms(H_reg)[2]
        bias_st = sector_norms(H_st)[2]
        var_st = A.sector_variances(sd, th, st, sigma, h, R=1)[2]
        covB = RB.cov_physical_frob(pts, sigma)
        idx = [k for k, (i, jj) in enumerate(C.frob_index(3)) if i < 2 and jj < 2]
        var_reg = float(np.trace(covB[np.ix_(idx, idx)]))
        noisy = f[:, None] + rng.normal(0, sigma, size=(len(f), n_mc))
        E = RB.fit_H(pts, noisy) - H_reg
        var_reg_mc = float(C.sector_sq_errors(E)[:, 2].mean())
        rows.append({"theta": th, "n_evaluations": len(pts),
                     "tangential_bias_stencil": bias_st, "tangential_bias_regression": bias_reg,
                     "tangential_var_stencil": var_st, "tangential_var_regression": var_reg,
                     "tangential_var_regression_mc": var_reg_mc,
                     "variance_ratio_stencil_over_regression": var_st / var_reg,
                     "bias_ratio_stencil_over_regression": bias_st / bias_reg})
    return rows

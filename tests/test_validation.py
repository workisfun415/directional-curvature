"""Paper A validation tests.

Fast (default `pytest`): exact constants, symbolic bias identities, A/B coordinate
equivalence, the shared-base noise identity. Slow (`pytest -m slow`): Monte Carlo,
bootstrap, span cap, parity slopes, design search.
"""
import math

import numpy as np
import pytest
import sympy as sp
import yaml
from pathlib import Path

from paperA_validation import common as C
from paperA_validation.experiments import accuracy as acc
from paperA_validation.experiments import bias as bias_exp
from paperA_validation.experiments import constants as K
from paperA_validation.impl_a import factorized as A
from paperA_validation.impl_b import physical as B
from paperA_validation.impl_b import symbolic_bias as SB

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "validation.yaml").read_text())["regenerated"]
TOL = CFG["tolerances"]


# ---------------------------------------------------------------- symbolic (exact)
@pytest.mark.parametrize("key,beta,s2,s2r", [("3pt", 1, 6, 5), ("4pt", sp.Rational(-11, 12), 46, 42)])
def test_stencil_constants_exact(key, beta, s2, s2r):
    st = C.STENCILS[key]
    r = SB.stencil_constants([int(x) for x in st.weights], st.p)
    assert r["moment_conditions_hold"]
    assert r["beta"] == beta and r["s2"] == s2 and r["s2_ray"] == s2r


@pytest.mark.parametrize("n,nu", [(3, 3), (3, 4), (4, 3), (4, 4)])
def test_sector_bias_symbolic(n, nu):
    r = SB.sector_bias_symbolic(n, nu)
    assert r["agree"]
    assert r["mixed_tilde_factor"] == sp.sqrt(2) * sp.Rational(nu, 2)


# ---------------------------------------------------------------- exact constants
@pytest.mark.parametrize("d", [1, 2, 3])
def test_uniform_planar_limit_exact(d):
    assert K.uniform_ball_lambda_exact(d) == K.predicted_uniform(d)


def test_uniform_constants_special_cases():
    assert math.isclose(K.kappa_limit_uniform(2), 3 * math.sqrt(5) / 2)
    assert math.isclose(K.kappa_limit_uniform(3), math.sqrt(24))
    assert math.isclose(K.kappa_limit_optimal(3), math.sqrt(8))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_optimal_design_value(d):
    assert math.isclose(K.S0_numeric(C.optimal_design(d)), float(K.predicted_optimal(d)),
                        rel_tol=TOL["exact_rel"] * 1e3)


def test_axis_pentagon_value():
    assert math.isclose(K.S0_numeric(C.axis_plus_pentagon()), 5 / 72, rel_tol=1e-9)


@pytest.mark.parametrize("fn,limit", [
    (lambda th: C.axis_plus_pentagon(), 5 / 72),
    (lambda th: C.centre_plus_rim(2, 0.5, centre_copies=5), 1 / 8),
    (lambda th: C.uniform_cap_quadrature(th), 1 / 24),
])
def test_finite_theta_intercepts(fn, limit):
    rows = K.ladder(fn, CFG["constants"]["theta_ladder"], "A")
    assert math.isclose(K.intercept_theta2(rows), limit, rel_tol=TOL["finite_theta_intercept_rel"])
    assert abs(rows[-1]["lmax"] - 1) < 1e-3


def test_single_angle_rim_singular():
    V = C.regular_polygon(7)
    lmin, _ = B.lambda_F(C.Design("rim", V, np.full(7, 1 / 7)), 0.1)
    assert lmin < 1e-12


# ---------------------------------------------------------------- A/B equivalence
@pytest.mark.parametrize("n", [2, 3, 4])
def test_A_B_conditioning_agree(n):
    rng = np.random.default_rng(0)
    des = C.random_filled_design(3 * n * n, n - 1, rng)
    for th in (0.3, 0.05):
        a, b = A.lambda_F(des, th), B.lambda_F(des, th)
        assert np.allclose(a, b, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("n", [3, 4])
def test_A_B_fit_agree_and_btilde_map(n):
    rng = np.random.default_rng(1)
    des = C.random_filled_design(40, n - 1, rng)
    H = C.symmetric_tensor(2, n, rng)
    th = 0.07
    q = np.array([C.ray(v, th) @ H @ C.ray(v, th) for v in des.V])
    HA, HB = A.fit_H(des, th, q), B.fit_H(des, th, q)
    assert np.abs(HA - HB).max() < TOL["implementation_agreement"]
    # b~ = sqrt(2) b: implementation A's raw coordinate against B's plain entries
    z = np.linalg.lstsq(A.design_matrix(des, th), q, rcond=None)[0]
    assert np.allclose(z[1:n], math.sqrt(2) * HB[: n - 1, n - 1], atol=1e-8)


def test_common_mode_is_identity():
    for n in (2, 3, 4):
        des = C.random_filled_design(30, n - 1, np.random.default_rng(n))
        HB = B.fit_H(des, 0.2, np.ones(30))
        assert np.allclose(HB, np.eye(n), atol=1e-9)


# ---------------------------------------------------------------- slow
@pytest.mark.slow
def test_variance_matches_theorem_5_2():
    rng = np.random.default_rng(3)
    des = C.random_filled_design(40, 2, rng)
    rows = acc.variance_check(des, [0.4, 0.1, 0.025], C.STENCILS["3pt"], 1e-7, 1e-2, 1, 20000, rng)
    for r in rows:
        assert np.allclose(r["ratio"], 1.0, atol=0.05)


@pytest.mark.slow
def test_parity_slopes():
    out = bias_exp.parity_slopes(CFG["bias"]["parity_theta"], np.random.default_rng(4))
    assert abs(out["axis+pentagon"]["slope"] - 1) < 0.15
    assert abs(out["centrally symmetric"]["slope"] - 2) < 0.15


@pytest.mark.slow
@pytest.mark.parametrize("key", ["3pt", "4pt"])
def test_span_law_exponents(key):
    mc = dict(CFG["montecarlo"])
    r = acc.montecarlo_exponents(mc, C.STENCILS[key], "shared_base", C.seed_for(CFG["base_seed"], "test_" + key))
    for s in r["sectors"]:
        for lab in ("h", "E"):
            ci = s[f"{lab}_slope_ci95"]
            pred = s[f"{lab}_slope_pred_finite_theta"]
            est = s[f"{lab}_slope_est"]
            assert (ci[0] <= pred <= ci[1]) or abs(est - pred) <= 0.01


@pytest.mark.slow
def test_cap_transition_slope():
    r = acc.cap_transition(CFG["montecarlo"], CFG["cap"], C.seed_for(CFG["base_seed"], "test_cap"))
    assert r["capped_slope"] is not None and abs(r["capped_slope"] + 2) < 0.1

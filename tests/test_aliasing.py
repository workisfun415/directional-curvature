"""Section 7 aliasing tests: A = B = moment formula, exact witness, product-design cancellation."""
import numpy as np
import pytest
import sympy as sp

from paperA_validation.experiments import aliasing as AL
from paperA_validation.impl_a import regression as RA
from paperA_validation.impl_b import symbolic_alias as SA

RNG = np.random.default_rng(0)
V = AL.planar_disk(40, RNG)
NONPRODUCT = RA.RegressionDesign(V, [tuple(x) for x in RNG.uniform(0.2, 2, size=(40, 2))], name="non-product")
PRODUCTS = {rs: RA.RegressionDesign(V, [rs] * 40) for rs in [(1.0, 2.0), (0.3, 0.9, 2.0), (0.5, 1.0, 1.5, 2.0)]}


@pytest.mark.parametrize("des", [NONPRODUCT] + list(PRODUCTS.values()))
def test_regression_A_equals_B(des):
    for th in (0.2, 0.02):
        assert AL.ab_agreement(des, AL.unit_tensor((2, 2, 2)), th, 0.05) < 1e-9


def test_exact_witness_nonproduct_nonzero_product_zero():
    g_np = SA.gamma02_exact()
    g_p = SA.gamma02_exact(radii=SA.WITNESS_RADII_PRODUCT)
    assert any(sp.simplify(x) != 0 for x in g_np)
    assert all(sp.simplify(x) == 0 for x in g_p)


@pytest.mark.parametrize("rs", list(PRODUCTS))
@pytest.mark.parametrize("idx,j,k", [((2, 2, 2), 0, 1), ((2, 2, 2), 0, 2), ((0, 2, 2), 1, 2)])
def test_product_design_divergent_blocks_vanish(rs, idx, j, k):
    assert RA.alias_limit_moment(PRODUCTS[rs], AL.unit_tensor(idx), j)[k] < 1e-10


def test_gamma02_moment_formula_matches_regression():
    T3 = AL.unit_tensor((2, 2, 2))
    mom = RA.alias_limit_moment(NONPRODUCT, T3, 0)[2]
    th, h = 0.005, 0.05
    pts = NONPRODUCT.points(th, h)
    reg = AL.sector_norms(AL.RB.fit_H(pts, AL.cubic_fvals(T3, pts)))[2] * th**2 / h
    assert mom > 1e-3 and abs(reg - mom) / mom < 1e-3


@pytest.mark.slow
def test_alias_slopes():
    out = AL.alias_paths({"np": NONPRODUCT, "p": PRODUCTS[(0.5, 1.0, 1.5, 2.0)]})
    exp = {"f333->mixed": -1, "f333->tangential": -2, "f133->tangential": -1}
    for name, e in exp.items():
        assert abs(out["np"][name]["slope"] - e) < 0.05
        assert out["p"][name]["slope"] > -0.05          # no divergence in the product design


@pytest.mark.slow
def test_stencil_vs_regression_variance_exact_vs_mc():
    rows = AL.stencil_vs_regression(V, (1.0, 2.0), 0.05, 1e-7, [0.1, 0.025], np.random.default_rng(1))
    for r in rows:
        assert abs(r["tangential_var_regression_mc"] / r["tangential_var_regression"] - 1) < 0.05
        assert r["variance_ratio_stencil_over_regression"] > 1


def test_exact_witness_all_divergent_blocks():
    np_ = SA.alias_blocks_exact()
    p_ = SA.alias_blocks_exact(radii=SA.WITNESS_RADII_PRODUCT)
    for key in ("Gamma01", "Gamma02", "Gamma12"):
        assert any(sp.simplify(x) != 0 for x in np_[key])
        assert all(sp.simplify(x) == 0 for x in p_[key])
    assert [sp.simplify(a - b) for a, b in zip(np_["Gamma02"], SA.gamma02_exact())] == [0, 0, 0]

"""Exact (SymPy) alias coefficient Gamma_02 for rational designs: the non-product witness.

Built in physical coordinates at theta = 0 (planar scaled features), independently of
impl_a.regression. Used to prove generic non-vanishing (Theorem 7.1(iii)): Gamma_02 is a
rational function of the design coordinates, nonzero at this witness, hence nonzero
outside a proper algebraic subset of non-product designs.
"""
import sympy as sp

S2 = sp.sqrt(2)
WITNESS_V = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
             (sp.Rational(1, 2), sp.Rational(1, 2)), (sp.Rational(-1, 2), sp.Rational(1, 2)),
             (sp.Rational(1, 2), sp.Rational(-1, 2))]
WITNESS_RADII_NONPRODUCT = [(1, 2), (1, 2), (1, 3), (1, 2), (2, 3), (1, 2), (1, 3), (2, 3)]
WITNESS_RADII_PRODUCT = [(1, 2)] * len(WITNESS_V)


def gamma02_exact(V=WITNESS_V, radii=WITNESS_RADII_NONPRODUCT):
    F, Q, d = [], [], []
    for (a, b), rs in zip(V, radii):
        for r in rs:
            r = sp.Integer(r)
            F.append([1, r, r * a, r * b, r**2 / 2, r**2 * a, r**2 * b])
            Q.append([r**2 * a**2 / 2, r**2 * b**2 / 2, r**2 * S2 * a * b / 2])
            d.append(r**3 / 6)
    F.append([1, 0, 0, 0, 0, 0, 0]); Q.append([0, 0, 0]); d.append(0)   # base point x0
    F, Q, d = sp.Matrix(F), sp.Matrix(Q), sp.Matrix(d)
    MFF, MFQ = F.T * F, F.T * Q
    S = Q.T * Q - MFQ.T * MFF.inv() * MFQ                 # Schur complement
    C = Q.T * d - MFQ.T * MFF.inv() * (F.T * d)           # residualized cross moment
    return [sp.nsimplify(sp.simplify(x)) for x in (S.inv() * C)]


def alias_blocks_exact(V=WITNESS_V, radii=WITNESS_RADII_NONPRODUCT):
    """Exact Gamma_01 (f_nnn -> mixed b~), Gamma_02 (f_nnn -> t') and Gamma_12 (f_1nn -> t')."""
    X, d0, d1 = [], [], []
    for (a, b), rs in zip(V, radii):
        for r in rs:
            r = sp.Integer(r)
            X.append([1, r, r * a, r * b, r**2 / 2, S2 / 2 * r**2 * a, S2 / 2 * r**2 * b,
                      r**2 * a**2 / 2, r**2 * b**2 / 2, r**2 * S2 * a * b / 2])
            d0.append(r**3 / 6)                   # P_0 = f_nnn = 1
            d1.append(r**3 * 3 * a / 6)           # P_1 = 3 f_1nn v_1 with f_1nn = 1
    X.append([1] + [0] * 9); d0.append(0); d1.append(0)
    X = sp.Matrix(X)
    G = (X.T * X).inv()
    w0, w1 = G * X.T * sp.Matrix(d0), G * X.T * sp.Matrix(d1)
    simp = lambda M: [sp.nsimplify(sp.simplify(x)) for x in M]
    return {"Gamma01": simp(w0[5:7]), "Gamma02": simp(w0[7:10]), "Gamma12": simp(w1[7:10])}

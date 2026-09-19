"""Shared definitions used by both implementations.

Only things that are *definitions* in the manuscript live here: the cap
parametrisation (2.1), the physical Frobenius coordinates of a symmetric
matrix, the stencils of Table 1, the designs, and the test functions.
The two implementations (impl_a, impl_b) must not share any fitting,
Gram-matrix or covariance code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------- geometry
def ray(v: np.ndarray, theta: float) -> np.ndarray:
    """Unit direction u(v, theta) = (theta v, sqrt(1 - theta^2 |v|^2)), eq. (2.1)."""
    v = np.asarray(v, float)
    return np.r_[theta * v, math.sqrt(1.0 - theta**2 * float(v @ v))]


def frob_index(n: int):
    """Ordering of physical Frobenius coordinates: diagonal first, then (i<j)."""
    return [(i, i) for i in range(n)] + [(i, j) for i in range(n) for j in range(i + 1, n)]


def sym_to_frob(H: np.ndarray) -> np.ndarray:
    n = H.shape[0]
    return np.array([H[i, j] if i == j else SQRT2 * H[i, j] for i, j in frob_index(n)])


def frob_to_sym(x: np.ndarray, n: int) -> np.ndarray:
    H = np.zeros((n, n))
    for val, (i, j) in zip(x, frob_index(n)):
        if i == j:
            H[i, i] = val
        else:
            H[i, j] = H[j, i] = val / SQRT2
    return H


def sector_sq_errors(E: np.ndarray) -> np.ndarray:
    """Squared Frobenius norm of the error in each physical sector (axial, mixed, tangential).

    The mixed sector counts both (alpha, n) and (n, alpha) entries, i.e. it is
    the squared norm in b-tilde = sqrt(2) b coordinates.
    """
    n = E.shape[-1]
    d = n - 1
    ax = E[..., n - 1, n - 1] ** 2
    mx = 2.0 * np.sum(E[..., :d, n - 1] ** 2, axis=-1)
    tg = np.sum(E[..., :d, :d] ** 2, axis=(-2, -1))
    return np.stack([ax, mx, tg], axis=-1)


# ---------------------------------------------------------------- stencils
@dataclass(frozen=True)
class Stencil:
    name: str
    weights: tuple
    p: int

    @property
    def nodes(self):
        return tuple(range(len(self.weights)))

    @property
    def beta(self) -> float:
        k = self.p + 2
        return sum(w * j**k for w, j in zip(self.weights, self.nodes)) / math.factorial(k)

    @property
    def w0(self) -> float:
        return self.weights[0]

    @property
    def s2(self) -> float:
        return float(sum(w * w for w in self.weights))

    @property
    def s2_ray(self) -> float:
        return self.s2 - self.w0**2


STENCILS = {
    "3pt": Stencil("3pt", (1.0, -2.0, 1.0), 1),
    "4pt": Stencil("4pt", (2.0, -5.0, 4.0, -1.0), 2),
}


# ---------------------------------------------------------------- designs
@dataclass
class Design:
    """Planar points v_i in B^d with weights summing to one."""
    name: str
    V: np.ndarray
    w: np.ndarray

    @property
    def d(self) -> int:
        return self.V.shape[1]

    @property
    def m(self) -> int:
        return self.V.shape[0]


def regular_polygon(k: int, r: float = 1.0, phase: float = 0.0) -> np.ndarray:
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.c_[np.cos(a), np.sin(a)]


def icosahedron() -> np.ndarray:
    phi = (1 + 5**0.5) / 2
    pts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            pts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    P = np.array(pts, float)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def rim_4design(d: int) -> np.ndarray:
    """A spherical 4-design on S^{d-1} (angular moments through degree 4 uniform)."""
    if d == 1:
        return np.array([[1.0], [-1.0]])
    if d == 2:
        return regular_polygon(5)
    if d == 3:
        return icosahedron()
    raise ValueError("rim 4-design implemented for d <= 3")


def centre_plus_rim(d: int, centre_weight: float, centre_copies: int = 1) -> Design:
    rim = rim_4design(d)
    V = np.vstack([np.zeros((centre_copies, d)), rim])
    w = np.r_[np.full(centre_copies, centre_weight / centre_copies),
              np.full(len(rim), (1 - centre_weight) / len(rim))]
    return Design(f"centre({centre_weight:.4g})+rim d={d}", V, w)


def axis_plus_pentagon() -> Design:
    V = np.vstack([np.zeros((1, 2)), regular_polygon(5)])
    return Design("axis+pentagon", V, np.full(6, 1 / 6))


def optimal_design(d: int) -> Design:
    cw = 0.5 if d <= 2 else 2.0 / (d + 2)
    return centre_plus_rim(d, cw)


def centrally_symmetric_design() -> Design:
    V = np.vstack([np.zeros((1, 2)), regular_polygon(6, 1.0), regular_polygon(6, 0.5)])
    return Design("axis+2 hexagons (centrally symmetric)", V, np.full(len(V), 1 / len(V)))


def random_filled_design(m: int, d: int, rng: np.random.Generator) -> Design:
    V = []
    while len(V) < m:
        x = rng.uniform(-1, 1, d)
        if x @ x <= 1:
            V.append(x)
    return Design(f"random filled m={m}", np.array(V), np.full(m, 1 / m))


def uniform_cap_quadrature(theta: float, n_polar: int = 64, n_az: int = 64) -> Design:
    """Uniform surface measure on the n=3 cap {u_3 >= sqrt(1-theta^2)}.

    Gauss-Legendre in u_3 (surface measure is uniform in u_3 for n=3) times an
    equispaced azimuth rule. Returned as planar points v = (u_1,u_2)/theta.
    """
    lo = math.sqrt(1 - theta**2)
    x, wx = np.polynomial.legendre.leggauss(n_polar)
    u3 = lo + (1 - lo) * (x + 1) / 2
    wx = wx / wx.sum()
    az = 2 * np.pi * np.arange(n_az) / n_az
    V, W = [], []
    for c, wc in zip(u3, wx):
        s = math.sqrt(max(0.0, 1 - c * c)) / theta
        for a in az:
            V.append((s * math.cos(a), s * math.sin(a)))
            W.append(wc / n_az)
    return Design(f"uniform cap quadrature theta={theta}", np.array(V), np.array(W))


# ---------------------------------------------------------------- test functions
def symmetric_tensor(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    import itertools
    T = rng.normal(size=(n,) * order)
    S = sum(T.transpose(p) for p in itertools.permutations(range(order)))
    return S / math.factorial(order)


def contract(T: np.ndarray, x: np.ndarray) -> float:
    for _ in range(T.ndim):
        T = T @ x
    return float(T)


@dataclass
class ModelFunction:
    H: np.ndarray
    tensors: dict  # order -> symmetric tensor

    def __call__(self, x: np.ndarray) -> float:
        val = 0.5 * float(x @ self.H @ x)
        for k, T in self.tensors.items():
            val += contract(T, x) / math.factorial(k)
        return val


def make_test_function(H, orders, n, rng) -> ModelFunction:
    return ModelFunction(np.asarray(H, float), {k: symmetric_tensor(k, n, rng) for k in orders})


def sector_bias_formula(tensor: np.ndarray, beta: float) -> dict:
    """Lemma 6.2 leading coefficients (physical T, plain b) for a nu-th derivative tensor."""
    nu = tensor.ndim
    n = tensor.shape[0]
    d = n - 1
    idx_n = (n - 1,)
    f_n = tensor[idx_n * nu]
    f_a = np.array([tensor[(a,) + idx_n * (nu - 1)] for a in range(d)])
    G = np.array([[tensor[(a, b) + idx_n * (nu - 2)] for b in range(d)] for a in range(d)])
    B0 = beta * f_n
    B1_plain = beta * (nu / 2) * f_a
    B2 = beta * (math.comb(nu, 2) * G - (nu / 2 - 1) * f_n * np.eye(d))
    return {"B0": B0, "B1_plain": B1_plain, "B1_tilde": SQRT2 * B1_plain, "B2": B2}


def seed_for(base_seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible stream per named experiment."""
    tag = [ord(c) for c in name]
    return np.random.default_rng(np.random.SeedSequence([base_seed] + tag))

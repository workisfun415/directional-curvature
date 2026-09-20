"""Noise (Theorem 5.2), span law (Theorem 7.1), span cap (Cor. 7.2), order crossover."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar, nnls

from .. import common as C
from ..impl_a import factorized as A
from ..impl_b import physical as B


# ---------------------------------------------------------------- Theorem 5.2
def variance_check(design, thetas, stencil, sigma, h, R, n_draws, rng, shared_base=True):
    """Monte Carlo sector variance (impl B, evaluation-level noise) vs exact formula (impl A)."""
    rows = []
    for th in thetas:
        zero = np.zeros(design.m)
        noisy = B.noisy_curvature_data(zero, stencil, sigma, h, R, n_draws, rng, shared_base)
        H = B.fit_H(design, th, noisy)
        mc = C.sector_sq_errors(H).mean(axis=0)
        if shared_base:
            th_var = A.sector_variances(design, th, stencil, sigma, h, R)
        else:  # historical: every ray has independent full-stencil noise, no common term
            full = C.Stencil("hist", stencil.weights, stencil.p)
            th_var = A.sector_variances(design, th, _NoCommon(full), sigma, h, R)
        rows.append({"theta": th, "mc": mc.tolist(), "theory": th_var.tolist(),
                     "ratio": (mc / th_var).tolist()})
    return rows


class _NoCommon:
    """Stencil view for the historical convention: all noise ray-specific."""
    def __init__(self, st):
        self.s2_ray = st.s2
        self.w0 = 0.0


# ---------------------------------------------------------------- exact MSE
def exact_mse(f, design, theta, stencil, h, sigma, R=1):
    """Exact expected squared sector error: ||E H^ - H||^2 (noise-free fit) + Var (Theorem 5.2)."""
    q = B.curvature_data(f, design, theta, stencil, h)
    bias = C.sector_sq_errors(B.fit_H(design, theta, q) - f.H)
    return bias + A.sector_variances(design, theta, stencil, sigma, h, R)


def optimal_h(f, design, theta, stencil, sigma, R, k, h_max=None, bounds=(1e-5, 2.0)):
    lo, hi = bounds
    if h_max is not None:
        hi = min(hi, h_max)
    g = lambda lh: exact_mse(f, design, theta, stencil, math.exp(lh), sigma, R)[k]
    grid = np.linspace(math.log(lo), math.log(hi), 80)
    vals = [g(x) for x in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if a == b:
        return math.exp(a), vals[i]
    res = minimize_scalar(g, bounds=(a, b), method="bounded", options={"xatol": 1e-6})
    return math.exp(res.x), float(res.fun)


# ---------------------------------------------------------------- Theorem 7.1: MC + bootstrap
def _fit_mse_curve(hs, mse, p):
    """Relative least squares for MSE(h) = a h^{2p} + b h^{-4}, a, b >= 0."""
    X = np.c_[hs ** (2 * p), hs ** -4.0] / mse[:, None]
    coef, _ = nnls(X, np.ones_like(mse))
    a, b = coef
    if a <= 0 or b <= 0:
        return np.nan, np.nan
    hstar = (2 * b / (p * a)) ** (1 / (2 * p + 4))
    return hstar, math.sqrt(a * hstar ** (2 * p) + b * hstar ** -4)


def montecarlo_exponents(cfg, stencil, noise_model, rng, orders=None):
    n, m = cfg["n"], cfg["rays"]
    sigma, R = cfg["sigma"], cfg["R"]
    shared = noise_model == "shared_base"
    orders = orders or [stencil.p + 2]
    f = C.make_test_function(cfg["H"], orders, n, rng)
    design = C.random_filled_design(m, n - 1, rng)
    thetas = np.array(cfg["theta"])
    per_theta = []
    draws = cfg["mc_draws"]
    for th in thetas:
        # centre the h grid on the predicted leading-order optima of the three sectors
        V = A.sector_variances(design, th, stencil if shared else _NoCommon(stencil), sigma, 1.0, R)
        # finite-theta leading-order constants: exact bias data beta*D^nu f[u^nu] (no rounding), fitted
        Tnu = f.tensors[stencil.p + 2]
        qbias = np.array([stencil.beta * C.contract(Tnu, C.ray(v, th)) for v in design.V])
        Hb = B.fit_H(design, th, qbias)
        Cb = np.sqrt(C.sector_sq_errors(Hb))
        hpred = (2 * V / (stencil.p * Cb ** 2)) ** (1 / (2 * stencil.p + 4))
        Epred = np.sqrt(Cb ** 2 * hpred ** (2 * stencil.p) + V / hpred ** 4)
        lo = np.log10(hpred.min()) - cfg["h_window_decades"]
        hi = np.log10(hpred.max()) + cfg["h_window_decades"]
        hs = np.logspace(lo, hi, cfg["h_points"])
        err = np.empty((len(hs), draws, 3))
        for i, h in enumerate(hs):
            q = B.curvature_data(f, design, th, stencil, h)
            noisy = B.noisy_curvature_data(q, stencil, sigma, h, R, draws, rng, shared)
            err[i] = C.sector_sq_errors(B.fit_H(design, th, noisy) - f.H)
        per_theta.append({"theta": float(th), "hs": hs, "err": err, "h_pred": hpred, "E_pred": Epred})

    def estimates(idx=None):
        hstar = np.full((len(thetas), 3), np.nan)
        estar = np.full((len(thetas), 3), np.nan)
        for t, rec in enumerate(per_theta):
            e = rec["err"] if idx is None else rec["err"][:, idx[t], :]
            mse = e.mean(axis=1)
            for k in range(3):
                hstar[t, k], estar[t, k] = _fit_mse_curve(rec["hs"], mse[:, k], stencil.p)
        return hstar, estar

    sel = thetas <= cfg["slope_theta_max"] + 1e-12
    lt = np.log(thetas[sel])

    def slopes(hstar, estar):
        return ([float(np.polyfit(lt, np.log(hstar[sel, k]), 1)[0]) for k in range(3)],
                [float(np.polyfit(lt, np.log(estar[sel, k]), 1)[0]) for k in range(3)])

    h0, e0 = estimates()
    sh, se = slopes(h0, e0)
    boot_h, boot_e = [], []
    for _ in range(cfg["bootstrap"]):
        idx = [rng.integers(0, draws, draws) for _ in thetas]
        hb, eb = estimates(idx)
        s1, s2 = slopes(hb, eb)
        boot_h.append(s1)
        boot_e.append(s2)
    boot_h, boot_e = np.array(boot_h), np.array(boot_e)
    p = stencil.p
    hp = np.array([r["h_pred"] for r in per_theta]); ep = np.array([r["E_pred"] for r in per_theta])
    sh_pred, se_pred = slopes(hp, ep)
    result = {"stencil": stencil.name, "noise_model": noise_model, "function_orders": list(orders), "sectors": []}
    for k in range(3):
        ci_h = np.nanpercentile(boot_h[:, k], [2.5, 97.5]).tolist()
        ci_e = np.nanpercentile(boot_e[:, k], [2.5, 97.5]).tolist()
        result["sectors"].append({
            "k": k,
            "h_exponent_asymptotic": -k / (p + 2), "E_exponent_asymptotic": -k * p / (p + 2),
            "h_slope_pred_finite_theta": sh_pred[k], "E_slope_pred_finite_theta": se_pred[k],
            "h_slope_est": sh[k], "h_slope_ci95": ci_h,
            "h_contains_pred": bool(ci_h[0] <= sh_pred[k] <= ci_h[1]),
            "E_slope_est": se[k], "E_slope_ci95": ci_e,
            "E_contains_pred": bool(ci_e[0] <= se_pred[k] <= ci_e[1]),
            "hstar_fit": h0[:, k].tolist(), "hstar_pred_LO": hp[:, k].tolist(),
            "Estar_fit": e0[:, k].tolist(), "Estar_pred_LO": ep[:, k].tolist(),
        })
    result["theta"] = thetas.tolist()
    result["slope_theta_subset"] = thetas[sel].tolist()
    return result


# ---------------------------------------------------------------- Cor. 7.2 and crossover (exact MSE)
def cap_transition(cfg_mc, cfg_cap, rng):
    n = cfg_mc["n"]
    st = C.STENCILS["3pt"]
    f = C.make_test_function(cfg_mc["H"], [3, 4], n, rng)
    des = C.random_filled_design(cfg_mc["rays"], n - 1, rng)
    sigma, R, hmax = cfg_mc["sigma"], cfg_mc["R"], cfg_cap["h_max"]
    rows = []
    for th in cfg_cap["theta"]:
        hu, eu = optimal_h(f, des, th, st, sigma, R, 2)
        hc, ec = optimal_h(f, des, th, st, sigma, R, 2, h_max=hmax)
        rows.append({"theta": th, "h_unconstrained": hu, "E_unconstrained": math.sqrt(eu),
                     "h_capped": hc, "E_capped": math.sqrt(ec)})
    Dp = A.D_prime(des, st)[2]
    C2 = float(np.linalg.norm(C.sector_bias_formula(f.tensors[3], st.beta)["B2"]))
    theta_c = (2 * Dp * sigma**2 / (st.p * C2**2 * des.m * hmax ** (2 * st.p + 4))) ** (1 / 4)
    capped = [r for r in rows if r["theta"] < theta_c]
    slope = (float(np.polyfit(np.log([r["theta"] for r in capped]),
                              np.log([r["E_capped"] for r in capped]), 1)[0])
             if len(capped) >= 2 else None)
    return {"rows": rows, "theta_c_pred_LO": theta_c, "capped_slope": slope, "capped_slope_pred": -2.0}


def crossover(cfg_mc, cfg_x, rng):
    n = cfg_mc["n"]
    f = C.make_test_function(cfg_mc["H"], [3, 4], n, rng)
    des = C.random_filled_design(cfg_mc["rays"], n - 1, rng)
    normH = float(np.linalg.norm(f.H))
    out = []
    for sigma in cfg_x["sigma"]:
        for th in cfg_x["theta"]:
            res = {}
            for key in ("3pt", "4pt"):
                st = C.STENCILS[key]
                g = lambda lh: float(exact_mse(f, des, th, st, math.exp(lh), sigma, cfg_mc["R"]).sum())
                grid = np.linspace(math.log(1e-5), math.log(2.0), 120)
                i = int(np.argmin([g(x) for x in grid]))
                r = minimize_scalar(g, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 119)]), method="bounded")
                res[key] = math.sqrt(r.fun) / normH
            win = min(res, key=res.get)
            out.append({"sigma": sigma, "theta": th, "rel_err_3pt": res["3pt"], "rel_err_4pt": res["4pt"],
                        "winner": win, "winner_usable": res[win] <= cfg_x["usable_rel_frobenius"]})
    return out

"""Regenerate every Paper A validation result from config/validation.yaml.

Usage:
    python run_all.py            # full run (Monte Carlo + bootstrap + design search)
    python run_all.py --quick    # reduced Monte Carlo sizes, for a smoke test only

Outputs: results/*.json, results/metadata.json, figures/*.pdf
Only the 'regenerated' section of the config is executed.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sympy as sp
import yaml

from paperA_validation import common as C
from paperA_validation.experiments import accuracy as acc
from paperA_validation.experiments import bias as bias_exp
from paperA_validation.experiments import constants as K
from paperA_validation.impl_b import symbolic_bias as SB
from paperA_validation import figures

ROOT = Path(__file__).resolve().parent


def _json(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, sp.Basic):
        return str(obj)
    raise TypeError(type(obj))


def dump(name, obj):
    (ROOT / "results" / name).write_text(json.dumps(obj, indent=2, default=_json))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable (not a git checkout)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "validation.yaml").read_text())["regenerated"]
    seed = cfg["base_seed"]
    (ROOT / "results").mkdir(exist_ok=True)
    t0 = time.time()

    # ---------------- symbolic (implementation B)
    sym = {"stencils": {k: SB.stencil_constants([int(x) for x in s.weights], s.p)
                        for k, s in C.STENCILS.items()},
           "sector_bias": {}}
    for n in cfg["bias"]["n_values"]:
        for nu in (3, 4):
            r = SB.sector_bias_symbolic(n, nu)
            sym["sector_bias"][f"n{n}_nu{nu}"] = {
                "agree_with_lemma_6_2": r["agree"],
                "derived_c": str(r["derived"]["c"]), "derived_b": [str(x) for x in r["derived"]["b"]],
                "derived_T": [[str(x) for x in row] for row in r["derived"]["T"].tolist()],
                "mixed_factor_in_btilde_coordinates": str(r["mixed_tilde_factor"]),
            }
    dump("symbolic.json", sym)

    # ---------------- constants
    const = {"planar_limit_exact": {}, "ladders": {}, "intercepts": {}}
    for n in cfg["constants"]["n_values"]:
        d = n - 1
        const["planar_limit_exact"][f"n{n}"] = {
            "uniform_lambda_min_S0": str(K.uniform_ball_lambda_exact(d)),
            "uniform_predicted": str(K.predicted_uniform(d)),
            "kappa_theta2_uniform_limit": K.kappa_limit_uniform(n),
            "optimal_lambda_min_S0_numeric": K.S0_numeric(C.optimal_design(d)),
            "optimal_predicted": str(K.predicted_optimal(d)),
            "kappa_theta2_optimal_limit": K.kappa_limit_optimal(n),
        }
    const["axis_pentagon_S0_numeric"] = K.S0_numeric(C.axis_plus_pentagon())
    ladder_th = [0.2, 0.1, 0.05, 0.025, 0.0125]
    q = cfg["constants"]["quadrature"]
    designs = {
        "uniform_cap_n3": lambda th: C.uniform_cap_quadrature(th, q["n_polar"], q["n_az"]),
        "axis_pentagon_n3": lambda th: C.axis_plus_pentagon(),
        "optimal_realised_n3": lambda th: C.centre_plus_rim(2, 0.5, centre_copies=5),
        "optimal_n4": lambda th: C.optimal_design(3),
    }
    for key, fn in designs.items():
        rowsA = K.ladder(fn, ladder_th, "A")
        rowsB = K.ladder(fn, ladder_th, "B")
        const["ladders"][key] = rowsA
        const["ladders"][key + "_implB"] = rowsB
        const["intercepts"][key] = K.intercept_theta2(
            [r for r in rowsA if r["theta"] in cfg["constants"]["theta_ladder"]])
    ds_file = ROOT / "results" / "design_search.json"
    const["design_search"] = (json.loads(ds_file.read_text()) if ds_file.exists()
                              else "not run - execute run_design_search.py (checkpointed) first")
    dump("constants.json", const)

    # ---------------- bias
    b = {"numeric": {}, "parity": None}
    for n in cfg["bias"]["n_values"]:
        for key, st in C.STENCILS.items():
            h = cfg["bias"]["h"]["p1" if st.p == 1 else "p2"]
            b["numeric"][f"n{n}_{key}"] = bias_exp.numeric_bias_coefficients(
                n, st, cfg["bias"]["theta"], h, C.seed_for(seed, f"bias_{n}_{key}"))
    b["parity"] = bias_exp.parity_slopes(cfg["bias"]["parity_theta"], C.seed_for(seed, "parity"))
    dump("bias.json", b)

    # ---------------- Theorem 5.2 variance
    mc = dict(cfg["montecarlo"])
    if args.quick:
        mc["mc_draws"], mc["bootstrap"] = 100, 40
    rng = C.seed_for(seed, "variance")
    des = C.random_filled_design(mc["rays"], mc["n"] - 1, rng)
    vth = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    ndraw = 2000 if args.quick else 20000
    var = {nm: acc.variance_check(des, vth, C.STENCILS["3pt"], mc["sigma"], 1e-2, mc["R"], ndraw, rng,
                                  shared_base=(nm == "shared_base"))
           for nm in ("shared_base", "historical_independent")}
    dump("variance_check.json", var)

    # ---------------- Theorem 7.1 Monte Carlo + bootstrap
    runs = []
    for key in mc["stencils"]:
        st = C.STENCILS[key]
        for nm in mc["noise_models"]:
            runs.append(acc.montecarlo_exponents(mc, st, nm, C.seed_for(seed, f"mc_{key}_{nm}")))
    # robustness: 3-point stencil with an additional quartic term (O(h^{p+1}) remainder present)
    runs.append(acc.montecarlo_exponents(mc, C.STENCILS["3pt"], "shared_base",
                                         C.seed_for(seed, "mc_3pt_robust"), orders=[3, 4]))
    dump("montecarlo.json", {"runs": runs, "note": (
        "Primary runs use pure-order test functions (cubic for 3pt, quartic for 4pt) so that the "
        "leading-order model is exact in h; slopes are compared with the finite-theta leading-order "
        "prediction. The robustness run adds a quartic term to the 3pt case; its deviation reflects "
        "the O(h^{p+1}) remainder omitted in Theorem 7.1.")})

    # ---------------- Cor. 7.2 and crossover (exact MSE)
    dump("cap.json", acc.cap_transition(mc, cfg["cap"], C.seed_for(seed, "cap")))
    dump("crossover.json", acc.crossover(mc, cfg["crossover"], C.seed_for(seed, "crossover")))

    # ---------------- metadata + figures
    meta = {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__,
        "repository_commit": git_commit(), "base_seed": seed,
        "mode": "quick" if args.quick else "full",
        "runtime_seconds": round(time.time() - t0, 1),
        "seed_scheme": "numpy SeedSequence([base_seed] + ord(c) for c in experiment_name)",
    }
    dump("metadata.json", meta)
    figures.make_all(ROOT / "results", ROOT / "figures")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

"""Figures 3-6, generated only from archived result files in results/.

Figures 1 (geometry) and 2 (sector schematic) are conceptual diagrams and are drawn
separately; they contain no data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SECTOR = ["axial (k=0)", "mixed (k=1)", "tangential (k=2)"]


def _load(results: Path, name: str):
    return json.loads((results / name).read_text())


def fig3_conditioning(results: Path, out: Path):
    r = _load(results, "constants.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    styles = {"uniform_cap_n3": ("uniform cap (theorem)", math.sqrt(24), "-"),
              "axis_pentagon_n3": ("axis + pentagon (value for this design)", math.sqrt(14.4), "--"),
              "optimal_realised_n3": ("5 axial + pentagon, m=10 (realises optimum)", math.sqrt(8), ":")}
    for key, (lab, lim, ls) in styles.items():
        rows = r["ladders"][key]
        th = [x["theta"] for x in rows]
        ax.plot(th, [x["kappa_theta2"] for x in rows], "o" + ls, label=lab)
        ax.axhline(lim, color="grey", lw=0.6, ls=ls)
    ax.set_xscale("log")
    ax.set_xlabel(r"cone half-angle $\theta$")
    ax.set_ylabel(r"$\kappa_F(\theta)\,\theta^2$")
    ax.set_title(r"Scaled conditioning, $n=3$ (grey: limits $\sqrt{24},\sqrt{14.4},\sqrt{8}$)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig3_conditioning.pdf")
    plt.close(fig)


def fig4_noise(results: Path, out: Path):
    r = _load(results, "variance_check.json")["shared_base"]
    th = np.array([x["theta"] for x in r])
    mc = np.array([x["mc"] for x in r])
    tv = np.array([x["theory"] for x in r])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k in range(3):
        ax.plot(th, tv[:, k], "-", label=f"{SECTOR[k]}: Theorem 5.2")
        ax.plot(th, mc[:, k], "o", ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("sector error variance (fixed h)")
    ax.set_title("Noise amplification incl. shared-base floor (points: Monte Carlo)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4_noise.pdf")
    plt.close(fig)


def fig5_span(results: Path, out: Path):
    r = _load(results, "montecarlo.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for res in r["runs"]:
        if res["noise_model"] != "shared_base" or res["function_orders"] != [res_order(res)]:
            continue
        th = np.array(res["theta"])
        mark = "o" if res["stencil"] == "3pt" else "s"
        for s in res["sectors"][1:]:
            lab = f'{res["stencil"]}, {SECTOR[s["k"]]}'
            axes[0].plot(th, s["hstar_fit"], mark, ms=4, label=lab)
            axes[0].plot(th, s["hstar_pred_LO"], "-", lw=0.8, color=axes[0].lines[-1].get_color())
            axes[1].plot(th, s["Estar_fit"], mark, ms=4, label=lab)
            axes[1].plot(th, s["Estar_pred_LO"], "-", lw=0.8, color=axes[1].lines[-1].get_color())
    for ax, yl in zip(axes, [r"optimal span $h_k^\star$", r"optimal RMS error $E_k^\star$"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(yl)
        ax.legend(fontsize=6)
    fig.suptitle("Theorem 7.1: fitted Monte Carlo optimum (points) vs leading-order model (lines)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_span.pdf")
    plt.close(fig)


def res_order(res):
    return {"3pt": 3, "4pt": 4}[res["stencil"]]


def fig6_cap(results: Path, out: Path):
    r = _load(results, "cap.json")
    th = [x["theta"] for x in r["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(th, [x["h_unconstrained"] for x in r["rows"]], "o-", label="unconstrained optimum")
    axes[0].plot(th, [x["h_capped"] for x in r["rows"]], "s--", label=r"with $h\leq h_{\max}$")
    axes[1].plot(th, [x["E_unconstrained"] for x in r["rows"]], "o-", label="unconstrained")
    axes[1].plot(th, [x["E_capped"] for x in r["rows"]], "s--", label="capped")
    for ax, yl in zip(axes, [r"tangential $h^\star$", r"tangential RMS error"]):
        ax.axvline(r["theta_c_pred_LO"], color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(yl)
        ax.legend(fontsize=7)
    fig.suptitle(r"Corollary 7.2: span cap (dotted line: predicted $\theta_{c,2}$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig6_cap.pdf")
    plt.close(fig)


def make_all(results: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig3_conditioning(results, out)
    fig4_noise(results, out)
    fig5_span(results, out)
    fig6_cap(results, out)

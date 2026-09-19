"""Checkpointed design search (numerical best found; Cor. 4.5 remark).

Runs restarts in batches and resumes from results/design_search.json, so it can be
executed in pieces:  python run_design_search.py --batch 10   (repeat until 'complete')
"""
import argparse, json
from pathlib import Path

import numpy as np
import yaml

from paperA_validation import common as C
from paperA_validation.experiments import constants as K

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "design_search.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "validation.yaml").read_text())["regenerated"]
    ds = cfg["constants"]["design_search"]
    state = json.loads(OUT.read_text()) if OUT.exists() else {
        str(m): {"done": 0, "best": None} for m in ds["m"]}
    for m in ds["m"]:
        s = state[str(m)]
        todo = min(args.batch, ds["restarts"] - s["done"])
        if todo <= 0:
            continue
        # one reproducible stream per (m, batch start)
        rng = C.seed_for(cfg["base_seed"], f"design_search_m{m}_from{s['done']}")
        r = K.design_search(m, ds["theta"], todo, rng)
        if s["best"] is None or r["best_found_lmin_over_theta4"] > s["best"]["best_found_lmin_over_theta4"]:
            s["best"] = r
        s["done"] += todo
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(state, indent=2))
        print(f"m={m}: {s['done']}/{ds['restarts']} restarts, best {s['best']['best_found_lmin_over_theta4']:.8f}")
        return
    print("complete")


if __name__ == "__main__":
    main()

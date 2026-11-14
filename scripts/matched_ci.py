#!/usr/bin/env python3
"""Matched-LOPO table, CIs and paired bootstrap (round-2 M5).

Reads the full LOPO per-program APE table produced by ``lopo_table.py`` and
restricts it to the 30 held-out programs, which is exactly the matched-LOPO
protocol (remove one program, fit on the other 158, score the same 30).

Outputs ``doc/figures/matched_lopo_per_program.csv`` and prints, for both
feature sets, the mean APE with a bootstrap 95 % CI, plus the paired 18 vs 33
difference (5000 resamples, seed 0).
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOPO = os.path.join(ROOT, "doc", "figures", "lopo_per_program_rich.csv")
HELD = os.path.join(ROOT, "doc", "figures", "holdout_per_program.csv")
OUT = os.path.join(ROOT, "doc", "figures", "matched_lopo_per_program.csv")
N_BOOT, SEED = 5000, 0


def ci(values):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    held = {r["program"] for r in csv.DictReader(open(HELD, newline=""))}
    table = {}
    for r in csv.DictReader(open(LOPO, newline="")):
        if r["model"] == "gbdt" and r["program"] in held:
            table.setdefault(r["program"], {})[r["feats"]] = float(r["ape"])
    progs = sorted(p for p in table if "18" in table[p] and "33" in table[p])
    a = np.array([table[p]["18"] for p in progs])
    b = np.array([table[p]["33"] for p in progs])
    print(f"matched-LOPO, n={len(progs)}, bootstrap {N_BOOT}, seed {SEED}")
    for name, v in (("18 features", a), ("33 features", b)):
        lo, hi = ci(v)
        print(f"  {name:<12s} mean={v.mean():>6.2f}%  "
              f"95% CI [{lo:.2f}, {hi:.2f}]")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    diff = (a[idx] - b[idx]).mean(axis=1)
    print(f"  paired Δ (18-33) = {np.mean(a - b):>+6.2f} pp  "
          f"95% CI [{np.percentile(diff, 2.5):+.2f}, "
          f"{np.percentile(diff, 97.5):+.2f}]  "
          f"positive={np.mean(diff > 0):.4f}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "ape_lopo18", "ape_lopo33"])
        for i, p in enumerate(progs):
            w.writerow([p, a[i], b[i]])
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()

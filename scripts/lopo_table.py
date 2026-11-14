#!/usr/bin/env python3
"""Single-source LOPO table, confidence intervals, and error decomposition.

Round-1 revision (M3 / M6 / M9 / M10 / M11).  Every number in Table 6 of the
revised manuscript is produced here, on one data source, with one protocol:

  * LOPO over the programs actually present in the chosen data directory
    (159 for ``data/phase_ir_rich``), so the header n and the values agree.
  * per-program APE -> median, sMAPE, bootstrap 95% CI (5000 resamples,
    seed 0; round-2 M6 unified every bootstrap in the paper to this setting).
  * oracle de-bias: remove each program's own mean log error and recompute
    MAPE; log-log slope of predicted vs. true program cycles.
  * correlation between a program's mean phase-level log bias and its
    program-level log error (the error-decomposition diagnostic).
  * per-program CSV (``--dump-csv``) used by the alignment sensitivity
    analysis (M6) and by the DTW figure (M1).
  * GBDT permutation importance on the held-out split (M11 collinearity
    follow-up: importance is not read off collinear coefficients).

Usage:
  python3 -W ignore scripts/lopo_table.py --jobs 12 \
      --dump-csv doc/figures/lopo_per_program_rich.csv
  python3 -W ignore scripts/lopo_table.py --data data/phase_ir --features 18 \
      --jobs 12 --dump-csv doc/figures/lopo_per_program_base.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase import (FEATS_ALL, FEATS_RICH_ALL, build_rows, prep,
                           program_totals)
from holdout_eval import TEST

EPS = 1e-9
GBDT_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=0)

_DF = None
_FEATS = None
_MODEL = None


def _fit_one(prog):
    """Leave-one-program-out fit; returns (prog, pred_arm, mean log bias)."""
    tr = _DF[_DF["program"] != prog]
    te = _DF[_DF["program"] == prog]
    if _MODEL == "const":
        tot = tr.groupby("program").agg(x86=("x86_cycles", "sum"),
                                        arm=("arm_cycles", "sum"))
        r = float(np.mean(np.log10(tot["arm"]) - np.log10(tot["x86"])))
        pred = float(te["x86_cycles"].sum() * 10 ** r)
        bias = float(np.mean(r - te["y"].values))
        return prog, pred, bias
    Xtr, Xte = prep(tr, te, _FEATS)
    if _MODEL == "ridge":
        m = Ridge(alpha=1.0)
    elif _MODEL == "lasso":
        m = Lasso(alpha=0.001, max_iter=50000)
    else:
        m = GradientBoostingRegressor(**GBDT_CFG)
    m.fit(Xtr, tr["y"].values)
    yhat = m.predict(Xte)
    ratio = 10 ** yhat
    pred = float((te["x86_cycles"].values * ratio).sum())
    bias = float(np.mean(yhat - te["y"].values))
    return prog, pred, bias


def run_config(df, feats, model, jobs):
    global _DF, _FEATS, _MODEL
    _DF, _FEATS, _MODEL = df, feats, model
    progs = sorted(df["program"].unique())
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        out = pool.map(_fit_one, progs, chunksize=4)
    out.sort()
    return {p: (pr, b) for p, pr, b in out}


def bootstrap_ci(ape, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    ape = np.asarray(ape, dtype=float)
    idx = rng.integers(0, len(ape), size=(n_boot, len(ape)))
    means = ape[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(label, true, pred, bias, n_boot=5000):
    ape = np.abs(pred - true) / true * 100
    smape = np.mean(2 * np.abs(pred - true) /
                    (np.abs(true) + np.abs(pred))) * 100
    lo, hi = bootstrap_ci(ape, n_boot=n_boot)
    log_err = np.log10(pred) - np.log10(true)
    oracle = np.mean(np.abs(pred / 10 ** bias - true) / true * 100)
    slope = np.polyfit(np.log10(true), np.log10(pred), 1)[0]
    corr_bias = float(np.corrcoef(bias, log_err)[0, 1]) if len(true) > 2 else 0.0
    print(f"  {label:<34s} n={len(true):<4d} MAPE={ape.mean():>7.2f}% "
          f"[{lo:5.2f},{hi:5.2f}] med={np.median(ape):>6.2f}% "
          f"sMAPE={smape:>6.2f}% slope={slope:.3f} "
          f"oracle={oracle:5.2f}% biasCorr={corr_bias:+.3f}", flush=True)
    return ape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir_rich")
    ap.add_argument("--agg", choices=["base", "rich"], default="",
                    help="改用随包提供的预聚合表 data/phase_*.csv.gz, 免去 16 GB 采集")
    ap.add_argument("--exclude", nargs="*",
                    default=["multi_ncut", "llvm_paq8p_paq8p",
                             "llvm_doeproxyappsc_SimpleMOC",
                             "llvm_optimizereval_optimizer_eval"])
    ap.add_argument("--features", default="both", choices=["18", "33", "both"])
    ap.add_argument("--models", nargs="*",
                    default=["const", "ridge", "lasso", "gbdt"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dump-csv", default="")
    ap.add_argument("--perm-importance", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=5000,
                    help="per-program CI 的 bootstrap 次数 (默认 5000, seed=0)")
    args = ap.parse_args()
    t0 = time.time()

    if args.agg:
        from load_aggregated import load_aggregated
        df = load_aggregated(args.agg, exclude=args.exclude)
        src = f"data/phase_{args.agg}.csv.gz"
    else:
        df = build_rows(args.data, args.exclude)
        src = args.data
    have_rich = all(f in df.columns for f in FEATS_RICH_ALL)
    if args.features == "33" and not have_rich:
        sys.exit("该数据目录没有 33 特征列")
    featsets = []
    if args.features in ("18", "both"):
        featsets.append(("18", FEATS_ALL))
    if args.features in ("33", "both") and have_rich:
        featsets.append(("33", FEATS_RICH_ALL))
    names = sorted(df["program"].unique())
    tot = program_totals(df)
    true = np.array([tot.loc[p, "arm"] for p in names])
    print(f"数据 {src}: {len(names)} 程序, {len(df)} phases, "
          f"特征集 {[f for f, _ in featsets]}, jobs={args.jobs}\n")

    rows = []
    for fname, feats in featsets:
        for model in args.models:
            if model == "const" and fname != featsets[0][0]:
                continue
            res = run_config(df, feats, model, args.jobs)
            pred = np.array([res[p][0] for p in names])
            bias = np.array([res[p][1] for p in names])
            label = ("Constant baseline" if model == "const" else
                     f"{model.upper() if model != 'gbdt' else 'GBDT 300/4/0.05'}"
                     f" [{fname} feats]")
            ape = report(label, true, pred, bias, n_boot=args.bootstrap)
            for i, p in enumerate(names):
                rows.append(dict(program=p, model=model, feats=fname,
                                 true_arm=true[i], pred_arm=pred[i],
                                 ape=ape[i], bias_dex=bias[i]))

    if args.perm_importance and have_rich:
        print("\n置换重要性 (33 特征 GBDT, 训练=129, 评估=held-out 30 程序的 phase)")
        tr = df[~df["program"].isin(TEST)]
        te = df[df["program"].isin(TEST)]
        Xtr, Xte = prep(tr, te, FEATS_RICH_ALL)
        m = GradientBoostingRegressor(**GBDT_CFG).fit(Xtr, tr["y"].values)
        pi = permutation_importance(m, Xte, te["y"].values, n_repeats=10,
                                    random_state=0, n_jobs=args.jobs,
                                    scoring="neg_mean_absolute_error")
        order = np.argsort(pi.importances_mean)[::-1]
        for j in order:
            print(f"  {FEATS_RICH_ALL[j]:<16s} "
                  f"{pi.importances_mean[j]:+.5f} ± "
                  f"{pi.importances_std[j]:.5f}", flush=True)

    if args.dump_csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)),
                    exist_ok=True)
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n逐程序结果已写入 {args.dump_csv}")
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

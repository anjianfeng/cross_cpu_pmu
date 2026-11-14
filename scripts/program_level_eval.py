#!/usr/bin/env python3
"""Program-level baselines, epsilon-PCLSLR, and phase-level error decomposition.

Round-1 revision (M3 / M6 / M11). Three questions that the phase-level pipeline
alone cannot answer:

  1. Does a *program-level* model (aggregate host features -> program-level
     ARM/x86 ratio) already reach the phase-level accuracy?  If yes, the
     phase-level machinery adds little.
  2. Does an epsilon-neighbourhood constrained Lasso (LACross's PCLSLR, applied
     at the granularity at which neighbourhoods are defined over programs)
     beat the global model?  This is the reimplemented local method that the
     v1 manuscript never ran.
  3. How much of the phase-level error is a per-program level bias rather than
     phase-shape error (oracle de-bias, log-log slope)?  The per-program errors
     are also dumped to CSV for the alignment sensitivity analysis (M6).

Protocol: data/phase_ir_rich (159 programs) unless --data says otherwise.
Program-level features are the x86-cycle-weighted mean of the phase features;
program-level target is log10(sum ARM cycles / sum x86 cycles).

Usage:
  python3 -W ignore scripts/program_level_eval.py \
      --dump-lopo doc/figures/lopo_per_program_rich.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import (FEATS_ALL, FEATS_RICH, FEATS_RICH_ALL, build_rows,
                           mape, prep, program_totals)
from holdout_eval import TEST

EPS = 1e-9
GBDT_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=0)


def aggregate(df, feats):
    """x86-cycle-weighted per-program aggregate of phase features."""
    rows, names = [], []
    for p, g in df.groupby("program"):
        w = g["x86_cycles"].values.astype(float)
        w = w / (w.sum() + EPS)
        rows.append((g[feats].values.astype(float) * w[:, None]).sum(0))
        names.append(p)
    return np.array(rows), names


def transform(A, B, feats):
    """log10 for magnitudes, identity for bounded ratios, then standardize."""
    linear = {"f_prog", "f_taken"} | set(FEATS_RICH)
    A = A.copy()
    B = B.copy()
    for j, f in enumerate(feats):
        if f not in linear:
            A[:, j] = np.log10(np.abs(A[:, j]) + EPS)
            B[:, j] = np.log10(np.abs(B[:, j]) + EPS)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    return (A - mu) / sd, (B - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir_rich")
    ap.add_argument("--exclude", nargs="*",
                    default=["multi_ncut", "llvm_paq8p_paq8p",
                             "llvm_doeproxyappsc_SimpleMOC",
                             "llvm_optimizereval_optimizer_eval"])
    ap.add_argument("--dump-lopo", default="",
                    help="phase-level LOPO per-program CSV 输出路径")
    ap.add_argument("--lopo", action="store_true",
                    help="同时跑 phase-level LOPO 误差分解 (较慢)")
    args = ap.parse_args()
    t0 = time.time()

    df = build_rows(args.data, args.exclude, extra_counts=True)
    tot = program_totals(df)
    feats = FEATS_RICH_ALL if all(f in df.columns for f in FEATS_RICH) \
        else FEATS_ALL
    names = sorted(df["program"].unique())
    y = np.log10(tot.loc[names, "arm"].values /
                 tot.loc[names, "x86"].values)
    X_raw, agg_names = aggregate(df, feats)
    assert agg_names == names
    print(f"数据 {args.data}: {len(names)} 程序, {len(df)} phases, "
          f"{len(feats)} 特征 -> 程序级 {X_raw.shape}\n")

    te_idx = [i for i, p in enumerate(names) if p in TEST]
    tr_idx = [i for i, p in enumerate(names) if p not in TEST]

    def report(label, pred_ratio, idx=None):
        """Report one protocol.

        v3 (round-2 M1): the LOPO fits are scored on *all* programs first and
        the 30-program subset is reported on a separate line.  In v2 the two
        were conflated: the log printed n=30 while the manuscript labelled the
        row ``LOPO | 159``.  Both values are now printed explicitly.
        """
        if idx is None:                       # LOPO: 全量 159 个程序
            err = np.abs(10 ** pred_ratio - 10 ** y) / 10 ** y * 100
            print(f"  {label + ' (all)':<42s} n={len(err):<3d} "
                  f"MAPE={err.mean():>7.2f}%  med={np.median(err):>7.2f}%",
                  flush=True)
            if te_idx:
                sel = err[te_idx]             # 同一拟合, 只在 30 个 held-out 计分
                print(f"  {label + ' (held-out 30)':<42s} n={len(sel):<3d} "
                      f"MAPE={sel.mean():>7.2f}%  "
                      f"med={np.median(sel):>7.2f}%", flush=True)
            return err.mean()
        yy = 10 ** y[idx]                     # 仅测试集预测
        sel = np.abs(10 ** pred_ratio - yy) / yy * 100
        print(f"  {label:<42s} n={len(sel):<3d} MAPE={sel.mean():>7.2f}%  "
              f"med={np.median(sel):>7.2f}%", flush=True)
        return sel.mean()

    # ---------------- 1. program-level global baselines ----------------
    print("=== 1. Program-level baselines (aggregate x86 features) ===")
    for proto, tri, tei in (("LOPO(159)", None, None),
                            ("held-out(129->30)", tr_idx, te_idx)):
        if proto.startswith("LOPO"):
            preds = np.zeros(len(y))
            Xn = transform(X_raw, X_raw, feats)[0]
            for i in range(len(y)):
                m = np.ones(len(y), bool)
                m[i] = False
                rg = Ridge(alpha=1.0).fit(Xn[m], y[m])
                preds[i] = rg.predict(Xn[i:i + 1])[0]
            report(f"Ridge α=1, {proto} (program)", preds)
            preds = np.zeros(len(y))
            for i in range(len(y)):
                m = np.ones(len(y), bool)
                m[i] = False
                gb = GradientBoostingRegressor(**GBDT_CFG).fit(Xn[m], y[m])
                preds[i] = gb.predict(Xn[i:i + 1])[0]
            report(f"GBDT 300/4/0.05, {proto} (program)", preds)
        else:
            Xtr, Xte = transform(X_raw[tri], X_raw[tei], feats)
            for nm, mk in (("Ridge α=1",
                            lambda: Ridge(alpha=1.0)),
                           ("GBDT 300/4/0.05",
                            lambda: GradientBoostingRegressor(**GBDT_CFG))):
                m = mk().fit(Xtr, y[tri])
                report(f"{nm}, {proto} (program)", m.predict(Xte), idx=tei)

    # ---------------- 2. epsilon-neighbourhood PCLSLR ----------------
    print("\n=== 2. ε-neighbourhood PCLSLR (program level, 6 raw counters) ===")
    counts = ["c_cycles", "c_inst", "c_cref", "c_cmiss", "c_br", "c_brmiss"]
    if not all(c in df.columns for c in counts):
        print("  跳过: 数据缺少 c_* 绝对计数列")
    else:
        Xc, _ = aggregate(df, counts)
        Xc_n = (Xc - Xc.mean(0)) / (Xc.std(0) + EPS)
        # 程序级目标用绝对 ARM cycles, 与 CLSLR 一致
        y_abs = tot.loc[names, "arm"].values.astype(float) / 1e9
        for eps in (0.5, 1.0, 1.5, 2.0, 3.0):
            errs = []
            empty = 0
            # v3 (round-2 M2): true leave-one-program-out.  v2 looped over
            # ``tr_idx`` (129 programs) and computed distances on ``tr_idx`` so
            # that the query point i was always its own neighbour (d=0<=eps);
            # nnls then fitted y_abs[i] and predicted i (target leakage).
            # Now the loop covers all 159 programs and the candidate pool is
            # every program except i, both for the distance and for nnls.
            for i in range(len(names)):
                cand = [j for j in range(len(names)) if j != i]
                dist = np.sqrt(((Xc_n[cand] - Xc_n[i]) ** 2).sum(1))
                nb = [j for j, d in zip(cand, dist) if d <= eps]
                if len(nb) < 2 * len(counts):
                    empty += 1
                    nb = list(np.array(cand)[np.argsort(dist)[:2 * len(counts)]])
                s = Xc[nb].mean(0) + EPS
                w, _ = nnls(Xc[nb] / s, y_abs[nb], maxiter=10000)
                pred = (Xc[i] / s) @ w * 1e9
                errs.append(abs(pred - tot.loc[names[i], "arm"])
                            / tot.loc[names[i], "arm"] * 100)
            print(f"  ε={eps:<4.1f} MAPE={np.mean(errs):>7.2f}%  "
                  f"med={np.median(errs):>7.2f}%  "
                  f"(邻域过小回退 {empty}/{len(names)})", flush=True)

    # ---------------- 3. phase-level error decomposition ----------------
    if args.lopo:
        print("\n=== 3. Phase-level LOPO 误差分解 (33 特征, seed=0) ===")
        rows = []
        preds = {}
        for i, p in enumerate(names):
            tr = df[df["program"] != p]
            te = df[df["program"] == p]
            Xtr, Xte = prep(tr, te, feats)
            gb = GradientBoostingRegressor(**GBDT_CFG).fit(Xtr, tr["y"].values)
            ratio = 10 ** gb.predict(Xte)
            preds[p] = float((te["x86_cycles"].values * ratio).sum())
            if (i + 1) % 20 == 0:
                print(f"    LOPO {i+1}/{len(names)}", flush=True)
        true = np.array([tot.loc[p, "arm"] for p in names])
        pred = np.array([preds[p] for p in names])
        ape = np.abs(pred - true) / true * 100
        log_err = np.log10(pred) - np.log10(true)
        bias = log_err.mean()
        oracle = np.mean(np.abs(pred / 10 ** bias - true) / true * 100)
        slope = np.polyfit(np.log10(true), np.log10(pred), 1)[0]
        print(f"  phase-level LOPO MAPE={ape.mean():.2f}%  "
              f"med={np.median(ape):.2f}%")
        print(f"  共同偏置={bias:+.4f} dex; oracle 去偏后 MAPE={oracle:.2f}%; "
              f"log-log slope={slope:.3f}")
        for p, t, pr, e in zip(names, true, pred, ape):
            rows.append(dict(program=p, true_arm=t, pred_arm=pr, ape=e))
        if args.dump_lopo:
            os.makedirs(os.path.dirname(os.path.abspath(args.dump_lopo)),
                        exist_ok=True)
            with open(args.dump_lopo, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  逐程序 LOPO 结果已写入 {args.dump_lopo}")

    print(f"\n总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-program DTW vs ratio alignment experiment (round-1 revision, M1/M6).

Produces the data behind Figure 5 and Table 9 without any synthesised point:
for every program, the shape correlation between the resampled x86 and ARM
cycle trajectories under each alignment, the leave-one-program-out MAPE under
the DTW alignment, and (optionally) the ratio-alignment MAPE read from the CSV
written by ``lopo_table.py``.  The scatter in Figure 5 is drawn directly from
the CSV written here.

The ratio-alignment MAPE is *not* recomputed when ``--mape-csv`` is given: the
same LOPO protocol is reused, so the ratio and DTW columns of Table 9 come from
one model configuration and one program set.

Usage:
  python3 -W ignore scripts/dtw_per_program.py --data data/phase_ir_rich \
      --rich --jobs 12 --mape-csv doc/figures/lopo_per_program_rich.csv \
      --dump-csv doc/figures/dtw_per_program_rich.csv
  python3 -W ignore scripts/dtw_per_program.py --data data/phase_ir --jobs 12 \
      --mape-csv doc/figures/lopo_per_program_base.csv \
      --dump-csv doc/figures/dtw_per_program_base.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase import (FEATS_ALL, FEATS_RICH_ALL, build_rows, prep,
                           program_totals)

EPS = 1e-9
CFG_BASE = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0)
CFG_RICH = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=0)

_DF = None
_FEATS = None
_CFG = None


def _fit_one(prog):
    tr = _DF[_DF["program"] != prog]
    te = _DF[_DF["program"] == prog]
    Xtr, Xte = prep(tr, te, _FEATS)
    gb = GradientBoostingRegressor(**_CFG).fit(Xtr, tr["y"].values)
    pred = float((te["x86_cycles"].values * 10 ** gb.predict(Xte)).sum())
    true = float(te["arm_cycles"].sum())
    return prog, abs(pred - true) / true * 100


def shape_corr(df):
    """程序内 x86/ARM cycles 轨迹的形状相关 (各自归一化到和为 1)."""
    out = {}
    for p, g in df.groupby("program"):
        x = g["x86_cycles"].values.astype(float)
        a = g["arm_cycles"].values.astype(float)
        x = x / (x.sum() + EPS)
        a = a / (a.sum() + EPS)
        out[p] = float(np.corrcoef(x, a)[0, 1])
    return out


def lopo_mape(df, feats, cfg, jobs):
    global _DF, _FEATS, _CFG
    _DF, _FEATS, _CFG = df, feats, cfg
    progs = sorted(df["program"].unique())
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        out = pool.map(_fit_one, progs, chunksize=4)
    return dict(out)


def read_mape_csv(path, feats="33", model="gbdt"):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r["feats"] == feats:
                out[r["program"]] = float(r["ape"])
    return out


def stats(corr, err):
    neg = [p for p in corr if corr[p] < 0]
    pos = [p for p in corr if corr[p] >= 0]
    return dict(n=len(corr), n_neg=len(neg), n_pos=len(pos),
                mean=float(np.mean(list(corr.values()))),
                n_gt07=sum(1 for v in corr.values() if v > 0.7),
                n_lt07=sum(1 for v in corr.values() if v <= 0.7),
                mape_neg=float(np.mean([err[p] for p in neg])) if neg else float("nan"),
                mape_pos=float(np.mean([err[p] for p in pos])) if pos else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir_rich")
    ap.add_argument("--exclude", nargs="*",
                    default=["multi_ncut", "llvm_paq8p_paq8p",
                             "llvm_doeproxyappsc_SimpleMOC",
                             "llvm_optimizereval_optimizer_eval"])
    ap.add_argument("--rich", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mape-csv", default="",
                    help="lopo_table.py 的逐程序 CSV (复用 ratio 对齐 MAPE)")
    ap.add_argument("--dump-csv", default="")
    args = ap.parse_args()
    t0 = time.time()

    feats = FEATS_ALL
    cfg = CFG_BASE
    if args.rich:
        feats, cfg = FEATS_RICH_ALL, CFG_RICH

    print("构建 ratio 对齐数据 ...", flush=True)
    df_r = build_rows(args.data, args.exclude, align="ratio")
    print("构建 DTW 对齐数据 ...", flush=True)
    df_d = build_rows(args.data, args.exclude, align="dtw")

    corr_r = shape_corr(df_r)
    corr_d = shape_corr(df_d)
    print(f"ratio: mean corr={np.mean(list(corr_r.values())):.3f}, "
          f"负相关 {sum(v < 0 for v in corr_r.values())}, "
          f">0.7 {sum(v > 0.7 for v in corr_r.values())}")
    print(f"dtw  : mean corr={np.mean(list(corr_d.values())):.3f}, "
          f"负相关 {sum(v < 0 for v in corr_d.values())}, "
          f">0.7 {sum(v > 0.7 for v in corr_d.values())}")

    if args.mape_csv:
        err_r = read_mape_csv(args.mape_csv,
                              feats="33" if args.rich else "18")
        print(f"ratio MAPE 读取自 {args.mape_csv} ({len(err_r)} 程序)")
    else:
        print("LOPO (ratio) ...", flush=True)
        err_r = lopo_mape(df_r, feats, cfg, args.jobs)

    print("LOPO (dtw) ...", flush=True)
    err_d = lopo_mape(df_d, feats, cfg, args.jobs)

    print(f"\nLOPO: ratio={np.mean(list(err_r.values())):.2f}%  "
          f"dtw={np.mean(list(err_d.values())):.2f}%  "
          f"delta={np.mean(list(err_d.values())) - np.mean(list(err_r.values())):+.2f}")
    print(f"MAPE 下降的程序: {sum(err_d[p] < err_r[p] for p in err_r)}/"
          f"{len(err_r)}")

    for tag, corr, err in (("ratio", corr_r, err_r), ("dtw", corr_d, err_d)):
        s = stats(corr, err)
        print(f"[{tag}] 负相关 n={s['n_neg']}/{s['n']} MAPE={s['mape_neg']:.2f}%; "
              f"非负 n={s['n_pos']} MAPE={s['mape_pos']:.2f}%; "
              f"剔除负相关后 MAPE={s['mape_pos']:.2f}%")

    rows = []
    for p in sorted(corr_r):
        rows.append(dict(program=p, corr_ratio=corr_r[p], corr_dtw=corr_d[p],
                         mape_ratio=err_r[p], mape_dtw=err_d[p],
                         delta=err_d[p] - err_r[p]))
    if args.dump_csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)),
                    exist_ok=True)
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"逐程序结果已写入 {args.dump_csv}")
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

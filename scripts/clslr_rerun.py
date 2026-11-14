#!/usr/bin/env python3
"""M2: 重新实现 LACross 原始 CLSLR 形式 (raw counter 特征, 绝对周期目标, theta>=0, 无截距).

背景: 论文 4.3 节曾把 "no finite solution" 归因于单调性假设被破坏, 这在优化理论上
不成立 (非负约束最小二乘是凸问题, 可行域非空)。本脚本按原式重跑, 报告:
  1) 有限但较差的 LOPO / held-out MAPE;
  2) 无约束解中负系数的个数 -- 单调性诊断;
  3) 共线性诊断 (VIF / 条件数), 说明负系数的解释边界;
  4) 6 个 raw counter 上的符号分析 (与原 CLSLR 特征集一致)。

v2 (round-1 revision, E1) 关键修复
----------------------------------
v1 版脚本直接对 1e8--1e10 量级的绝对计数列调用 ``scipy.optimize.nnls``。NNLS 默认
``maxiter = 3 * n_features``, 列未缩放时条件数很大, 迭代预算不足, 于是输出退化解,
得到 LOPO 19766%/held-out 19083% 的伪结果 (见 /tmp/clslr.log, 用 ``--legacy`` 复现)。
``--legacy`` 之外, 本脚本显式给出 ``maxiter=10000`` 并按训练折列均值做列缩放
(等价重参数化, 不改变模型类)。``data/phase_ir`` 上 LOPO=21.92%, held-out=20.45%,
与稿件 v1 表格逐位一致; ``data/phase_ir_rich`` 上为 19.05% / 16.39%。

用法:
  python3 -W ignore scripts/clslr_rerun.py --data data/phase_ir
  python3 -W ignore scripts/clslr_rerun.py --data data/phase_ir --legacy   # 复现旧日志
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import build_rows, mape
from holdout_eval import TEST

FEATS = ["f_ipc", "f_cref", "f_cmiss", "f_br", "f_brmiss",
         "f_cmiss_rate", "f_brmiss_rate", "f_prog",
         "f_s_ld", "f_s_st", "f_s_alu", "f_s_fp", "f_s_vec",
         "f_s_div", "f_s_call", "f_s_br", "f_bb_size", "f_taken"]
# 原始 CLSLR 用的是绝对计数 (perf 事件数), 不是速率
FEATS_COUNT = ["c_cycles", "c_inst", "c_cref", "c_cmiss", "c_br", "c_brmiss"]
ALPHAS = [1e-8, 1e-6]
SCALE = 1e9          # y 归一到 1e9 量级, 避免坐标下降收敛过慢
MAXITER = 10000      # NNLS 迭代预算; --legacy 时退回 scipy 默认 (3*n_features)


def col_scale(Xtr, Xte, on=True):
    """按训练折列均值缩放 (等价重参数化); 关闭则复现 v1 的数值退化."""
    if not on:
        return Xtr, Xte
    s = Xtr.mean(0) + 1e-12
    return Xtr / s, Xte / s


def score(err, label):
    err = np.asarray(err)
    print(f"  {label:<42s} MAPE={err.mean():>7.2f}%  med={np.median(err):>7.2f}%")
    return err.mean()


def vif_report(df):
    """18 个结构特征上的 VIF / 条件数 (标准化空间)."""
    X = df[FEATS].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cond = float(np.linalg.cond(X))
    vifs = []
    for j in range(X.shape[1]):
        A = np.delete(X, j, axis=1)
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ coef
        r2 = 1 - resid.var() / (X[:, j].var() + 1e-12)
        vifs.append(1.0 / max(1 - r2, 1e-9))
    order = np.argsort(vifs)[::-1][:5]
    print(f"  共线性诊断 (n={len(df)} phases, 18 特征): 条件数={cond:.1f}")
    print("  VIF 最高的 5 个特征: " +
          ", ".join(f"{FEATS[i]}={vifs[i]:.1f}" for i in order))


def run_lopo(df, scaled=True):
    progs = sorted(df["program"].unique())
    err = {a: [] for a in ALPHAS}
    err_nnls = []
    neg_lasso, neg_ols, neg_raw = [], [], []
    for i, p in enumerate(progs):
        if i % 20 == 0:
            print(f"    LOPO fold {i}/{len(progs)}", flush=True)
        tr = df[df["program"] != p]
        te = df[df["program"] == p]
        Xtr = tr[FEATS_COUNT].values.astype(float)
        ytr = tr["arm_cycles"].values.astype(float) / SCALE
        Xte = te[FEATS_COUNT].values.astype(float)
        Xtr, Xte = col_scale(Xtr, Xte, scaled)
        true = te["arm_cycles"].sum()

        w, _ = nnls(Xtr, ytr, maxiter=MAXITER)
        err_nnls.append(abs((Xte @ w * SCALE).sum() - true) / true * 100)
        for a in ALPHAS:
            m = Lasso(alpha=a, positive=True, fit_intercept=False,
                      max_iter=5000, tol=1e-4).fit(Xtr, ytr)
            err[a].append(abs((m.predict(Xte) * SCALE).sum() - true)
                          / true * 100)
        coef, *_ = np.linalg.lstsq(tr[FEATS].values.astype(float), ytr,
                                   rcond=None)
        neg_ols.append(int((coef < 0).sum()))
        wraw, *_ = np.linalg.lstsq(
            tr[FEATS_COUNT].values.astype(float), ytr, rcond=None)
        neg_raw.append(int((wraw < 0).sum()))
    print(f"\nLOPO ({len(progs)} 程序, {len(df)} phases, "
          f"scaled={scaled})")
    score(err_nnls, "CLSLR-original (NNLS, theta>=0, no intercept)")
    for a in ALPHAS:
        score(err[a], f"CLSLR-original (Lasso alpha={a:g}, positive)")
    print(f"  无约束 OLS 负系数 (18 结构特征): {np.mean(neg_ols):.1f} / "
          f"{len(FEATS)}  (min {min(neg_ols)}, max {max(neg_ols)})")
    print(f"  无约束 OLS 负系数 (CLSLR 的 6 个 raw counter): "
          f"{np.mean(neg_raw):.1f} / {len(FEATS_COUNT)}")


def run_heldout(df, scaled=True):
    progs = sorted(df["program"].unique())
    tr_progs = [p for p in progs if p not in TEST]
    te_progs = [p for p in progs if p in TEST]
    tr = df[df["program"].isin(tr_progs)]
    te = df[df["program"].isin(te_progs)]
    Xtr = tr[FEATS_COUNT].values.astype(float)
    ytr = tr["arm_cycles"].values.astype(float) / SCALE
    Xte = te[FEATS_COUNT].values.astype(float)
    Xtr, Xte = col_scale(Xtr, Xte, scaled)
    prog = te["program"].values
    print(f"\nHeld-out: 训练 {len(tr_progs)} / 测试 {len(te_progs)} 程序")

    def errs(pred_phase, label):
        e = []
        for p in te_progs:
            m = prog == p
            true = te.loc[m, "arm_cycles"].sum()
            e.append(abs(pred_phase[m].sum() * SCALE - true) / true * 100)
        score(e, label)

    w, _ = nnls(Xtr, ytr, maxiter=MAXITER)
    errs(Xte @ w, "CLSLR-original (NNLS, theta>=0, no intercept)")
    for a in ALPHAS:
        m = Lasso(alpha=a, positive=True, fit_intercept=False,
                  max_iter=5000, tol=1e-4).fit(Xtr, ytr)
        errs(m.predict(Xte), f"CLSLR-original (Lasso alpha={a:g}, positive)")
    coef, *_ = np.linalg.lstsq(tr[FEATS].values.astype(float), ytr, rcond=None)
    print(f"  无约束 OLS 负系数 (18 结构特征): {int((coef < 0).sum())} / "
          f"{len(FEATS)}")
    wraw, *_ = np.linalg.lstsq(tr[FEATS_COUNT].values.astype(float), ytr,
                               rcond=None)
    print(f"  无约束 OLS 负系数 (6 raw counter): {int((wraw < 0).sum())} / "
          f"{len(FEATS_COUNT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--exclude", nargs="*",
                    default=["multi_ncut", "llvm_paq8p_paq8p"])
    ap.add_argument("--unscaled", action="store_true",
                    help="关闭列缩放 (配合 --legacy 复现 v1 的数值退化)")
    ap.add_argument("--legacy", action="store_true",
                    help="退回 scipy 默认 maxiter=3*n_features, 复现 19766/19083")
    args = ap.parse_args()
    global MAXITER
    if args.legacy:
        MAXITER = None
    t0 = time.time()
    df = build_rows(args.data, args.exclude, extra_counts=True)
    miss = [f for f in FEATS + FEATS_COUNT if f not in df.columns]
    if miss:
        sys.exit(f"缺少特征列: {miss}")
    print(f"数据 {args.data}: {df['program'].nunique()} 程序, "
          f"{len(df)} phases")
    vif_report(df)
    run_lopo(df, scaled=not args.unscaled)
    run_heldout(df, scaled=not args.unscaled)
    print(f"\n总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

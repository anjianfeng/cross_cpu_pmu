#!/usr/bin/env python3
"""域偏移应对策略对照：重要性加权 / OOD 门控 / k-center 覆盖选择。

两个跨域场景（训练池与测试集分布不同）：
  S1 异质→同质 : 训练 = 异质 131, 测试 = PolyBench 30
  S2 held-out  : 训练 = PolyBench30+LLVM101, 测试 = MiBench+SD-VBS+SPEC 30

每个场景对比：
  1 常数基线
  2 GBDT baseline
  3 GBDT + 密度比重要性加权 (IW)
  4 GBDT + OOD 门控（与常数基线按程序级距离混合）
  5 GBDT + k-center 覆盖选择（只训练被选中的子集）
  6 GBDT + k-center + IW
  7 GBDT + 随机同规模子集（k-center 的对照）

用法:
  python3 -W ignore scripts/domain_adapt.py --exclude multi_ncut llvm_paq8p_paq8p
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import (LINEAR_FEATS, build_rows, mape, program_totals)

EPS = 1e-9
HOMOG = set("""2mm 3mm adi atax bicg cholesky correlation covariance deriche
doitgen durbin fdtd-2d floyd-warshall gemm gemver gesummv gramschmidt heat-3d
jacobi-1d jacobi-2d lu ludcmp mvt nussinov seidel-2d symm syr2k syrk trisolv
trmm""".split())
# held-out 测试集 = MiBench + SD-VBS + SPEC
HOLDOUT = set("""adpcm_decode adpcm_encode aes basicmath bitcount blowfish crc
dijkstra fft gsm patricia qsort sha stringsearch susan
disparity localization mser multi_ncut sift stitch svm texture_synthesis tracking
spec_deepsjeng spec_gcc spec_lbm spec_leela spec_mcf spec_nab spec_xz""".split())


def tf(df, feats):
    """与 analyze_phase.prep 相同的单边变换（对数化非线性特征）。"""
    X = df[feats].values.astype(float).copy()
    for j, f in enumerate(feats):
        if f not in LINEAR_FEATS:
            X[:, j] = np.log10(np.abs(X[:, j]) + EPS)
    return X


def prog_matrix(df, feats):
    """程序级行为指纹：phase 特征均值（去掉无信息的 f_prog）。"""
    use = [f for f in feats if f != "f_prog"]
    tmp = df.copy()
    X = tf(tmp, use)
    tmp[use] = X
    M = tmp.groupby("program")[use].mean()
    M = (M - M.mean()) / (M.std() + EPS)
    return M


def const_ratio(df, train_progs):
    tot = program_totals(df)
    tr = tot.loc[train_progs]
    r = (np.log10(tr["arm"]) - np.log10(tr["x86"])).mean()
    return lambda prog: float(tot.loc[prog, "x86"] * 10 ** r)


def kcenter_select(cand_progs, targ_progs, M, k):
    """贪心集合覆盖：选 k 个候选，使目标程序到最近候选的距离和最小。"""
    Zc = M.loc[cand_progs].values
    Zt = M.loc[targ_progs].values
    D = cdist(Zt, Zc)                      # (nt, nc)
    covered = np.full(len(Zt), np.inf)
    chosen = []
    for _ in range(min(k, len(cand_progs))):
        gain = np.minimum(covered[:, None], D).sum(0)
        c = int(np.argmin(gain))
        chosen.append(c)
        covered = np.minimum(covered, D[:, c])
    return [cand_progs[i] for i in chosen]


def predict_model(df, test_progs, train_progs, feats, factory,
                  iw=False, gate=False, gate_scale=1.0, M=None):
    """在 train_progs 上训练，返回 {prog: 预测 cycles}。"""
    tr = df[df["program"].isin(train_progs)]
    Xtr = tf(tr, feats)
    mu, sd = Xtr.mean(0), Xtr.std(0) + EPS
    Xtr_n = (Xtr - mu) / sd
    ytr = tr["y"].values

    w = None
    if iw:
        # 密度比 p_test/p_train，用逻辑回归在 phase 特征空间判别
        te = df[df["program"].isin(test_progs)]
        Xte_raw = tf(te, feats)
        Xa = np.vstack([Xtr_n, (Xte_raw - mu) / sd])
        lab = np.r_[np.zeros(len(Xtr_n)), np.ones(len(Xte_raw))]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xa, lab)
        p = clf.predict_proba(Xtr_n)[:, 1]
        w = np.clip(p / (1 - p + EPS), 0.1, 10.0)
        w = w / w.mean()

    model = factory().fit(Xtr_n, ytr, sample_weight=w)

    # 程序级距离（用于门控）：训练池内部最近邻距离的稳健尺度
    alpha = None
    if gate:
        Ztr = M.loc[train_progs].values
        Dtr = cdist(Ztr, Ztr)
        np.fill_diagonal(Dtr, np.inf)
        d0 = np.median(Dtr.min(1)) + EPS
        Zte = M.loc[test_progs].values
        dt = cdist(Zte, Ztr).min(1)
        alpha = {p: float(np.exp(-d / (gate_scale * d0)))
                 for p, d in zip(test_progs, dt)}

    const = const_ratio(df, train_progs)
    preds = {}
    for prog in test_progs:
        te = df[df["program"] == prog]
        Xte_n = (tf(te, feats) - mu) / sd
        ratio = 10 ** model.predict(Xte_n)
        p_model = float((te["x86_cycles"].values * ratio).sum())
        if gate:
            a = alpha[prog]
            p_model = a * p_model + (1 - a) * const(prog)
        preds[prog] = p_model
    return preds


def score(preds, test_progs, tot):
    names = [p for p in test_progs if p in preds]
    true = tot.loc[names, "arm"].values
    pred = np.array([preds[p] for p in names])
    err = np.abs(pred - true) / true * 100
    return mape(true, pred), np.median(err)


def run_scenario(df, feats, M, tot, title, test_progs, pool_progs):
    print(f"\n=== {title} ===")
    print(f"测试 {len(test_progs)} 程序 / 训练池 {len(pool_progs)} 程序")

    fac = lambda: GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)

    const = const_ratio(df, pool_progs)
    m, med = score({p: const(p) for p in test_progs}, test_progs, tot)
    print(f"  {'常数基线':<28s} MAPE={m:>6.2f}%  中位={med:>6.2f}%")

    def go(label, **kw):
        t0 = time.time()
        pr = predict_model(df, test_progs, kw.pop("train_progs", pool_progs),
                           feats, fac, M=M, **kw)
        m, med = score(pr, test_progs, tot)
        print(f"  {label:<28s} MAPE={m:>6.2f}%  中位={med:>6.2f}%  "
              f"({time.time()-t0:.0f}s)")
        return m, pr

    go("GBDT baseline")
    go("GBDT + IW 重要性加权", iw=True)
    for lam in (0.5, 1.0, 2.0, 4.0):
        go(f"GBDT + OOD 门控 λ={lam}", gate=True, gate_scale=lam)

    k = min(30, len(pool_progs))
    sel = kcenter_select(pool_progs, test_progs, M, k)
    go(f"GBDT + k-center({k})", train_progs=sel)
    go(f"GBDT + k-center({k}) + IW", train_progs=sel, iw=True)

    # 随机子集多 seed，确认不是单次抽样偶然
    ms = []
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.RandomState(seed)
        rand = list(rng.choice(pool_progs, size=k, replace=False))
        m, _ = go(f"GBDT + 随机{k} seed={seed}", train_progs=rand)
        ms.append(m)
    print(f"  {'随机子集均值':<28s} MAPE={np.mean(ms):>6.2f}%  "
          f"±{np.std(ms):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--feats", default="all", choices=["all", "base"])
    args = ap.parse_args()

    from analyze_phase import FEATS_ALL, FEATS_BASE
    feats = FEATS_ALL if args.feats == "all" else FEATS_BASE

    t0 = time.time()
    df = build_rows(args.data, args.exclude)
    df = df[~df["program"].isin(args.exclude)]
    progs = sorted(df["program"].unique())
    tot = program_totals(df)
    M = prog_matrix(df, feats)
    print(f"数据 {args.data}: {len(progs)} 程序, {len(df)} phases, "
          f"{len(feats)} 特征")

    poly = [p for p in progs if p in HOMOG]
    hetero = [p for p in progs if p not in HOMOG]
    holdout = [p for p in progs if p in HOLDOUT]
    pool2 = [p for p in progs if p not in HOLDOUT]

    run_scenario(df, feats, M, tot, "S1 异质→同质 (训练异质131, 测试PolyBench30)",
                 poly, hetero)
    run_scenario(df, feats, M, tot,
                 "S2 held-out (训练PolyBench+LLVM, 测试MiBench+SDVBS+SPEC)",
                 holdout, pool2)

    print(f"\n总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

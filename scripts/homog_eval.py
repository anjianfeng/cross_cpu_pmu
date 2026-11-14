#!/usr/bin/env python3
"""训练集同质性对照实验。

固定测试程序集合，只更换训练集来源，量化「训练集同质性」对 MAPE 的贡献：
  A 跨套件训练 : 训练=异质子集, 测试=同质子集 (held-out, 模拟论文训练/测试形态差异)
  B 同质训练   : 训练=同质子集留一 (训练/测试同分布, 模拟论文 ICPC 内部 CV)
  C 全量训练   : 训练=全部其余程序 (当前主实验口径)
  D 同质迁移   : 训练=同质子集, 测试=异质程序 (反向极端)

同质子集 = PolyBench 30 个数值内核（代码骨架高度同构：循环嵌套 + 数组访存）。
异质子集 = 其余 131 个（MiBench/SD-VBS/SPEC/LLVM）。

用法:
  python3 -W ignore scripts/homog_eval.py \
      --exclude multi_ncut llvm_paq8p_paq8p
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import (build_rows, FEATS_ALL, mape, prep,
                           program_totals)

# PolyBench/C 4.2.1 数值内核（本数据集中实际配对的 30 个）
HOMOG = set("""2mm 3mm adi atax bicg cholesky correlation covariance deriche
doitgen durbin fdtd-2d floyd-warshall gemm gemver gesummv gramschmidt heat-3d
jacobi-1d jacobi-2d lu ludcmp mvt nussinov seidel-2d symm syr2k syrk trisolv
trmm""".split())


def eval_trainset(df, test_progs, train_pool, feats, factory):
    """在指定训练池上留一拟合并预测 test_progs 的整程序 cycles。"""
    preds = {}
    for prog in test_progs:
        pool = [p for p in train_pool if p != prog]
        if not pool:
            continue
        tr = df[df["program"].isin(pool)]
        te = df[df["program"] == prog]
        Xtr, Xte = prep(tr, te, feats)
        m = factory().fit(Xtr, tr["y"].values)
        ratio = 10 ** m.predict(Xte)
        preds[prog] = float((te["x86_cycles"].values * ratio).sum())
    return preds


def const_trainset(df, test_progs, train_pool):
    """同一训练池上的常数 log 比值基线。"""
    tot = program_totals(df)
    preds = {}
    for prog in test_progs:
        pool = [p for p in train_pool if p != prog]
        sub = tot.loc[pool]
        r = (np.log10(sub["arm"]) - np.log10(sub["x86"])).mean()
        preds[prog] = float(tot.loc[prog, "x86"] * 10 ** r)
    return preds


def report(label, preds, test_progs, tot, train_pool, ntest):
    names = [p for p in test_progs if p in preds]
    true = tot.loc[names, "arm"].values
    pred = np.array([preds[p] for p in names])
    err = np.abs(pred - true) / true * 100
    tr = tot.loc[[p for p in train_pool if p != names[0]]]
    # 训练池比值离散度 vs 测试池比值离散度
    tr_r = np.log10(tr["arm"].values / tr["x86"].values)
    te_r = np.log10(tot.loc[names, "arm"].values / tot.loc[names, "x86"].values)
    print(f"  {label:<22s} MAPE={mape(true, pred):>6.2f}%  "
          f"中位={np.median(err):>6.2f}%  "
          f"train σ(log ratio)={tr_r.std():.3f}  test σ={te_r.std():.3f}")
    return mape(true, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--no-gbdt", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="只跑指定前缀（A-F），默认全部")
    args = ap.parse_args()

    t0 = time.time()
    df = build_rows(args.data, args.exclude)
    df = df[~df["program"].isin(args.exclude)]
    progs = sorted(df["program"].unique())
    homog = [p for p in progs if p in HOMOG]
    hetero = [p for p in progs if p not in HOMOG]
    tot = program_totals(df)
    print(f"数据 {args.data}: {len(progs)} 程序 "
          f"(同质 PolyBench {len(homog)} / 异质 {len(hetero)}), "
          f"{len(df)} phases, {len(FEATS_ALL)} 特征\n")

    models = [("常数基线", None),
              ("Lasso α=0.001", lambda: Lasso(alpha=0.001, max_iter=20000))]
    if not args.no_gbdt:
        models.append(("GBDT 150/3", lambda: GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42)))

    # 场景: (名称, 测试集, 训练池)
    scenarios = [
        ("A 跨套件训练", homog, hetero),
        ("B 同质训练", homog, homog),
        ("C 全量训练", homog, progs),
        ("D 同质迁移异质", hetero, homog),
        ("E 异质内部训练", hetero, hetero),
        ("F 全量训练", hetero, progs),
    ]

    for tname, test_set, train_pool in scenarios:
        if args.scenarios and tname[0] not in args.scenarios:
            continue
        print(f"测试集={tname} (n={len(test_set)}), 训练池 n={len(train_pool)}")
        for label, fac in models:
            t1 = time.time()
            if fac is None:
                pr = const_trainset(df, test_set, train_pool)
            else:
                pr = eval_trainset(df, test_set, train_pool, FEATS_ALL, fac)
            report(f"{label}", pr, test_set, tot, train_pool, len(test_set))
            if fac is not None:
                print(f"  {'':22s} 用时 {time.time()-t1:.0f}s", flush=True)
        print()

    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

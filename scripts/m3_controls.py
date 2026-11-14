#!/usr/bin/env python3
"""M3 受控对照: 验证 top-down 特征的 "跨 suite 迁移" 论断.

审稿人指出原文用两个不同测试集上的 Δ 推断迁移性, 至少有四个替代解释:
  (1) 测试集难度不同 (LOPO 161 vs held-out 30)
  (2) held-out 训练集 LLVM 占 77%, 增益可能来自特征多样性/正则化
  (3) LOPO 中同 suite 邻居已含冗余信息
  (4) 只有一折, 可能由个别 suite 主导

本脚本给出五组对照:
  A) 同一 30 个 held-out 程序上的 2x2: 协议 (LOPO / held-out) x 特征 (18 / 33)
  B) leave-one-suite-out 五折, 报均值±标准差
  C) rich15 拆分: top-down 8 个比例 vs cache/mem 7 个速率
  D) 安慰剂: 18 特征 + 15 个随机置换特征
  E) 训练集构成: 仅 LLVM(101) / 均衡 (LLVM 30 + PolyBench 30)

用法:
  python3 -u -W ignore scripts/m3_controls.py                # 全部
  python3 -u -W ignore scripts/m3_controls.py --blocks A B   # 指定组
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase import (FEATS_ALL, FEATS_RICH, build_rows, mape, prep,
                           program_totals)
from holdout_eval import MIBENCH, POLYBENCH, SDVBS, SPEC, TEST

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "doc", "figures")
N_BOOT = 5000          # round-2 M5/M6: one resampling count and seed everywhere
SEED = 0

TOP_DOWN = FEATS_RICH[:8]      # 8 个 top-down 瓶颈比例
CACHE_MEM = FEATS_RICH[8:]     # 7 个 cache/TLB/store 速率
FEATS33 = FEATS_ALL + FEATS_RICH


def gbdt():
    return GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=0)


class Ctx:
    """数据与真值容器, 显式传递以免依赖全局变量."""

    def __init__(self, df):
        self.df = df
        tot = program_totals(df)
        self.arm = tot["arm"].to_dict()
        self.x86 = tot["x86"].to_dict()
        self.progs = sorted(df["program"].unique())

    def truth(self, progs):
        return np.array([self.arm[p] for p in progs])

    @staticmethod
    def to_program(te, pred_phase):
        """phase 比值预测 → 程序绝对 ARM cycles."""
        ratio = 10 ** pred_phase
        progs = te["program"].values
        x86 = te["x86_cycles"].values
        out = {}
        for p in np.unique(progs):
            m = progs == p
            out[p] = float((x86[m] * ratio[m]).sum())
        return out

    def evaluate(self, train_df, test_progs, feats, label="", return_ape=False):
        te = self.df[self.df["program"].isin(test_progs)]
        Xtr, Xte = prep(train_df, te, feats)
        model = gbdt().fit(Xtr, train_df["y"].values)
        pred = self.to_program(te, model.predict(Xte))
        arr = np.array([pred[p] for p in test_progs])
        m = mape(self.truth(test_progs), arr)
        ape = np.array([abs(pred[p] - self.arm[p]) / self.arm[p] * 100
                        for p in test_progs])
        if label:
            print(f"  {label:<44s} n={len(test_progs):<3d} "
                  f"MAPE={m:>7.2f}%", flush=True)
        return (m, ape) if return_ape else m

    def const_baseline(self, train_df, test_progs, label=""):
        tot = program_totals(train_df)
        r = (np.log10(tot["arm"]) - np.log10(tot["x86"])).mean()
        arr = np.array([self.x86[p] * 10 ** r for p in test_progs])
        m = mape(self.truth(test_progs), arr)
        if label:
            print(f"  {label:<44s} n={len(test_progs):<3d} "
                  f"MAPE={m:>7.2f}%", flush=True)
        return m

    def lopo(self, test_progs, feats, label="", return_ape=False):
        """逐程序留一, 误差只在给定程序集合上统计 (与 held-out 同测试集)."""
        errs = []
        for i, p in enumerate(test_progs, 1):
            tr = self.df[self.df["program"] != p]
            te = self.df[self.df["program"] == p]
            Xtr, Xte = prep(tr, te, feats)
            model = gbdt().fit(Xtr, tr["y"].values)
            pr = self.to_program(te, model.predict(Xte))[p]
            errs.append(abs(pr - self.arm[p]) / self.arm[p] * 100)
            if i % 10 == 0 or i == len(test_progs):
                print(f"    ... LOPO {label} {i}/{len(test_progs)}", flush=True)
        m = float(np.mean(errs))
        if label:
            print(f"  {label:<44s} n={len(test_progs):<3d} "
                  f"MAPE={m:>7.2f}%", flush=True)
        return (m, np.array(errs)) if return_ape else m


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """按程序配对的 bootstrap: 返回 mean(a-b), 95% CI, 正值比例.

    ``a`` 与 ``b`` 是同一批程序的逐程序 APE (a=18 特征, b=33 特征),
    正差值表示 33 特征误差更低。round-2 M5/M6 统一使用
    n_boot=5000, seed=0, 并在日志中写出这三个值。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return (float(np.mean(a - b)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d > 0)))


def report_paired(label, a, b):
    d, lo, hi, frac = paired_bootstrap(a, b)
    print(f"  Δ vs base [{label:<26s}] = {d:>+6.2f} pp  "
          f"95% CI [{lo:>+6.2f}, {hi:>+6.2f}]  positive {frac:.4f}  "
          f"(n_boot={N_BOOT}, seed={SEED})", flush=True)
    return dict(delta=d, lo=lo, hi=hi, frac=frac)


def matched_protocol(ctx, test_progs):
    """A) 同一测试集上的 2x2: 协议 x 特征, 逐程序 APE + 配对 CI."""
    print("\n=== A. 同测试集 2x2 (30 个 held-out 程序) ===", flush=True)
    tr = ctx.df[~ctx.df["program"].isin(TEST)]
    out, apes = {}, {}
    ctx.const_baseline(tr, test_progs, "常数基线 (held-out 训练集)")
    for feats, fs in ((FEATS_ALL, "18"), (FEATS33, "33")):
        out[("held-out", fs)], apes[("held-out", fs)] = ctx.evaluate(
            tr, test_progs, feats, f"held-out, {fs} 特征", return_ape=True)
    for feats, fs in ((FEATS_ALL, "18"), (FEATS33, "33")):
        out[("LOPO(30)", fs)], apes[("LOPO(30)", fs)] = ctx.lopo(
            test_progs, feats, f"LOPO, {fs} 特征 (同 30 程序)", return_ape=True)
    print("  配对 bootstrap (APE18 - APE33, 正值=33 特征更好):", flush=True)
    ho = report_paired("held-out 30", apes[("held-out", "18")],
                       apes[("held-out", "33")])
    ml = report_paired("matched-LOPO 30", apes[("LOPO(30)", "18")],
                       apes[("LOPO(30)", "33")])
    path = os.path.join(FIG, "matched_lopo_per_program.csv")
    os.makedirs(FIG, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "ape_heldout18", "ape_heldout33",
                    "ape_lopo18", "ape_lopo33"])
        for i, p in enumerate(test_progs):
            w.writerow([p, apes[("held-out", "18")][i],
                        apes[("held-out", "33")][i],
                        apes[("LOPO(30)", "18")][i],
                        apes[("LOPO(30)", "33")][i]])
    print(f"  逐程序 APE 已写入 {os.path.relpath(path)}", flush=True)
    return out, ho, ml


def loso(ctx):
    """B) leave-one-suite-out 五折, 附逐折配对 CI (round-2 M7)."""
    print("\n=== B. Leave-one-suite-out 五折 ===", flush=True)
    suites = [("PolyBench", POLYBENCH), ("MiBench", MIBENCH),
              ("SD-VBS", SDVBS), ("SPEC", SPEC)]
    llvm = set(ctx.progs) - POLYBENCH - MIBENCH - SDVBS - SPEC
    suites.append(("LLVM", llvm))
    print(f"  折内训练集规模: LLVM-only 折 n={len(llvm)} "
          f"(v2 正文/图误写为 101)", flush=True)
    res = {"18": [], "33": []}
    rows = []
    for sname, pool in suites:
        te_progs = sorted(p for p in ctx.progs if p in pool)
        tr = ctx.df[~ctx.df["program"].isin(te_progs)]
        ape = {}
        for feats, fs in ((FEATS_ALL, "18"), (FEATS33, "33")):
            m, ape[fs] = ctx.evaluate(
                tr, te_progs, feats, f"test={sname}, {fs} 特征",
                return_ape=True)
            res[fs].append(m)
        st = report_paired(f"LOSO fold {sname}", ape["18"], ape["33"])
        print(f"  fold {sname:<9s} n={len(te_progs):<3d} "
              f"18={res['18'][-1]:>6.2f}%  33={res['33'][-1]:>6.2f}%  "
              f"Δ={st['delta']:>+5.2f} CI=[{st['lo']:>+5.2f},{st['hi']:>+5.2f}]",
              flush=True)
        for i, p in enumerate(te_progs):
            rows.append([p, sname, ape["18"][i], ape["33"][i]])
    for fs in ("18", "33"):
        a = np.array(res[fs])
        print(f"  {fs} 特征: mean={a.mean():.2f}%  sd={a.std(ddof=1):.2f}%  "
              f"min={a.min():.2f}  max={a.max():.2f}", flush=True)
    print(f"  Δ(折均值) = "
          f"{np.mean(res['18']) - np.mean(res['33']):+.2f} pp", flush=True)
    path = os.path.join(FIG, "loso_per_program.csv")
    os.makedirs(FIG, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "fold", "ape18", "ape33"])
        w.writerows(rows)
    print(f"  逐折逐程序 APE 已写入 {os.path.relpath(path)}", flush=True)


def ablation(ctx, n_perm=1):
    """C) rich15 拆分 + D) 安慰剂, 附配对 CI (round-2 M5 / round-3 M7).

    ``n_perm`` > 1 时对 rich 15 列做多次独立置换 (seeds 0..n_perm-1),
    报告 placebo MAPE 的均值/min/max/2.5-97.5 分位; Table 10 的 placebo
    行使用逐程序 APE 在置换上的均值, 其配对 CI 仍是程序级而非置换级。
    """
    print("\n=== C/D. 特征拆分与安慰剂 (held-out) ===", flush=True)
    tr = ctx.df[~ctx.df["program"].isin(TEST)]
    test_progs = sorted(p for p in ctx.progs if p in TEST)
    apes = {}
    for feats, label in ((FEATS_ALL, "base 18"),
                         (FEATS_ALL + TOP_DOWN, "+ top-down 8"),
                         (FEATS_ALL + CACHE_MEM, "+ cache/mem 7"),
                         (FEATS33, "+ all rich 15")):
        m, apes[label] = ctx.evaluate(tr, test_progs, feats, label,
                                      return_ape=True)
    draws, ape_draws = [], []
    for pi in range(n_perm):
        rng = np.random.default_rng(pi)
        d = ctx.df.copy()
        for c in FEATS_RICH:
            d[c] = rng.permutation(d[c].values)
        placebo = Ctx(d)
        tag = (f"placebo draw {pi} (seed {pi})" if n_perm > 1
               else "placebo: 15 permuted features")
        m, ape = placebo.evaluate(
            d[~d["program"].isin(TEST)], test_progs, FEATS33, tag,
            return_ape=True)
        draws.append(m)
        ape_draws.append(ape)
    if n_perm > 1:
        a = np.array(draws)
        print(f"  placebo {n_perm} 次置换: mean={a.mean():.2f}%  "
              f"min={a.min():.2f}%  max={a.max():.2f}%  "
              f"p2.5={np.percentile(a, 2.5):.2f}%  "
              f"p97.5={np.percentile(a, 97.5):.2f}%  "
              f"sd={a.std(ddof=1):.2f}%", flush=True)
        apes["placebo"] = np.mean(np.vstack(ape_draws), axis=0)
        print(f"  placebo 逐程序 APE 的置换均值 MAPE={apes['placebo'].mean():.2f}%",
              flush=True)
    else:
        apes["placebo"] = ape_draws[0]
    print("  配对 bootstrap (APE18 - APE变体, 正值=变体更好):", flush=True)
    for label in ("+ top-down 8", "+ cache/mem 7", "+ all rich 15", "placebo"):
        report_paired(label, apes["base 18"], apes[label])


def _suite_counts(pool):
    """按套件统计池内程序数 (round-3 M2)."""
    return dict(PolyBench=sum(p in POLYBENCH for p in pool),
                MiBench=sum(p in MIBENCH for p in pool),
                SDVBS=sum(p in SDVBS for p in pool),
                SPEC=sum(p in SPEC for p in pool),
                LLVM=sum(p not in (POLYBENCH | MIBENCH | SDVBS | SPEC)
                         for p in pool))


def composition(ctx):
    """E) 训练集构成对照."""
    print("\n=== E. 训练集构成 (held-out 协议) ===", flush=True)
    test_progs = sorted(p for p in ctx.progs if p in TEST)
    llvm = [p for p in ctx.progs if p not in TEST and p not in POLYBENCH]
    poly = [p for p in ctx.progs if p in POLYBENCH]
    rng = np.random.default_rng(0)
    bal = sorted(rng.choice(llvm, size=min(30, len(llvm)), replace=False)) + poly
    # round-2 M7: the LLVM-only pool is 99 programs, not 101.
    for name, pool in ((f"LLVM {len(llvm)}", llvm), ("balanced 60", bal)):
        c = _suite_counts(pool)
        print(f"  {name}: n={len(pool):<3d} "
              f"PolyBench {c['PolyBench']}, MiBench {c['MiBench']}, "
              f"SD-VBS {c['SDVBS']}, SPEC {c['SPEC']}, LLVM {c['LLVM']}",
              flush=True)
        if name.startswith("balanced"):
            print(f"  balanced: PolyBench {c['PolyBench']}, MiBench {c['MiBench']}, "
                  f"SD-VBS {c['SDVBS']}, SPEC {c['SPEC']}, LLVM {c['LLVM']}",
                  flush=True)
        tr = ctx.df[ctx.df["program"].isin(pool)]
        for feats, fs in ((FEATS_ALL, "18"), (FEATS33, "33")):
            ctx.evaluate(tr, test_progs, feats, f"train={name}, {fs} 特征")


BLOCKS = {"B": loso, "C": ablation, "E": composition}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir_rich")
    ap.add_argument("--exclude", nargs="*",
                    default=["multi_ncut", "llvm_paq8p_paq8p"])
    ap.add_argument("--blocks", nargs="*", default=["A", "B", "C", "E"],
                    help="要运行的组, 如 A B C E (C 同时含 D 安慰剂)")
    ap.add_argument("--placebo-perms", type=int, default=1,
                    help="D 组独立置换次数 (seeds 0..N-1); round-3 M7")
    args = ap.parse_args()
    t0 = time.time()
    df = build_rows(args.data, args.exclude)
    miss = [f for f in FEATS33 if f not in df.columns]
    if miss:
        sys.exit(f"缺少特征列: {miss}")
    ctx = Ctx(df)
    test_progs = sorted(p for p in ctx.progs if p in TEST)
    print(f"数据: {len(ctx.progs)} 程序, {len(df)} phases, "
          f"测试 {len(test_progs)} 程序", flush=True)
    for b in args.blocks:
        key = b.upper()
        if key == "A":
            matched_protocol(ctx, test_progs)
        elif key == "C":
            ablation(ctx, n_perm=args.placebo_perms)
        elif key in BLOCKS:
            BLOCKS[key](ctx)
        else:
            sys.exit(f"未知组: {b}")
    print(f"\n总用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

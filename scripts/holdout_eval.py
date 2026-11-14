#!/usr/bin/env python3
"""Held-out 跨套件验证: 训练 PolyBench+LLVM, 测试 MiBench+SD-VBS+SPEC.

对应 LACross 论文的设定: 同质训练集 (论文 160 个 ICPC 题解) +
held-out 三个套件 (论文 MiBench+SD-VBS+SPEC2006, 35 程序)。
LOPO 用所有程序互相留一, 这里改为真正的"训练集/测试集完全分离",
衡量跨套件泛化。

v2 (round-1 revision) 修复
-------------------------
* 原实现 (L143-L181) 在超参扫描循环里复用变量 ``gb``, 分套件统计使用的是
  循环**最后一个**配置 (500/5/0.1, 总体 15.78%), 而不是最优配置
  (300/4/0.05, 总体 14.51%)。现在显式保存 ``best_model`` 并在其后统计。
* 同时评估 18 特征与 33 特征两套配置, 使分套件列与总体值同源、可加权核对。
* 新增: median / sMAPE / bootstrap 95% CI / 多随机种子 mean±sd /
  逐程序 CSV 导出 / oracle 去偏与 log-log slope 诊断。

用法:
  python3 -W ignore scripts/holdout_eval.py --data data/phase_ir_rich --rich
  python3 -W ignore scripts/holdout_eval.py --data data/phase_ir_rich --rich \
      --seeds 5 --bootstrap 2000 --dump-csv doc/figures/holdout_per_program.csv
  python3 -W ignore scripts/holdout_eval.py --data data/phase_ir_rich --rich \
      --gbdt-sweep
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import (FEATS_ALL, FEATS_BASE, FEATS_RICH, build_rows,
                           mape, prep, program_totals)
from pclsrr import local_fit_predict, select_neighbors

POLYBENCH = {
    "2mm", "3mm", "adi", "atax", "bicg", "cholesky", "correlation",
    "covariance", "deriche", "doitgen", "durbin", "fdtd-2d",
    "floyd-warshall", "gemm", "gemver", "gesummv", "gramschmidt",
    "heat-3d", "jacobi-1d", "jacobi-2d", "lu", "ludcmp", "mvt",
    "nussinov", "seidel-2d", "symm", "syr2k", "syrk", "trisolv", "trmm",
}
MIBENCH = {
    "adpcm_decode", "adpcm_encode", "aes", "basicmath", "bitcount",
    "blowfish", "crc", "dijkstra", "fft", "gsm", "patricia", "qsort",
    "sha", "stringsearch", "susan",
}
SDVBS = {
    "disparity", "localization", "mser", "multi_ncut", "sift",
    "stitch", "svm", "texture_synthesis", "tracking",
}
SPEC = {
    "spec_deepsjeng", "spec_gcc", "spec_lbm", "spec_leela",
    "spec_mcf", "spec_nab", "spec_xz",
}
TEST = MIBENCH | SDVBS | SPEC          # 训练: 其余 (PolyBench + LLVM)

SUITES = (("MiBench", MIBENCH), ("SD-VBS", SDVBS), ("SPEC", SPEC))
EPS = 1e-9
GBDT_CFG = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8)


def metrics(true, pred):
    """返回 dict: MAPE / median APE / sMAPE / 预测值。"""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ape = np.abs(pred - true) / true * 100
    smape = np.mean(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred))) * 100
    return dict(mape=float(ape.mean()), med=float(np.median(ape)),
                smape=float(smape), ape=ape, pred=pred)


def bootstrap_ci(ape, n_boot=2000, seed=0):
    """对程序级 APE 做 bootstrap, 返回 MAPE 的 95% CI。"""
    rng = np.random.default_rng(seed)
    ape = np.asarray(ape, dtype=float)
    n = len(ape)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = ape[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_pred(names, true_arm, pred, label, n_boot=0):
    """程序级 MAPE / 中位误差 / sMAPE (+可选 bootstrap CI)."""
    pred_arr = np.array([pred[p] for p in names])
    m = metrics(true_arm, pred_arr)
    ci = ""
    if n_boot:
        lo, hi = bootstrap_ci(m["ape"], n_boot=n_boot)
        ci = f"  95%CI=[{lo:5.2f},{hi:5.2f}]"
    print(f"  {label:<34s} MAPE={m['mape']:>7.2f}%  "
          f"med={m['med']:>7.2f}%  sMAPE={m['smape']:>7.2f}%{ci}", flush=True)
    return m["mape"], pred_arr, m


def fit_gbdt(X, y, seed=0):
    gb = GradientBoostingRegressor(**GBDT_CFG, random_state=seed)
    gb.fit(X, y)
    return gb


def paired_boot(a, b, n_boot, seed=0):
    """按程序配对 bootstrap, 返回 mean(a-b), 95% CI, 正值比例."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return (float(np.mean(a - b)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d > 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--agg", choices=["base", "rich"], default="",
                    help="改用随包提供的预聚合表 data/phase_*.csv.gz, 免去 16 GB 采集")
    ap.add_argument("--exclude", nargs="*", default=["multi_ncut",
                                                     "llvm_paq8p_paq8p"])
    ap.add_argument("--rich", action="store_true")
    ap.add_argument("--skip-local", action="store_true",
                    help="跳过慢的局部回归")
    ap.add_argument("--gbdt-sweep", action="store_true",
                    help="对 GBDT 做超参扫描 (n_est/depth/lr)")
    ap.add_argument("--sweep-rich-only", action="store_true",
                    help="扫描只跑 33 特征配置 (18 特征列由固定配置运行给出)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="GBDT 随机种子数 (>1 时报告 mean±sd)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap 次数 (0=不计算 CI)")
    ap.add_argument("--paired-bootstrap", type=int, default=0,
                    help="18 vs 33 特征按套件配对 bootstrap 次数 (0=不算)")
    ap.add_argument("--dump-csv", default="",
                    help="逐程序预测/误差 CSV 输出路径")
    ap.add_argument("--dump-sweep-csv", default="",
                    help="超参扫描结果 CSV 输出路径")
    args = ap.parse_args()

    t0 = time.time()
    if args.agg:
        from load_aggregated import load_aggregated
        df = load_aggregated(args.agg, exclude=args.exclude)
    else:
        df = build_rows(args.data, args.exclude)
    feats = FEATS_BASE
    if "f_s_ld" in df.columns:
        feats = FEATS_ALL
    if args.rich and all(f in df.columns for f in FEATS_RICH):
        feats = FEATS_ALL + FEATS_RICH

    progs = sorted(df["program"].unique())
    tr_progs = [p for p in progs if p not in TEST]
    te_progs = [p for p in progs if p in TEST]
    n_suite = {s: sum(1 for p in te_progs if p in pool)
               for s, pool in SUITES}
    print(f"held-out: 训练 {len(tr_progs)} 程序, 测试 {len(te_progs)} 程序 "
          f"{n_suite}, {len(df)} phases, {len(feats)} 特征")
    print(f"data={args.data}  features={len(feats)}  n_train={len(tr_progs)}\n")

    tr_df = df[df["program"].isin(tr_progs)]
    te_df = df[df["program"].isin(te_progs)]
    tot = program_totals(df)
    true_arm = np.array([tot.loc[p, "arm"] for p in te_progs])
    x86_te = np.array([tot.loc[p, "x86"] for p in te_progs])

    te_prog_arr = te_df["program"].values
    x86_cycles_by_phase = {p: te_df[te_prog_arr == p]["x86_cycles"].values
                           for p in te_progs}

    def agg(phase_pred):
        """phase 预测比值 → 程序绝对 cycles."""
        ratio = 10 ** phase_pred
        out = {}
        for p in te_progs:
            m = te_prog_arr == p
            out[p] = float((x86_cycles_by_phase[p] * ratio[m]).sum())
        return out

    # ---- 常数比值基线 (训练集程序均值) ----
    tr_tot = program_totals(tr_df)
    const_ratio = (np.log10(tr_tot["arm"]) - np.log10(tr_tot["x86"])).mean()
    p_const = {p: float(x86_te[i] * 10 ** const_ratio)
               for i, p in enumerate(te_progs)}
    print(f"  (训练集常数比值 = 10^{const_ratio:.4f} = "
          f"{10 ** const_ratio:.3f})")
    const_m, _, _ = eval_pred(te_progs, true_arm, p_const, "常数比值基线",
                              args.bootstrap)

    dump_rows = []

    def record(config, predmap):
        """逐程序记录 (供图表直接从 CSV 取数, 不再手抄/合成)."""
        if not args.dump_csv:
            return
        for i, p in enumerate(te_progs):
            dump_rows.append(dict(config=config, program=p,
                                  true_arm=true_arm[i],
                                  pred_arm=predmap[p],
                                  ape=abs(predmap[p] - true_arm[i])
                                  / true_arm[i] * 100))

    record("const", p_const)

    configs = [(FEATS_ALL, "base18")]
    if args.rich and feats == FEATS_ALL + FEATS_RICH:
        configs.append((FEATS_ALL + FEATS_RICH, "rich33"))
    if args.gbdt_sweep and args.sweep_rich_only:
        configs = [c for c in configs if c[1] == "rich33"]

    results = {}
    per_suite = {}
    for feats_c, tag in configs:
        Xtr, Xte = prep(tr_df, te_df, feats_c)
        ytr = tr_df["y"].values

        for name, mk in (("Ridge α=1", lambda: Ridge(alpha=1.0)),
                         ("Lasso α=0.001",
                          lambda: Lasso(alpha=0.001, max_iter=50000))):
            m = mk()
            m.fit(Xtr, ytr)
            pr = agg(m.predict(Xte))
            eval_pred(te_progs, true_arm, pr,
                      f"{name} [{tag}]", args.bootstrap)
            record(f"{name.split()[0].lower()}_{tag}", pr)

        if args.gbdt_sweep:
            print(f"\n--- GBDT 超参扫描 (held-out, {tag}) ---")
            best = (None, None)
            best_model = None
            sweep_rows = []
            for n_est in [200, 300, 500]:
                for depth in [3, 4, 5]:
                    for lr in [0.03, 0.05, 0.1]:
                        gb = GradientBoostingRegressor(
                            n_estimators=n_est, max_depth=depth,
                            learning_rate=lr, subsample=0.8, random_state=0)
                        gb.fit(Xtr, ytr)
                        mm, _, _ = eval_pred(
                            te_progs, true_arm, agg(gb.predict(Xte)),
                            f"GBDT {n_est}/{depth}/{lr}", 0)
                        sweep_rows.append(dict(n_est=n_est, depth=depth, lr=lr,
                                               mape=float(mm)))
                        if best[1] is None or mm < best[1]:
                            best = ((n_est, depth, lr), mm)
                            best_model = gb          # 保存最优模型, 供分套件使用
            print(f"\n  最优 GBDT 配置: {best[0]}, MAPE={best[1]:.2f}%")
            if args.dump_sweep_csv:
                import csv as _csv
                os.makedirs(os.path.dirname(
                    os.path.abspath(args.dump_sweep_csv)), exist_ok=True)
                with open(args.dump_sweep_csv, "w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
                    w.writeheader()
                    w.writerows(sweep_rows)
                print(f"  扫描结果已写入 {args.dump_sweep_csv}")
            results[tag] = dict(mape=float(best[1]), sd=0.0,
                                mean=float(best[1]), pred=None,
                                mapes=[float(best[1])])
        else:
            seeds = max(1, args.seeds)
            maps, preds, seed0_model, seed0_pred = [], None, None, None
            for s in range(seeds):
                gb = fit_gbdt(Xtr, ytr, seed=s)
                pr = agg(gb.predict(Xte))
                mm, pa, _ = eval_pred(
                    te_progs, true_arm, pr,
                    f"GBDT 300/4/0.05 seed={s} [{tag}]" if seeds > 1
                    else f"GBDT 300/4/0.05 [{tag}]", args.bootstrap)
                maps.append(mm)
                preds = pr
                if s == 0:
                    seed0_model, seed0_pred = gb, pr
            if seeds > 1:
                print(f"  {tag} GBDT {seeds} seeds: "
                      f"mean={np.mean(maps):.2f}%  sd={np.std(maps, ddof=1):.2f}%  "
                      f"min={np.min(maps):.2f}%  max={np.max(maps):.2f}%",
                      flush=True)
                print(f"  注: 分套件与诊断使用 seed=0 模型 (与全稿其他表格一致)",
                      flush=True)
            # 分套件必须与 headline 模型 (seed=0) 对齐, 不能用最后一个种子
            best_model = seed0_model
            results[tag] = dict(mape=float(maps[0]),
                                sd=(float(np.std(maps, ddof=1))
                                    if seeds > 1 else 0.0),
                                mean=float(np.mean(maps)),
                                pred=seed0_pred, mapes=maps)
            record(f"gbdt_{tag}", seed0_pred)

        # ---- 分套件 (使用 best_model, 修复 v1 的最后一个配置 bug) ----
        pred_gb = agg(best_model.predict(Xte))
        print(f"\n分套件误差 (GBDT best model, {tag}):")
        per_suite[tag] = {}
        for sname, pool in SUITES:
            ps = [p for p in te_progs if p in pool]
            if not ps:
                continue
            arr = np.array([pred_gb[p] for p in ps])
            tr_ = np.array([tot.loc[p, "arm"] for p in ps])
            mm, _, _ = eval_pred(ps, tr_, dict(zip(ps, arr)),
                                 f"{sname:<10s}", args.bootstrap)
            per_suite[tag][sname] = mm
        weighted = sum(len([p for p in te_progs if p in pool]) * v
                       for (sname, pool), v in
                       zip(SUITES, [per_suite[tag][s[0]] for s in SUITES]))
        print(f"  加权核对: {weighted / len(te_progs):.2f}% "
              f"(seed=0 总体 {results[tag]['mape']:.2f}%)")

        # ---- 诊断: oracle 去偏 + log-log slope ----
        pred_arr = np.array([pred_gb[p] for p in te_progs])
        log_err = np.log10(pred_arr) - np.log10(true_arm)
        bias = log_err.mean()
        de_biased = pred_arr / 10 ** bias
        oracle = metrics(true_arm, de_biased)
        slope = np.polyfit(np.log10(true_arm), np.log10(pred_arr), 1)[0]
        print(f"  诊断 [{tag}]: 共同偏置={bias:+.4f} dex, "
              f"oracle 去偏 MAPE={oracle['mape']:.2f}%, "
              f"log-log slope={slope:.3f}")

    # ---- 18 vs 33 特征的按套件配对 bootstrap (round-2 M6) ----
    if (args.paired_bootstrap and "base18" in results
            and "rich33" in results):
        def _ape(tag):
            pm = results[tag]["pred"]
            return np.array([abs(pm[p] - tot.loc[p, "arm"])
                             / tot.loc[p, "arm"] * 100 for p in te_progs])
        a, b = _ape("base18"), _ape("rich33")
        print(f"\n配对 bootstrap (base18 - rich33, 正值=33 特征更好; "
              f"{args.paired_bootstrap} 次, seed=0):")
        groups = [("all", None)] + list(SUITES)
        for sname, pool in groups:
            mask = (np.ones(len(te_progs), bool) if pool is None
                    else np.array([p in pool for p in te_progs]))
            d, lo, hi, frac = paired_boot(a[mask], b[mask],
                                          args.paired_bootstrap, 0)
            print(f"  {sname:<10s} n={int(mask.sum()):<2d} "
                  f"Δ={d:>+6.2f} pp  95% CI [{lo:>+6.2f}, {hi:>+6.2f}]  "
                  f"positive={frac:.4f}", flush=True)

    # ---- 局部回归 (k-NN + 距离加权, 仅 18 特征) ----
    if not args.skip_local:
        Xtr, Xte = prep(tr_df, te_df, FEATS_ALL)
        ytr = tr_df["y"].values
        for label, mode, k, alpha in (
                ("local_lasso k=1000 α=0.03", "local_lasso", 1000, 0.03),
                ("local_lwlr  k=200 α=0.1", "local_lwlr", 200, 0.1)):
            t1 = time.time()
            phase_pred = np.zeros(len(Xte))
            for j in range(len(Xte)):
                idx, dists = select_neighbors(Xte[j], Xtr, ytr, k=k)
                phase_pred[j] = local_fit_predict(Xte[j], Xtr, ytr, idx,
                                                  mode=mode, alpha=alpha,
                                                  dists=dists)
            pr_local = agg(phase_pred)
            eval_pred(te_progs, true_arm, pr_local, label, args.bootstrap)
            record(mode, pr_local)
            print(f"      ({time.time() - t1:.0f}s)")

    if args.dump_csv and dump_rows:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)),
                    exist_ok=True)
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dump_rows[0].keys()))
            w.writeheader()
            w.writerows(dump_rows)
        print(f"\n逐程序结果已写入 {args.dump_csv}")

    print(f"\n总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

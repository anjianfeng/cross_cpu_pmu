#!/usr/bin/env python3
"""phase 级跨平台预测分析（LACross 风格）。

思路：
  1. 每个程序在两平台各有一条 phase 轨迹（每行 = 一个由基本块数切出的窗口）。
     两平台的机器级基本块数因 ISA/寄存器压力/内联不同而不一致，因此不能直接
     按行号对齐；改为按「动态指令进度」重采样到 M 个等进度区间：
     两平台完成同一算法，指令进度的同比例点对应同一段程序工作（同一循环迭代）。
  2. 特征只取 x86 端（源平台）的量，标签为 ARM 端（目标平台）该进度区间的
     cycles 与 x86 的比值。这样模型输入在预测时是"源平台可直接测量"的量。
  3. 以程序为单位做 LOPO：用其余程序训练，预测被留出程序的每个 phase，
     再把 phase 预测求和得到整程序 cycles，与常数比值基线对比。

两组特征：
  基础组  6 个 PMU 事件推出的每周期事件率
  结构组  IR 插桩采集的指令构成（访存/整数/浮点/向量/除法/调用/分支密度、
          基本块平均大小、条件分支走向），来自 phase_profiler_ir.c

用法:
  python3 -W ignore scripts/analyze_phase.py
  python3 -W ignore scripts/analyze_phase.py --data data/phase_ir
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

M_BINS = 100          # 每个程序重采样成多少个等进度区间
EPS = 1e-9

EVENTS = ["cycles", "instructions", "cache_refs", "cache_misses",
          "branches", "branch_misses"]
STRUCT_COLS = ["s_ld", "s_st", "s_alu", "s_fp", "s_vec", "s_div",
               "s_call", "s_br", "s_inst", "s_bb", "s_taken", "s_notaken"]
# x86 富特征列（phase_collect_rich.py 采集，ARM 侧没有）
RICH_COLS = ["x_slots", "x_topdown_retiring", "x_topdown_bad_spec",
             "x_topdown_fe_bound", "x_topdown_be_bound",
             "x_topdown_mem_bound", "x_topdown_br_mispredict",
             "x_topdown_heavy_ops", "x_topdown_fetch_lat",
             "x_L1_dcache_load_misses", "x_LLC_load_misses",
             "x_dTLB_load_misses", "x_iTLB_load_misses",
             "x_L1_icache_load_misses", "x_mem_stores"]


def load_trace(path):
    df = pd.read_csv(path)
    valid = (df["cycles"] > 0) & (df["instructions"] > 0)
    return df[valid].reset_index(drop=True)


def resample(df, m_bins, cols, progress_col="instructions"):
    """按指定累计进度列重采样，返回 (m_bins, len(cols)) 的区间增量。

    progress_col 决定用什么来度量"进度"：
      "instructions"  — 假设完成 20% 指令 ≈ 同一算法阶段
      "cycles"        — 假设完成 20% 周期 ≈ 同一算法阶段（访存密集段周期占比更大，
                        对两平台差异更鲁棒）
    """
    vals = df[cols].values.astype(float)
    cum_prog = np.cumsum(df[progress_col].values.astype(float))
    total = cum_prog[-1]
    if total <= 0:
        return None

    cum = np.vstack([np.zeros((1, len(cols))), np.cumsum(vals, axis=0)])
    cum_prog = np.concatenate([[0.0], cum_prog])     # 与 cum 对齐
    bounds = np.linspace(0, total, m_bins + 1)
    out = np.zeros((m_bins, len(cols)))
    for j in range(len(cols)):
        v = np.interp(bounds, cum_prog, cum[:, j])
        out[:, j] = np.diff(v)
    return out


def dtw_path(x, y):
    """两条 (T, D) 序列的 DTW 最优对齐路径，向量化实现。
    返回长度 T 的 path：path[i]=j 表示 x 的第 i 帧对齐 y 的第 j 帧。"""
    T, D = x.shape
    # L2 距离矩阵 (T, T)
    dist = np.zeros((T, T))
    for i in range(T):
        dist[i] = np.sqrt(np.sum((x[i] - y) ** 2, axis=1))
    # DP
    dtw = np.full((T, T), np.inf)
    dtw[0, 0] = dist[0, 0]
    dtw[1:, 0] = np.cumsum(dist[1:, 0])
    dtw[0, 1:] = np.cumsum(dist[0, 1:])
    for i in range(1, T):
        for j in range(1, T):
            dtw[i, j] = dist[i, j] + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    # 回溯
    path = np.zeros(T, dtype=int)
    i, j = T - 1, T - 1
    path[i] = j
    while i > 0 or j > 0:
        choices = []
        if i > 0 and j > 0:
            choices.append((dtw[i-1, j-1], i-1, j-1))
        if i > 0:
            choices.append((dtw[i-1, j], i-1, j))
        if j > 0:
            choices.append((dtw[i, j-1], i, j-1))
        _, i, j = min(choices)
        path[i] = j
    return path


def build_rows(data_dir, exclude=(), progress_col="instructions", align="ratio",
               extra_counts=False):
    """构建 phase 训练样本。

    progress_col: 用哪个累计量度量 phase 进度
      "instructions" — 动态指令数（默认，两平台指令数天然同量纲）
      "cycles"       — 周期数（访存密集段占更大周期比例）

    align: phase 对齐方式
      "ratio" — 直接重采样到 M 个等进度区间，按行号对齐（假设"相同进度比例 = 同一阶段"）
      "dtw"   — 重采样后用 DTW 在两侧行为向量间求最优对齐路径，取 path 索引 ARM 值
    """
    rows = []
    for xf in sorted(glob.glob(os.path.join(data_dir, "x86", "*.csv"))):
        name = os.path.basename(xf)[:-4]
        if name in exclude:
            continue
        af = os.path.join(data_dir, "arm", f"{name}.csv")
        if not os.path.exists(af):
            continue
        dx, da = load_trace(xf), load_trace(af)
        if len(dx) < 10 or len(da) < 10:
            continue

        has_struct = all(c in dx.columns for c in STRUCT_COLS)
        has_rich = all(c in dx.columns for c in RICH_COLS)
        cols = EVENTS + (STRUCT_COLS if has_struct else []) + \
               (RICH_COLS if has_rich else [])
        # x86 和 ARM 各自用自己的 progress_col 重采样
        rx = resample(dx, M_BINS, cols, progress_col=progress_col)
        ra = resample(da, M_BINS, EVENTS, progress_col=progress_col)
        if rx is None or ra is None:
            continue

        # 对齐：ratio 直接按行号；dtw 找最优路径
        if align == "dtw":
            # 用两侧归一化的 cycles 轨迹做对齐信号（都是 0~1，形状可比）
            # 两侧的 6 个 perf event + IPC 都是形状对齐的好信号
            def norm(t): return t / (t.sum() + EPS)
            x_sig = np.column_stack([norm(rx[:, j]) for j in range(min(6, rx.shape[1]))])
            a_sig = np.column_stack([norm(ra[:, j]) for j in range(min(6, ra.shape[1]))])
            path = dtw_path(x_sig, a_sig)
            ra_aligned = ra[path]
        else:
            ra_aligned = ra

        ie = {c: i for i, c in enumerate(cols)}
        for m in range(M_BINS):
            cyc_x = max(rx[m, ie["cycles"]], 1.0)
            inst_x = rx[m, ie["instructions"]]
            cr, cm = rx[m, ie["cache_refs"]], rx[m, ie["cache_misses"]]
            br, bm = rx[m, ie["branches"]], rx[m, ie["branch_misses"]]
            # ARM 端取对齐后的 cycles
            cyc_a = ra_aligned[m, 0]   # EVENTS 首列是 cycles
            row = {
                "program": name, "phase": m,
                "x86_cycles": rx[m, ie["cycles"]],
                "arm_cycles": cyc_a,
                "f_ipc": inst_x / cyc_x,
                "f_cref": cr / cyc_x,
                "f_cmiss": cm / cyc_x,
                "f_br": br / cyc_x,
                "f_brmiss": bm / cyc_x,
                "f_cmiss_rate": cm / max(cr, 1.0),
                "f_brmiss_rate": bm / max(br, 1.0),
                "f_prog": m / M_BINS,
                "y": np.log10(max(cyc_a, 1.0) / cyc_x),
            }
            if extra_counts:      # 绝对计数特征, 供原始 CLSLR 形式使用
                row.update({"c_cycles": cyc_x, "c_inst": inst_x,
                            "c_cref": cr, "c_cmiss": cm,
                            "c_br": br, "c_brmiss": bm})
            if has_struct:
                s_inst = max(rx[m, ie["s_inst"]], EPS)
                s_bb = max(rx[m, ie["s_bb"]], EPS)
                tk, nt = rx[m, ie["s_taken"]], rx[m, ie["s_notaken"]]
                for key, col in (("ld", "s_ld"), ("st", "s_st"),
                                 ("alu", "s_alu"), ("fp", "s_fp"),
                                 ("vec", "s_vec"), ("div", "s_div"),
                                 ("call", "s_call"), ("br", "s_br")):
                    row["f_s_" + key] = rx[m, ie[col]] / s_inst
                row["f_bb_size"] = s_inst / s_bb
                row["f_taken"] = (tk + 1.0) / (tk + nt + 2.0)
            if has_rich:
                # topdown 归一到 slots（0~1 的比例，尺度无关）
                slots = max(rx[m, ie["x_slots"]], 1.0)
                kinst = max(inst_x, 1.0) / 1000.0
                row["f_x_retire"] = rx[m, ie["x_topdown_retiring"]] / slots
                row["f_x_badspec"] = rx[m, ie["x_topdown_bad_spec"]] / slots
                row["f_x_fe"] = rx[m, ie["x_topdown_fe_bound"]] / slots
                row["f_x_be"] = rx[m, ie["x_topdown_be_bound"]] / slots
                row["f_x_mem"] = rx[m, ie["x_topdown_mem_bound"]] / slots
                row["f_x_brspec"] = rx[m, ie["x_topdown_br_mispredict"]] / slots
                row["f_x_heavy"] = rx[m, ie["x_topdown_heavy_ops"]] / slots
                row["f_x_fetchlat"] = rx[m, ie["x_topdown_fetch_lat"]] / slots
                # 每千条指令的 miss 数（跨程序可比）
                row["f_x_l1dm"] = rx[m, ie["x_L1_dcache_load_misses"]] / kinst
                row["f_x_llcm"] = rx[m, ie["x_LLC_load_misses"]] / kinst
                row["f_x_dtlbm"] = rx[m, ie["x_dTLB_load_misses"]] / kinst
                row["f_x_itlbm"] = rx[m, ie["x_iTLB_load_misses"]] / kinst
                row["f_x_icm"] = rx[m, ie["x_L1_icache_load_misses"]] / kinst
                row["f_x_store"] = rx[m, ie["x_mem_stores"]] / max(inst_x, 1.0)
                # LLC 占 L1 miss 的比例：区分「容量型」与「流式/预取型」访存
                row["f_x_llc_share"] = rx[m, ie["x_LLC_load_misses"]] / \
                    max(rx[m, ie["x_L1_dcache_load_misses"]], 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


FEATS_BASE = ["f_ipc", "f_cref", "f_cmiss", "f_br", "f_brmiss",
              "f_cmiss_rate", "f_brmiss_rate", "f_prog"]
FEATS_STRUCT = ["f_s_ld", "f_s_st", "f_s_alu", "f_s_fp", "f_s_vec",
                "f_s_div", "f_s_call", "f_s_br", "f_bb_size", "f_taken"]
FEATS_ALL = FEATS_BASE + FEATS_STRUCT
FEATS_RICH = ["f_x_retire", "f_x_badspec", "f_x_fe", "f_x_be", "f_x_mem",
              "f_x_brspec", "f_x_heavy", "f_x_fetchlat",
              "f_x_l1dm", "f_x_llcm", "f_x_dtlbm", "f_x_itlbm",
              "f_x_icm", "f_x_store", "f_x_llc_share"]
FEATS_RICH_ALL = FEATS_ALL + FEATS_RICH
LINEAR_FEATS = {"f_prog", "f_taken"} | {f for f in FEATS_RICH}


def prep(train, test, feats):
    def tf(df):
        X = df[feats].values.astype(float).copy()
        for j, f in enumerate(feats):
            if f not in LINEAR_FEATS:            # 0~1 线性量不取 log
                X[:, j] = np.log10(np.abs(X[:, j]) + EPS)
        return X
    Xtr, Xte = tf(train), tf(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def program_totals(df):
    return df.groupby("program").agg(x86=("x86_cycles", "sum"),
                                     arm=("arm_cycles", "sum"))


def lopo(df, factory, feats):
    preds = {}
    for prog in df["program"].unique():
        tr = df[df["program"] != prog]
        te = df[df["program"] == prog]
        Xtr, Xte = prep(tr, te, feats)
        m = factory()
        m.fit(Xtr, tr["y"].values)
        ratio = 10 ** m.predict(Xte)
        # 预测的是比值，还原为目标平台绝对周期后再求和
        preds[prog] = float((te["x86_cycles"].values * ratio).sum())
    return preds


def const_baseline(df):
    """训练程序 log(arm总/x86总) 的均值，应用到被留出程序的 x86 总周期。"""
    tot = program_totals(df)
    preds = {}
    for prog in tot.index:
        tr = tot.drop(index=prog)
        ratio = (np.log10(tr["arm"]) - np.log10(tr["x86"])).mean()
        preds[prog] = float(tot.loc[prog, "x86"] * 10 ** ratio)
    return preds


def mape(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    return np.mean(np.abs(true - pred) / true) * 100


SWEEP = [dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8,
              random_state=42)
         for n, d, lr in [(150, 3, 0.05), (300, 3, 0.05), (500, 4, 0.05)]]


def evaluate(df, true_arm, names, feats, label, verbose=True):
    """返回 (最好 MAPE, 配置, 预测向量, 逐配置结果表)。"""
    rows = []
    best = None
    for cfg in SWEEP:
        pr = lopo(df, lambda c=cfg: GradientBoostingRegressor(**c), feats)
        pred = np.array([pr[p] for p in names])
        m = mape(true_arm, pred)
        c_ = np.corrcoef(np.log10(true_arm), np.log10(pred))[0, 1]
        r_ = np.sqrt(np.mean((np.log10(true_arm) - np.log10(pred)) ** 2))
        rows.append((cfg["n_estimators"], cfg["max_depth"],
                     cfg["learning_rate"], m, c_, r_))
        if best is None or m < best[0]:
            best = (m, cfg, pred, c_, r_)
    if verbose:
        print(f"\n[{label}] GBDT 超参搜索（LOPO，{len(feats)} 特征）")
        print(f"{'n_est':>6s}{'depth':>6s}{'lr':>6s}{'MAPE%':>8s}"
              f"{'corr':>8s}{'RMSLE':>8s}")
        print("-" * 42)
        for n, d, lr, m, c_, r_ in rows:
            print(f"{n:>6d}{d:>6d}{lr:>6.2f}{m:>8.2f}{c_:>8.4f}{r_:>8.4f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--no-struct", action="store_true",
                    help="即使有结构特征列也只使用基础 8 特征")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="排除的程序（两侧数据同时排除，保证可比）")
    ap.add_argument("--fast", action="store_true",
                    help="只跑最优已知超参，用于快速对比两组特征")
    ap.add_argument("--rich", action="store_true",
                    help="使用 x86 富特征（data/phase_ir_rich）")
    ap.add_argument("--progress-col", default="instructions",
                    choices=["instructions", "cycles"],
                    help="phase 进度的度量列（默认 instructions）")
    ap.add_argument("--align", default="ratio", choices=["ratio", "dtw"],
                    help="phase 对齐方式：ratio=直接按进度比例对齐，dtw=DTW 求最优对齐")
    ap.add_argument("--table3", action="store_true",
                    help="只打印程序级 ARM/x86 比值分布与常数基线 (Table 3)")
    args = ap.parse_args()

    df = build_rows(args.data, args.exclude,
                    progress_col=args.progress_col, align=args.align)
    if args.rich:
        df = df.dropna(subset=FEATS_RICH)
    feats_main = FEATS_RICH_ALL if args.rich else FEATS_ALL
    if args.fast:
        del SWEEP[1:]
    if args.exclude:
        df = df[~df["program"].isin(args.exclude)]
        print("已排除: " + ", ".join(args.exclude))
    if df.empty:
        print("没有可用数据（检查 %s 下 x86/arm 是否成对）" % args.data)
        return
    progs = sorted(df["program"].unique())
    if args.table3:
        tot = program_totals(df)
        ratio = (tot["arm"] / tot["x86"]).values
        preds = const_baseline(df)
        true = tot["arm"].values
        pred = np.array([preds[p] for p in tot.index])
        print(f"Table 3 复算: n={len(ratio)}, data={args.data}")
        print(f"  ratio mean = {ratio.mean():.3f}")
        print(f"  ratio median = {np.median(ratio):.3f}")
        print(f"  CV (ddof=1) = {ratio.std(ddof=1)/ratio.mean()*100:.2f}%")
        print(f"  CV (ddof=0) = {ratio.std(ddof=0)/ratio.mean()*100:.2f}%")
        print(f"  range = {ratio.min():.3f} -- {ratio.max():.3f} "
              f"({ratio.max()/ratio.min():.1f}x)")
        print(f"  p5--p95 = {np.percentile(ratio,5):.3f} -- "
              f"{np.percentile(ratio,95):.3f}")
        print(f"  constant-baseline LOPO MAPE = {mape(true, pred):.2f}%")
        for p in ("spec_lbm", "gemm", "2mm", "susan", "aes", "seidel-2d"):
            if p in tot.index:
                print(f"  {p:<10s} ratio = "
                      f"{tot.loc[p,'arm']/tot.loc[p,'x86']:.2f}x")
        return
    has_struct = "f_s_ld" in df.columns and not args.no_struct
    print(f"数据目录 {args.data}")
    print(f"phase 样本 {len(df)} 行，程序 {len(progs)} 个，"
          f"每程序 {M_BINS} 个进度区间，"
          f"特征 {len(feats_main) if has_struct else len(FEATS_BASE)} 维\n")

    tot = program_totals(df)
    true_arm = tot["arm"].values
    names = list(tot.index)

    # ---- 参照模型 ----
    p_const = const_baseline(df)
    pred_c = np.array([p_const[p] for p in names])
    print(f"{'方法':<18s}{'MAPE%':>8s}{'中位误差%':>10s}{'log corr':>10s}{'RMSLE':>8s}")
    print("-" * 54)
    err_c = np.abs(pred_c - true_arm) / true_arm * 100
    print(f"{'常数比值基线':<18s}{mape(true_arm, pred_c):>8.2f}"
          f"{np.median(err_c):>10.2f}"
          f"{np.corrcoef(np.log10(true_arm), np.log10(pred_c))[0, 1]:>10.4f}"
          f"{np.sqrt(np.mean((np.log10(true_arm) - np.log10(pred_c)) ** 2)):>8.4f}")

    for label, fac in (("岭回归 α=1", lambda: Ridge(alpha=1.0)),
                       ("Lasso α=0.001", lambda: Lasso(alpha=0.001, max_iter=20000))):
        pr = lopo(df, fac, feats_main if has_struct else FEATS_BASE)
        pred = np.array([pr[p] for p in names])
        err = np.abs(pred - true_arm) / true_arm * 100
        print(f"{label:<18s}{mape(true_arm, pred):>8.2f}{np.median(err):>10.2f}"
              f"{np.corrcoef(np.log10(true_arm), np.log10(pred))[0, 1]:>10.4f}"
              f"{np.sqrt(np.mean((np.log10(true_arm) - np.log10(pred)) ** 2)):>8.4f}")

    # ---- GBDT：基础特征 vs 基础+结构特征（同数据、同 LOPO 口径）----
    m0, cfg0, p0, c0, r0 = evaluate(df, true_arm, names, FEATS_BASE, "基础特征")
    err0 = np.abs(p0 - true_arm) / true_arm * 100
    print(f"{'GBDT 基础':<18s}{m0:>8.2f}{np.median(err0):>10.2f}"
          f"{c0:>10.4f}{r0:>8.4f}")

    if has_struct:
        m1, cfg1, p1, c1, r1 = evaluate(df, true_arm, names, FEATS_ALL,
                                        "基础+结构特征")
        err1 = np.abs(p1 - true_arm) / true_arm * 100
        print(f"{'GBDT 基础+结构':<18s}{m1:>8.2f}{np.median(err1):>10.2f}"
              f"{c1:>10.4f}{r1:>8.4f}")
        print(f"\n结构特征带来的 MAPE 变化: {m0:.2f}% -> {m1:.2f}% "
              f"({m1 - m0:+.2f} 个百分点)")

        # ---- 逐程序对比 ----
        print(f"\n逐程序误差（常数基线 / 基础 GBDT / 基础+结构 GBDT）")
        print(f"{'程序':<18s}{'真实ARM':>13s}{'常数%':>9s}{'基础%':>9s}{'结构%':>9s}")
        print("-" * 60)
        win = 0
        for i, p in enumerate(names):
            t = true_arm[i]
            ec = abs(pred_c[i] - t) / t * 100
            if err1[i] < err0[i]:
                win += 1
            print(f"{p:<18s}{int(t):>13,}{ec:>9.2f}{err0[i]:>9.2f}"
                  f"{err1[i]:>9.2f}")
        print(f"\n结构特征在 {win}/{len(names)} 个程序上优于基础特征")

        best_pred, best_feats, best_cfg = p1, FEATS_ALL, cfg1
    else:
        print("\n当前数据没有结构特征列（s_*），跳过对比")
        best_pred, best_feats, best_cfg = p0, FEATS_BASE, cfg0

    if args.rich:
        mr, cfgr, prr, cr, rr = evaluate(df, true_arm, names, FEATS_RICH_ALL,
                                         "基础+结构+富特征")
        errr = np.abs(prr - true_arm) / true_arm * 100
        print(f"{'GBDT 基础+结构+富':<18s}{mr:>8.2f}{np.median(errr):>10.2f}"
              f"{cr:>10.4f}{rr:>8.4f}")
        if has_struct:
            print(f"\n富特征带来的 MAPE 变化: {m1:.2f}% -> {mr:.2f}% "
                  f"({mr - m1:+.2f} 个百分点)")
        best_pred, best_feats, best_cfg = prr, FEATS_RICH_ALL, cfgr

    # 最优特征集下的长尾程序
    err = np.abs(best_pred - true_arm) / true_arm * 100
    order = np.argsort(-err)[:5]
    print("\n误差最大的 5 个程序: " +
          ", ".join(f"{names[i]} {err[i]:.1f}%" for i in order))

    # ---- 特征重要性与 phase 粒度拟合（全数据训练）----
    print("\n特征重要性（全数据训练，仅看模型学到了什么）")
    X = prep(df, df, best_feats)[0]
    g = GradientBoostingRegressor(**best_cfg).fit(X, df["y"].values)
    yp = g.predict(X)
    print(f"phase 级 log10(ARM/x86 比值) 相关系数 = "
          f"{np.corrcoef(df['y'].values, yp)[0, 1]:.4f}")
    imp = sorted(zip(best_feats, g.feature_importances_), key=lambda x: -x[1])
    for f, v in imp:
        print(f"  {v:>7.3f}  {f}")


if __name__ == "__main__":
    main()

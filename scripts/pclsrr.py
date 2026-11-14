#!/usr/bin/env python3
"""PCLSLR / 局部回归实现 (LACross 方法).

支持三种局部模型:
  - local_ols     : 邻域内最小二乘 (带可选截距)
  - local_ridge   : 邻域内岭回归 (L2 正则)
  - local_lasso   : 邻域内 Lasso (L1 正则, 论文的 CLSLR)

邻域选择:
  - epsilon (L2 半径) 或 k (k 近邻)

目标 y = log10(arm_cycles / x86_cycles).

用法:
  python3 -W ignore scripts/pclsrr.py --data data/phase_ir --exclude multi_ncut llvm_paq8p_paq8p
  python3 -W ignore scripts/pclsrr.py --data data/phase_ir --mode local_ols --k 500
  python3 -W ignore scripts/pclsrr.py --data data/phase_ir --mode local_lasso --sweep
"""

import argparse
import numpy as np
import pandas as pd
import sys, os, time
from sklearn.linear_model import Lasso, Ridge, LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import build_rows, FEATS_BASE, FEATS_STRUCT, FEATS_ALL, FEATS_RICH, FEATS_RICH_ALL, program_totals, mape

EPS = 1e-9


def select_neighbors(Xq, Xtr, ytr, epsilon=None, k=None):
    """返回 (邻域索引, 对应距离). 优先 epsilon, 若空或给定 k 则用 k."""
    diff = Xtr - Xq
    dists = np.sqrt((diff * diff).sum(1))
    if epsilon is not None and epsilon > 0:
        idx = np.where(dists <= epsilon)[0]
        if len(idx) > 0:
            return idx, dists[idx]
    if k is not None and k > 0:
        idx = np.argsort(dists)[:min(k, len(dists))]
        return idx, dists[idx]
    # 回退: k=2d+1
    d = len(Xq)
    idx = np.argsort(dists)[:min(2 * d + 1, len(dists))]
    return idx, dists[idx]


def local_fit_predict(Xq, Xtr, ytr, idx, mode="local_ols", alpha=1.0,
                      fit_intercept=True, dists=None):
    """在选定邻域上拟合局部模型, 返回对查询点 Xq 的预测.

    idx: 邻域索引
    mode: local_ols / local_ridge / local_lasso / local_lwlr
    alpha: L1 或 L2 正则强度
    fit_intercept: 是否拟合截距 (论文 CLSLR 不拟合截距, 因为标准化后 θ 均值为 0)
    dists: 邻域内样本到查询点的距离 (lwlr 加权用)
    """
    if len(idx) == 0:
        return float(np.mean(ytr))
    Xn = Xtr[idx]
    yn = ytr[idx]
    if len(idx) < 2:
        return float(np.mean(yn))

    sample_weight = None
    if mode == "local_lwlr":
        # 高斯核 + 自适应带宽 (邻域内最大距离), LOESS 风格的距离加权
        h = max(float(dists.max()) if dists is not None else 1.0, 1e-9)
        sample_weight = np.exp(-0.5 * (dists / h) ** 2)
        m = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif mode == "local_ols":
        m = LinearRegression(fit_intercept=fit_intercept)
    elif mode == "local_ridge":
        m = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif mode == "local_lasso":
        m = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=50000, positive=False)
    else:
        raise ValueError(f"unknown mode {mode}")
    m.fit(Xn, yn, sample_weight=sample_weight)
    return float(m.predict(Xq.reshape(1, -1))[0])


def lopo_local(df, feats, mode="local_ols", epsilon=None, k=None, alpha=1.0, fit_intercept=True):
    """LOPO 交叉验证 + 局部回归."""
    names = sorted(df["program"].unique())
    pred = {}

    def tf(df_in):
        X = df_in[feats].values.astype(float).copy()
        for j, f in enumerate(feats):
            if f not in ("f_prog", "f_taken"):
                X[:, j] = np.log10(np.abs(X[:, j]) + EPS)
        return X

    for idx_prog, prog in enumerate(names):
        tr_df = df[df["program"] != prog]
        te_df = df[df["program"] == prog]
        Xtr = tf(tr_df)
        ytr = tr_df["y"].values
        Xte = tf(te_df)
        yte = te_df["y"].values
        # 全局标准化
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd

        phase_pred = np.zeros(len(Xte_n))
        for j in range(len(Xte_n)):
            nbr_idx, nbr_dists = select_neighbors(Xte_n[j], Xtr_n, ytr,
                                                  epsilon=epsilon, k=k)
            phase_pred[j] = local_fit_predict(Xte_n[j], Xtr_n, ytr, nbr_idx,
                                              mode=mode, alpha=alpha,
                                              fit_intercept=fit_intercept,
                                              dists=nbr_dists)

        ratio = 10 ** phase_pred
        pred[prog] = float((te_df["x86_cycles"].values * ratio).sum())

        if (idx_prog + 1) % 20 == 0 or idx_prog + 1 == len(names):
            nbr_msg = f"k={k}" if k else f"ε={epsilon}"
            print(f"  LOOPO {idx_prog+1}/{len(names)}: prog={prog}, mode={mode}, {nbr_msg}")

    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--mode", default="local_ols",
                    choices=["local_ols", "local_ridge", "local_lasso",
                             "local_lwlr"])
    ap.add_argument("--epsilon", type=float, default=None,
                    help="L2 邻域半径 (标准化特征空间)")
    ap.add_argument("--k", type=int, default=None,
                    help="k 近邻数")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="L1/L2 正则强度")
    ap.add_argument("--no-intercept", action="store_true",
                    help="不拟合截距 (论文 CLSLR 不拟合)")
    ap.add_argument("--sweep", action="store_true",
                    help="粗扫参数")
    ap.add_argument("--rich", action="store_true",
                    help="用富特征 33 维 (需 data/phase_ir_rich)")
    ap.add_argument("--no-struct", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df = build_rows(args.data, args.exclude)
    progs = sorted(df["program"].unique())
    feats = FEATS_ALL if ("f_s_ld" in df.columns and not args.no_struct) else FEATS_BASE
    if args.rich and all(f in df.columns for f in FEATS_RICH):
        feats = FEATS_ALL + FEATS_RICH
    print(f"局部回归: {len(progs)} 程序, {len(df)} phases, {len(feats)} 特征\n")

    tot = program_totals(df)
    true_arm = tot["arm"].values
    names = list(tot.index)

    def run_and_print(label, pred_dict):
        err_arr = np.array([pred_dict[p] for p in names])
        m = mape(true_arm, err_arr)
        c_ = np.corrcoef(np.log10(true_arm), np.log10(err_arr))[0, 1]
        r_ = np.sqrt(np.mean((np.log10(true_arm) - np.log10(err_arr)) ** 2))
        errs = np.abs(err_arr - true_arm) / true_arm * 100
        print(f"  {label:<40s} MAPE={m:>6.2f}%  corr={c_:.4f}  "
              f"med={np.median(errs):.2f}%")
        return m

    # 常数基线
    p_const = {}
    for prog in names:
        tr = tot.drop(index=prog)
        ratio = (np.log10(tr["arm"]) - np.log10(tr["x86"])).mean()
        p_const[prog] = float(tot.loc[prog, "x86"] * 10 ** ratio)
    run_and_print("常数比值基线", p_const)

    if args.sweep:
        print("\n=== 模式 & 邻域扫描 ===")
        configs = []
        for mode in ["local_ols", "local_ridge", "local_lasso", "local_lwlr"]:
            # epsilon 扫描
            for eps in [0.5, 1.0, 2.0, 3.0, 5.0]:
                configs.append((mode, eps, None, 0.0))
            # k 扫描
            for kk in [50, 200, 500, 1000, 2000]:
                configs.append((mode, None, kk, 0.0 if mode == "local_ols" else
                               (0.01 if mode == "local_lasso" else
                                (0.1 if mode == "local_lwlr" else 1.0))))

        best = None
        for cfg in configs:
            mode, eps, kk, alpha = cfg
            t1 = time.time()
            label = f"{mode:12s} ε={eps} k={kk} α={alpha}"
            print(f"\n{label}...", flush=True)
            p = lopo_local(df, feats, mode=mode, epsilon=eps, k=kk,
                           alpha=alpha, fit_intercept=not args.no_intercept)
            dur = time.time() - t1
            m = run_and_print(label, p)
            if best is None or m < best[0]:
                best = (m, cfg, p)
            print(f"  用时 {dur:.0f}s")

        m_best, cfg_best, p_best = best
        print(f"\n最佳: {cfg_best[0]}, ε={cfg_best[1]}, k={cfg_best[2]}, α={cfg_best[3]}")
        print(f"  MAPE = {m_best:.2f}%")
    else:
        if args.epsilon is None and args.k is None:
            args.k = 500  # 默认 k 近邻
        print(f"\n运行 {args.mode}, ε={args.epsilon}, k={args.k}, α={args.alpha}")
        t1 = time.time()
        p = lopo_local(df, feats, mode=args.mode, epsilon=args.epsilon, k=args.k,
                       alpha=args.alpha, fit_intercept=not args.no_intercept)
        dur = time.time() - t1
        m = run_and_print(f"{args.mode}", p)
        print(f"用时 {dur:.0f}s")

        # 误差最大的 5 个
        errs = np.abs(np.array([p[prog] for prog in names]) - true_arm) / true_arm * 100
        order = np.argsort(-errs)[:5]
        print("\n误差最大 5 程序: " + ", ".join(f"{names[i]} {errs[i]:.1f}%" for i in order))


if __name__ == "__main__":
    main()

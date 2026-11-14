#!/usr/bin/env python3
"""深入建模分析：用 16 个已采集程序，系统比较不同特征/模型。

核心问题：
1. 为什么常数基线（ARM ≈ 2.04 × x86）这么强？
2. PMU 特征到底有没有信号？
3. 哪些特征最有用？（物理可解释）
4. 对异常程序（sha ARM/x86=1.11、dijkstra=1.28，远低于均值），PMU 特征能否识别并修正？
"""

import glob, os, statistics as st, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EPS = 1e-12

def med(df, key):
    col = [c for c in df.columns if key in c]
    if not col: return np.nan
    vals = df[col[0]].dropna()
    return st.median([int(v) for v in vals]) if len(vals) else np.nan

def load_all():
    frames = []
    for sub in ["mibench_v2", "sdvbs", "polybench", "spec"]:
        for fp in sorted(glob.glob(os.path.join(DATA_DIR, sub, "*_arm.csv"))):
            name = os.path.basename(fp).replace("_arm.csv", "")
            xfp = fp.replace("_arm.csv", "_x86.csv")
            frames.append((name, pd.read_csv(xfp), pd.read_csv(fp), sub))
    return frames

def build_df(frames, feat_mode="raw"):
    """feat_mode:
      raw     → 每 cycle 事件数 m/cycles（线性空间）
      log     → log10(m/cycles + eps)
      phys    → 显式物理量：IPC、cache miss ratio、分支预测准确率、topdown 占比
    """
    rows = []
    for name, xdf, adf, sub in frames:
        xc = med(xdf, "cpu_core/cycles/")
        xi = med(xdf, "cpu_core/instructions/")
        ac = med(adf, "armv8_pmuv3_0/cpu_cycles/")
        ai = med(adf, "armv8_pmuv3_0/inst_retired/")
        if np.isnan(xc) or np.isnan(ac): continue

        r = {"name": name, "suite": sub,
             "x86_cycles": xc, "x86_ipc": xi/xc if xc else np.nan,
             "arm_cycles": ac, "arm_ipc": ai/ac if ac else np.nan,
             "ratio": ac/xc,
             "y_log": np.log10(ac),
             "x_log": np.log10(xc)}

        if feat_mode == "phys":
            # 显式物理量，从 x86 端 PMU 构造
            def g(key): return med(xdf, key)
            # 分支
            br_total = g("br_inst_retired.all_branches")
            br_miss  = g("br_misp_retired.all_branches")
            r["br_acc"] = (br_total - br_miss) / br_total if br_total > 0 else np.nan
            r["br_per_cycle"] = br_total / xc
            # 缓存（Golden Cove 事件）
            l1d_loads   = g("l1d_uops_load")   or g("mem-loads")
            l1d_misses  = g("l1d_miss")
            llc_misses  = g("llc_miss")
            llc_access  = g("llc_access")   or l1d_loads
            r["l1d_miss_rate"] = l1d_misses / l1d_loads if l1d_loads > 0 else np.nan
            r["llc_miss_rate"] = llc_misses / llc_access if llc_access > 0 else np.nan
            # Topdown
            ret = g("topdown-retiring");  bad = g("topdown-bad-spec")
            fe  = g("topdown-fe-bound");  be  = g("topdown-be-bound")
            mem = g("topdown-mem-bound"); hea = g("topdown-heavy-ops")
            tot = ret + bad + fe + be + mem + hea
            for k, v in [("td_retiring", ret), ("td_bad_spec", bad),
                         ("td_fe_bound", fe), ("td_be_bound", be),
                         ("td_mem_bound", mem), ("td_heavy", hea)]:
                r[k] = v / tot if tot > 0 else np.nan
            # 指令每 cycle
            r["x86_ipc"] = xi / xc
            r["inst_per_byte"] = xi / (g("mem-loads") + g("mem-stores") + EPS) if g("mem-loads") and g("mem-stores") else np.nan
        else:
            for col in xdf.columns:
                if col in ("program","group","rep") or col == "cpu_core/cycles/":
                    continue
                if not col.startswith("cpu_core/"): continue
                # 只用关键事件
                name_body = col[len("cpu_core/"):-1]
                BLOCK = ["arith.", "assists.", "dsb.", "idq.", "xq.", "rtm.", "saferestarts",
                         "ms_decoder", "mite", "rs.", "eu.", "notes.", "other"]
                KEEP  = ["cycles","instructions","mem-loads","mem-stores","topdown-",
                         "br_","l1d","l2m","llc","dtlb","itlb","tlb","bus"]
                if any(b in name_body for b in BLOCK): continue
                if not any(k in name_body for k in KEEP): continue
                m = med(xdf, col)
                if np.isnan(m): continue
                base = m / xc
                if feat_mode == "log":
                    r[col] = np.log10(base + EPS)
                else:
                    r[col] = base
        rows.append(r)
    return pd.DataFrame(rows)

FILL = -12.0   # log10(1e-12)：表示该事件未计数 / 计数为 0

def prep(df_train, df_test):
    """缺失事件填充 + 标准化 + 截断，返回 (Xtr, Xte)。

    SPEC 程序的 CSV 里部分事件为空（<not counted>），直接 fillna(0) 会被
    解释成"每周期 1 次"这一荒谬值，导致回归系数爆炸（实测 MAPE 1e70）。
    正确做法是填 log10(EPS)=-12。标准化后再截断到 ±8，避免近常量列
    （std≈0）被放大成极端 z-score。
    """
    Xtr = df_train.fillna(FILL).values.astype(float)
    Xte = df_test.fillna(FILL).values.astype(float)
    s = StandardScaler().fit(Xtr)
    Xtr = np.clip(s.transform(Xtr), -8, 8)
    Xte = np.clip(s.transform(Xte), -8, 8)
    return np.nan_to_num(Xtr), np.nan_to_num(Xte)


def lopo(df, feat_cols, model_factory, scaler=True):
    y_true, y_pred, names = [], [], []
    for i in range(len(df)):
        tr = df.drop(df.index[i])
        te = df.iloc[[i]]
        Xtr, Xte = prep(tr[feat_cols], te[feat_cols])
        ytr = tr["y_log"].values
        m = model_factory()
        m.fit(Xtr, ytr)
        p = float(m.predict(Xte)[0])
        y_true.append(float(df.iloc[i]["arm_cycles"]))
        y_pred.append(10**p)
        names.append(df.iloc[i]["name"])
    return names, np.array(y_true), np.array(y_pred)

def metrics(y_true, y_pred):
    ape = np.abs(y_true - y_pred) / y_true
    mape = np.mean(ape) * 100
    corr = np.corrcoef(np.log10(y_true), np.log10(y_pred))[0, 1]
    rmsle = np.sqrt(np.mean((np.log10(y_true) - np.log10(y_pred))**2))
    return mape, corr, rmsle

def const_baseline(df):
    """用训练集的 log 比值均值预测"""
    yt, yp = [], []
    for i in range(len(df)):
        tr = df.drop(i)
        ratio_log = (tr["y_log"] - tr["x_log"]).mean()
        yt.append(df.iloc[i]["arm_cycles"])
        yp.append(10**(df.iloc[i]["x_log"] + ratio_log))
    return np.array(yt), np.array(yp)

def main():
    frames = load_all()
    print(f"加载 {len(frames)} 个程序")

    print("\n" + "="*60)
    print("一、程序 ARM/x86 比值排序")
    print("="*60)
    df = build_df(frames, feat_mode="raw")
    df_s = df.sort_values("ratio")
    for _, r in df_s.iterrows():
        marker = " ⚠️" if abs(r["ratio"] - df["ratio"].mean()) > df["ratio"].std() else ""
        print(f"  {r['suite']:12s} {r['name']:<22s} ratio={r['ratio']:.2f}{marker}")

    print("\n均值={:.2f}  std={:.2f}  min={:.2f}  max={:.2f}".format(
        df["ratio"].mean(), df["ratio"].std(), df["ratio"].min(), df["ratio"].max()))
    print("异常程序（偏离 > 1σ）: ",
          df[abs(df["ratio"] - df["ratio"].mean()) > df["ratio"].std()]["name"].tolist())

    print("\n" + "="*60)
    print("二、不同特征工程 + 模型组合的 LOPO 对比")
    print("="*60)

    configs = [
        ("物理特征 phys",   "phys"),
        ("log 特征 (81事件)", "log"),
        ("raw 特征",         "raw"),
    ]
    models = [
        ("常数基线",      None),
        ("岭回归 α=0.1",  lambda: Ridge(alpha=0.1)),
        ("岭回归 α=1",    lambda: Ridge(alpha=1.0)),
        ("岭回归 α=10",   lambda: Ridge(alpha=10.0)),
        ("Lasso α=0.001", lambda: Lasso(alpha=0.001, max_iter=10000)),
        ("梯度提升树",    lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ]

    print(f"{'特征集':<18s}{'模型':<16s}{'MAPE%':>8s}{'log corr':>10s}{'RMSLE':>8s}{'特征数':>8s}")
    print("-"*74)
    for feat_label, mode in configs:
        d = build_df(frames, feat_mode=mode)
        feats = [c for c in d.columns if c not in ("name","suite","x86_cycles","arm_cycles","ratio","y_log","x_log","x86_ipc","arm_ipc")]
        for mlabel, factory in models:
            if factory is None:
                yt, yp = const_baseline(d)
            else:
                _, yt, yp = lopo(d, feats, factory)
            mape, corr, rmsle = metrics(yt, yp)
            print(f"{feat_label:<18s}{mlabel:<16s}{mape:>8.2f}{corr:>10.4f}{rmsle:>8.4f}{len(feats):>8d}")
        print()

    print("\n" + "="*60)
    print("三、特征重要性（岭回归 α=1，用 log 特征训练全部程序）")
    print("="*60)
    df = build_df(frames, feat_mode="log")
    feats = [c for c in df.columns if c.startswith("cpu_core/")]
    Xs, _ = prep(df[feats], df[feats].iloc[[0]])
    y = df["y_log"].values
    r = Ridge(alpha=1.0).fit(Xs, y)
    coefs = sorted(zip(feats, r.coef_), key=lambda x: abs(x[1]), reverse=True)
    print("Top 15 岭回归系数（对 log10(ARM cycles) 的影响）:")
    for f, c in coefs[:15]:
        mark = "↑ 增大 ARM cycles" if c > 0 else "↓ 减小 ARM cycles"
        print(f"  {c:>+7.3f}  {mark:20s}  {f}")

    print("\n" + "="*60)
    print("四、Lasso 稀疏特征（α=0.005）")
    print("="*60)
    l = Lasso(alpha=0.005, max_iter=20000).fit(Xs, y)
    picked = [(f, c) for f, c in zip(feats, l.coef_) if abs(c) > 1e-4]
    picked.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"Lasso 选出 {len(picked)} / {len(feats)} 个非零特征:")
    for f, c in picked[:20]:
        print(f"  {c:>+7.3f}  {f}")

    print("\n" + "="*60)
    print("五、异常程序逐程序对比（常数 vs Lasso α=0.001）")
    print("="*60)
    df = build_df(frames, feat_mode="log")
    feats = [c for c in df.columns if c.startswith("cpu_core/")]
    yt_const, yp_const = const_baseline(df)
    nc, yt2, yp_lasso = lopo(df, feats, lambda: Lasso(alpha=0.001, max_iter=10000))
    print(f"{'程序':<22s}{'ARM/x86':>8s}{'真实cycles':>14s}{'常数误差%':>10s}{'岭回归误差%':>10s}")
    print("-"*74)
    for i, n in enumerate(nc):
        row = df[df["name"] == n].iloc[0]
        c_err = abs(yp_const[i] - yt_const[i]) / yt_const[i] * 100
        m_err = abs(yp_lasso[i] - yt2[i]) / yt2[i] * 100
        flag = "✅ ML赢" if m_err < c_err else ""
        print(f"{n:<22s}{row['ratio']:>8.2f}{int(yt_const[i]):>14,}{c_err:>10.2f}{m_err:>10.2f} {flag}")

    print("\n总结：样本量仅 16 个时，常数基线（均值比 2.04）对大多数程序已经"
          "足够好。ML 模型要超过它，需要更多样本（≥50）让特征真正起作用。")

if __name__ == "__main__":
    main()

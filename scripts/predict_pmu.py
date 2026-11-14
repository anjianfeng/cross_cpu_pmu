#!/usr/bin/env python3
"""最小实验 v2：对数空间 + 筛选公共 PMU 特征。

改进点：
1. y = log10(ARM cycles)，特征取 log10(事件数 / cycles + eps) 以压缩长尾
2. 只保留两平台共有的事件（通用事件 + cycles/instructions + 常见缓存/TLB/分支）
3. 岭回归 + 调参、对数空间常数基线
"""

import csv, glob, os, statistics as st, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EPS = 1e-12

# 目标：只保留 ARM 也能计数的事件（ARM 33 事件 → 对应 x86 上的通用事件）
# 用关键词匹配，同时保留 Golden Cove 上有意义的通用缓存/分支事件
ARM_EVENT_KEYWORDS = [
    "cycles", "instructions", "br", "branch",
    "cache", "L1", "L2", "LLC", "miss", "load", "store", "prefetch",
    "tlb",
    "bus",
]

def is_common_event(col):
    """x86 端特征筛选：保留通用事件（13 个）+ 有意义的微架构事件。

    - 通用事件格式: cpu_core/cycles/ （name 中无点号）
    - 微架构事件格式: cpu_core/br_inst_retired.cond/ （name 中带点号）

    我们关心的特征（都能在 ARM 端找到语义上的对应）：
      分支: br_*, topdown-bad-spec
      缓存: mem-loads, mem-stores, 带 l1d/l2m/llc 的 Golden Cove 事件
      TLB: 带 tlb 的
      指令级: topdown-*, instructions, cycles
    排除: arith.*, assists.*, dsb.*, idq.*, xq.* 等纯 Golden Cove 内部事件
    """
    if not col.startswith("cpu_core/") or col.endswith("cycles/"):
        return False
    name = col[len("cpu_core/"):-1]   # 事件名本体（去掉前缀和末尾斜杠）

    # 保留的关键词
    keep_any = [
        "cycles", "instructions",
        "mem-loads", "mem-stores",
        "topdown-",
        "br_", "branch",
        "l1d", "l1i", "l2m", "l2_hit", "llc", "llc_miss", "cache", "dtlb", "itlb", "tlb",
        "bus",
        "retiring",
    ]
    # 纯 Golden Cove 内部事件（排除）
    block_any = [
        "arith.", "assists.", "baclears", "cpl.", "dsb.", "eu.",
        "idq.", "ild.", "ip.", "itlb.", "lsd.", "machine.",
        "ms_decoder", "ms_dsb", "mite", "notes.", "offcore", "other",
        "pcount.", "rs.", "rtm.", "saferestarts", "smx.", "srq.",
        "tx", "uops.", "xq.", "cycles_ge",
    ]
    if any(k in name for k in block_any):
        return False
    return any(k in name for k in keep_any)

def load_all():
    frames = []
    for sub in ["mibench_v2", "sdvbs"]:
        for fp in sorted(glob.glob(os.path.join(DATA_DIR, sub, "*_arm.csv"))):
            name = os.path.basename(fp).replace("_arm.csv", "")
            xfp = fp.replace("_arm.csv", "_x86.csv")
            a = pd.read_csv(fp)
            x = pd.read_csv(xfp)
            frames.append((name, x, a, sub))
    return frames

def med(df, key):
    col = [c for c in df.columns if key in c]
    if not col:
        return np.nan
    vals = df[col[0]].dropna()
    return st.median([int(v) for v in vals]) if len(vals) else np.nan

def build_samples(frames):
    rows = []
    for name, xdf, adf, sub in frames:
        xc = med(xdf, "cpu_core/cycles/")
        ac = med(adf, "armv8_pmuv3_0/cpu_cycles/")
        if np.isnan(xc) or np.isnan(ac):
            print(f"SKIP {name}"); continue
        r = {"name": name, "suite": sub,
             "x86_cycles": xc, "arm_cycles": ac,
             "y_log": np.log10(ac)}
        # 特征：x86 端每 cycle 计数，再 log10
        for col in xdf.columns:
            if col in ("program","group","rep") or col == "cpu_core/cycles/":
                continue
            if not is_common_event(col):
                continue
            m = med(xdf, col)
            if np.isnan(m):
                continue
            r[col] = np.log10(m / xc + EPS)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"样本数: {len(df)}, 特征数: {len([c for c in df.columns if c.startswith('cpu_core/')])}")
    return df

def lopo(df, model_factory, scaler=True):
    feats = [c for c in df.columns if c.startswith("cpu_core/")]
    y_true, y_pred, names = [], [], []
    for i, (_, tr) in enumerate(df.iterrows()):
        train = df.drop(tr.name)
        Xtr = train[feats].fillna(-np.log10(EPS)).values
        ytr = train["y_log"].values
        Xte = tr[feats].fillna(-np.log10(EPS)).values.reshape(1, -1)
        if scaler:
            s = StandardScaler().fit(Xtr)
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        m = model_factory()
        m.fit(Xtr, ytr)
        p = float(m.predict(Xte)[0])
        y_true.append(10**float(tr["y_log"]))
        y_pred.append(10**p)
        names.append(tr["name"])
    return names, np.array(y_true), np.array(y_pred)

def metrics(y_true, y_pred):
    ape = np.abs(y_true - y_pred) / y_true
    mape = np.mean(ape) * 100
    corr = np.corrcoef(np.log10(y_true), np.log10(y_pred))[0, 1]
    rmsle = np.sqrt(np.mean((np.log10(y_true) - np.log10(y_pred))**2))
    return mape, corr, rmsle

def main():
    frames = load_all()
    df = build_samples(frames)

    feats = [c for c in df.columns if c.startswith("cpu_core/")]
    print(f"\n用的 {len(feats)} 个公共特征 (前 10): {feats[:10]}")

    print(f"\n=== 程序 ===")
    for _, r in df.iterrows():
        print(f"  {r['suite']:12s} {r['name']:<22s} x86={r['x86_cycles']:>14,.0f}  ARM={r['arm_cycles']:>14,.0f}  ARM/x86={r['arm_cycles']/r['x86_cycles']:.2f}")

    models = {
        "常数基线(训练集均值比)": "const",
        "岭回归(alpha=1)":       lambda: Ridge(alpha=1.0),
        "岭回归(alpha=100)":     lambda: Ridge(alpha=100),
        "Lasso(alpha=0.01)":     lambda: Lasso(alpha=0.01, max_iter=10000),
        "随机森林":              lambda: RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
        "梯度提升树":            lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    }

    print(f"\n{'模型':<30s}{'MAPE%':>8s}{'log10 corr':>12s}{'RMSLE':>10s}")
    print("-"*62)
    for label, factory in models.items():
        if factory == "const":
            yt, yp, nm = [], [], []
            for i in range(len(df)):
                tr = df.drop(i)
                ratio = (tr["y_log"] - np.log10(tr["x86_cycles"])).mean()
                yt.append(df.iloc[i]["arm_cycles"])
                yp.append(10**(np.log10(df.iloc[i]["x86_cycles"]) + ratio))
                nm.append(df.iloc[i]["name"])
        else:
            nm, yt, yp = lopo(df, factory)
        mape, corr, rmsle = metrics(np.array(yt), np.array(yp))
        print(f"{label:<30s}{mape:>8.2f}{corr:>12.4f}{rmsle:>10.4f}")

    # 岭回归最佳模型逐程序
    print(f"\n=== 岭回归(alpha=100) 逐程序详情 (LOPO) ===")
    nm, yt, yp = lopo(df, lambda: Ridge(alpha=100))
    print(f"{'程序':<22s}{'真实ARM cycles':>14s}{'预测cycles':>14s}{'绝对误差%':>10s}{'ARM/x86':>10s}")
    for n, t, p in zip(nm, yt, yp):
        row = df[df["name"] == n].iloc[0]
        print(f"{n:<22s}{t:>14,.0f}{p:>14,.0f}{abs(p-t)/t*100:>10.2f}{row['arm_cycles']/row['x86_cycles']:>10.2f}")

    print(f"\n=== 常数基线逐程序详情 ===")
    for i in range(len(df)):
        tr = df.drop(i)
        ratio = (tr["y_log"] - np.log10(tr["x86_cycles"])).mean()
        row = df.iloc[i]
        t = row["arm_cycles"]
        p = 10**(np.log10(row["x86_cycles"]) + ratio)
        print(f"{row['name']:<22s}{t:>14,.0f}{p:>14,.0f}{abs(p-t)/t*100:>10.2f}")

    # 关键观察：常数基线其实就是 "ARM = 2.06 * x86" 这个固定乘数
    ratio = (df["arm_cycles"] / df["x86_cycles"]).mean()
    print(f"\n训练集均值 ARM/x86 = {ratio:.2f}  即 ARM ≈ {ratio:.2f} × x86")

if __name__ == "__main__":
    main()

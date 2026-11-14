#!/usr/bin/env python3
"""量化不同对齐策略下标签本身的信噪比。

对每个程序：
  - ratio 对齐：直接按进度比例取 arm_cycles / x86_cycles 的对数比值
  - dtw 对齐：先求 DTW 对齐路径，再取 arm_cycles / x86_cycles 的对数比值

然后看：
  1. 同一程序内 phase 间 log 比值的方差（越小越好 = 标签越一致）
  2. 两侧 phase 轨迹形状的 Pearson 相关（越大越好 = 对齐越好）
  3. 用标签的均值做预测（如果模型比这个好就有意义）
"""

import glob, os, sys
import csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_phase as a

EPS = 1e-9
EXCLUDE = ("multi_ncut", "llvm_paq8p_paq8p")


def load_one(xf, af, progress_col):
    dx, da = a.load_trace(xf), a.load_trace(af)
    if len(dx) < 10 or len(da) < 10:
        return None
    cols = a.EVENTS + a.STRUCT_COLS
    rx = a.resample(dx, a.M_BINS, cols, progress_col=progress_col)
    ra = a.resample(da, a.M_BINS, a.EVENTS, progress_col=progress_col)
    if rx is None or ra is None:
        return None
    return rx, ra


def dtw_align(rx, ra):
    x_sig = rx[:, :6] / (rx[:, 1:2] + EPS)
    a_sig = ra / (ra[:, 1:2] + EPS)
    path = a.dtw_path(x_sig, a_sig)
    return path


def program_corr(rx, ra_cycles):
    """归一化 cycles 轨迹的 Pearson 相关。"""
    x = rx[:, 0] / max(rx[:, 0].sum(), EPS)
    y = ra_cycles / max(ra_cycles.sum(), EPS)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def phase_labels(rx, ra_aligned):
    """返回每个 phase 的 log10(arm_cycles / x86_cycles) 标签。"""
    cyc_x = np.maximum(rx[:, 0], 1.0)
    cyc_a = np.maximum(ra_aligned[:, 0], 1.0)
    return np.log10(cyc_a / cyc_x)


def main():
    progress_col = sys.argv[3] if len(sys.argv) > 3 else "instructions"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/phase_ir"
    dump = sys.argv[2] if len(sys.argv) > 2 else ""

    # 先收集所有配对
    pairs = []
    for xf in sorted(glob.glob(os.path.join(data_dir, "x86", "*.csv"))):
        name = os.path.basename(xf)[:-4]
        if name in EXCLUDE: continue
        af = os.path.join(data_dir, "arm", name + ".csv")
        if os.path.exists(af):
            pairs.append((name, xf, af))

    print(f"分析 {len(pairs)} 对程序，progress_col={progress_col}\n")

    ratio_stats = {"corrs": [], "label_stds": []}
    dtw_stats = {"corrs": [], "label_stds": []}
    improvements = []

    for name, xf, af in pairs:
        result = load_one(xf, af, progress_col)
        if result is None: continue
        rx, ra = result

        # ratio 对齐
        ra_ratio = ra
        corr_ratio = program_corr(rx, ra_ratio[:, 0])
        labels_ratio = phase_labels(rx, ra_ratio)
        std_ratio = np.std(labels_ratio)

        # dtw 对齐
        path = dtw_align(rx, ra)
        ra_dtw = ra[path]
        corr_dtw = program_corr(rx, ra_dtw[:, 0])
        labels_dtw = phase_labels(rx, ra_dtw)
        std_dtw = np.std(labels_dtw)

        ratio_stats["corrs"].append(corr_ratio)
        ratio_stats["label_stds"].append(std_ratio)
        dtw_stats["corrs"].append(corr_dtw)
        dtw_stats["label_stds"].append(std_dtw)
        improvements.append((name, corr_ratio, corr_dtw, std_ratio, std_dtw))

    N = len(improvements)
    print(f"{'指标':<28s}{'ratio':>10s}{'dtw':>10s}{'Δdtw-ratio':>12s}{'改善率':>10s}")
    print("-" * 72)

    for metric, unit in [
        ("corrs",      "x86/ARM 形状相关"),
        ("label_stds", "标签 log 比值 std"),
    ]:
        vals_r = ratio_stats[metric]
        vals_d = dtw_stats[metric]
        diff = np.mean(vals_d) - np.mean(vals_r)
        rate = np.mean(vals_d < vals_r) if metric == "label_stds" else np.mean(vals_d > vals_r)
        print(f"{unit:<28s}{np.mean(vals_r):>10.4f}{np.mean(vals_d):>10.4f}{diff:>+12.4f}{rate:>9.1%}")

    print(f"\n按形状相关改善排序（前 15）：")
    improvements.sort(key=lambda x: x[2] - x[1], reverse=True)
    print(f"  {'程序':<30s}{'ratio':>7s}{'dtw':>7s}{'Δ':>7s}{'std_ratio':>10s}{'std_dtw':>10s}")
    for name, cr, cd, sr, sd in improvements[:15]:
        print(f"  {name:<30s}{cr:>7.3f}{cd:>7.3f}{cd-cr:>+7.3f}{sr:>10.4f}{sd:>10.4f}")

    # 形状相关分布
    print(f"\n形状相关分布：")
    for label, vals in [("ratio", ratio_stats["corrs"]), ("dtw", dtw_stats["corrs"])]:
        print(f"  {label}: >0.9: {sum(v>0.9 for v in vals)}, "
              f"0.7-0.9: {sum(0.7<=v<=0.9 for v in vals)}, "
              f"<0.7: {sum(v<0.7 for v in vals)}, "
              f"负相关: {sum(v<0 for v in vals)}")

    print(f"\nDTW 让形状相关变好的程序: {sum(cd > cr for _, cr, cd, _, _ in improvements)}/{N}")
    print(f"DTW 让标签方差变小的程序: {sum(sd < sr for _, _, _, sr, sd in improvements)}/{N}")

    if dump:
        os.makedirs(os.path.dirname(os.path.abspath(dump)), exist_ok=True)
        with open(dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["program", "corr_ratio", "corr_dtw",
                        "std_ratio", "std_dtw"])
            for name, cr, cd, sr, sd in improvements:
                w.writerow([name, f"{cr:.6f}", f"{cd:.6f}",
                            f"{sr:.6f}", f"{sd:.6f}"])
        print(f"逐程序对齐质量已写入 {dump}")


if __name__ == "__main__":
    main()

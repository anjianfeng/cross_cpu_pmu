#!/usr/bin/env python3
"""Within-program standard deviation of the phase-level log ratio.

Round-4 M11: give the 0.021 / 0.025 dex sentence in Section 4.2 a script and a
log.  The definition follows the pipeline exactly:

  * each platform trace is resampled to M_BINS = 100 equal-instruction-progress
    segments with ``analyze_phase.load_trace`` + ``analyze_phase.resample``;
  * the label of segment m is y(m) = log10(max(a(m), 1) / max(x(m), 1));
  * the within-program sd is the population sd (ddof = 0) of the 100 y values;
  * the corpus statistic is the median of the per-program sds over all programs
    that have both traces (159 for data/phase_ir_rich).

Usage:
  python3 -W ignore scripts/within_sd.py \
      --log doc/reviews/round4/logs/within_sd_v5.log
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase import M_BINS, load_trace, resample

RICH_EXCLUDE = ["multi_ncut", "llvm_paq8p_paq8p",
                "llvm_doeproxyappsc_SimpleMOC",
                "llvm_optimizereval_optimizer_eval"]


def within_sd(program, x86_csv, arm_csv):
    dx = load_trace(x86_csv)
    da = load_trace(arm_csv)
    if len(dx) < 10 or len(da) < 10:
        return None
    rx = resample(dx, M_BINS, ["cycles"], progress_col="instructions")
    ra = resample(da, M_BINS, ["cycles"], progress_col="instructions")
    if rx is None or ra is None:
        return None
    x = np.maximum(rx[:, 0], 1.0)
    a = np.maximum(ra[:, 0], 1.0)
    y = np.log10(a / x)
    return float(np.std(y, ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phase_ir_rich")
    ap.add_argument("--exclude", nargs="*", default=RICH_EXCLUDE)
    ap.add_argument("--log", default="doc/reviews/round4/logs/within_sd_v5.log")
    args = ap.parse_args()

    names = []
    for path in sorted(glob.glob(os.path.join(args.data, "x86", "*.csv"))):
        p = os.path.basename(path)[:-4]
        if p in args.exclude:
            continue
        if os.path.exists(os.path.join(args.data, "arm", f"{p}.csv")):
            names.append(p)

    per = {}
    for p in names:
        sd = within_sd(p,
                       os.path.join(args.data, "x86", f"{p}.csv"),
                       os.path.join(args.data, "arm", f"{p}.csv"))
        if sd is not None:
            per[p] = sd

    vals = np.array(list(per.values()))
    median = float(np.median(vals))
    lines = [
        "within-program sd of y = log10(max(ARM cycles,1)/max(x86 cycles,1))",
        f"data = {args.data}",
        f"M_BINS = {M_BINS} (equal-instruction-progress resampling, "
        "analyze_phase.resample)",
        "sd = population standard deviation over the 100 phases (ddof=0)",
        f"programs with both traces = {len(vals)}",
        "",
    ]
    for p in sorted(per):
        lines.append(f"{p:<40s} sd={per[p]:.4f} dex")
    lines += [
        "",
        f"2mm within-program sd = {per['2mm']:.4f} dex",
        f"median across {len(vals)} programs = {median:.4f} dex",
        "manuscript reports 0.021 dex (2mm) and 0.025 dex (median), "
        "i.e. the values rounded to three decimals",
    ]

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines[-3:]:
        print(line)


if __name__ == "__main__":
    main()

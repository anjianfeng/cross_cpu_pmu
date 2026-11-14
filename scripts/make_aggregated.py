#!/usr/bin/env python3
"""把原始 phase_ir + phase_ir_rich CSV 分别聚合, 生成两份交付文件.

为什么两份?
  phase_ir 和 phase_ir_rich 是两次不同时间的 PMU 采集 (分别用了 phase_collect.py
  和 phase_collect_rich.py), 基础事件值在某些程序上有差异. 论文 Table 7 / 表 3 /
  表 8 的两列分别来自两源:  Base 18 Feats 列 → phase_ir (24.34% 等),  Rich 33 Feats
  列 → phase_ir_rich (14.51% 等). 两份聚合让读者能复现任一列.

输出 (相对仓库根):
  data/phase_base.csv.gz    ← phase_ir 源, 161 programs, 16100 phases, 22 列
                              (8 base feats + 10 struct feats, 无 rich 列)
  data/phase_rich.csv.gz    ← phase_ir_rich 源, 159 programs, 15900 phases, 38 列
                              (18 feats + 15 rich top-down + y + cycles)

用法:
  python3 scripts/make_aggregated.py
"""
import os, gzip, sys
sys.path.insert(0, os.path.dirname(__file__))
from analyze_phase import build_rows

EXCLUDE = ["multi_ncut", "llvm_paq8p_paq8p"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(src, out_name, label, force_base_only=False):
    out = os.path.join(ROOT, "data", out_name)
    df = build_rows(src, EXCLUDE)
    if force_base_only:
        df = df.drop(columns=[c for c in df.columns if c.startswith("f_x")])
    df = df.sort_values(["program", "phase"]).reset_index(drop=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    sz = os.path.getsize(out) / 1024
    print(f"[{label:<18s}] {out_name:<28s}  {sz:>6.0f} KB  "
          f"{df.program.nunique():>3} progs × {len(df)} phases  "
          f"{df.shape[1]} cols")
    return df


def main():
    # phase_ir 里 build_rows 若检测不到 x_slots 等列就不生成 f_x_*; 保险起见强制 drop
    one("data/phase_ir", "phase_base.csv.gz", "BASE (18 feats)",
        force_base_only=True)
    one("data/phase_ir_rich", "phase_rich.csv.gz", "RICH (33 feats)",
        force_base_only=False)


if __name__ == "__main__":
    main()

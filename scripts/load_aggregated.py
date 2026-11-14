#!/usr/bin/env python3
"""加载 data/phase_base.csv.gz 或 data/phase_rich.csv.gz.

与 scripts/analyze_phase.build_rows(data_dir, exclude) 等价, 但直接读预聚合
CSV, 省掉 phase 重采样 (原始 phase_ir / phase_ir_rich 合计 16 GB).

用法:
  from load_aggregated import load_aggregated
  df_base = load_aggregated("base")    # 161 progs, 18 feats, 复现 Table 7 Base 列
  df_rich = load_aggregated("rich")    # 159 progs, 33 feats, 复现 Table 7 Rich 列

下游脚本里原来的:
  df = build_rows("data/phase_ir", ["multi_ncut", "llvm_paq8p_paq8p"])
替换为:
  df = load_aggregated("base")
或 (原来用 phase_ir_rich + --rich 标志时):
  df = load_aggregated("rich")
"""
import os
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "base": os.path.join(ROOT, "data", "phase_base.csv.gz"),
    "rich": os.path.join(ROOT, "data", "phase_rich.csv.gz"),
}
DEFAULT_EXCLUDE = {"multi_ncut", "llvm_paq8p_paq8p"}


def load_aggregated(kind="rich", exclude=None, path=None):
    """返回聚合好的 DataFrame.

    kind: "rich" (默认, 33 特征, 159 程序, phase_ir_rich 源, 复现 Table 7 Rich 列)
          "base" (18 特征, 161 程序, phase_ir 源, 复现 Table 7 Base 列)
    exclude:  额外的程序名排除; multi_ncut / llvm_paq8p_paq8p 始终排除
    path:    可选, 手动指定 CSV 路径 (覆盖 kind)
    """
    p = path or PATHS.get(kind)
    if not p or not os.path.exists(p):
        raise FileNotFoundError(
            f"找不到 {p}. 请先运行 `python3 scripts/make_aggregated.py`, "
            "或从论文仓库下载 data/phase_base.csv.gz 与 data/phase_rich.csv.gz.")
    df = pd.read_csv(p, compression="gzip")
    exc = DEFAULT_EXCLUDE | set(exclude or ())
    df = df[~df.program.isin(exc)].reset_index(drop=True)
    return df


if __name__ == "__main__":
    for k in ("base", "rich"):
        df = load_aggregated(k)
        print(f"[{k}] {df.program.nunique():>3} progs × {len(df):>6} phases, "
              f"{df.shape[1]} cols, y range [{df.y.min():.2f}, {df.y.max():.2f}]")

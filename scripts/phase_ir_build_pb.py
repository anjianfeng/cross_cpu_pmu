#!/usr/bin/env python3
"""用 IR 插桩管线构建全部 PolyBench（x86 + aarch64）。

与 phase_build.sh 的区别：不是 -fsanitize-coverage，而是在优化后的 IR 上
给每个基本块编号并插入 __phase_bb(id)，同时导出静态结构表，
使每个 phase 除了 6 个 PMU 事件外还能得到指令构成特征。

用法: python3 scripts/phase_ir_build_pb.py [输出目录]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_ir_build import ROOT, build_target

SRC = os.path.join(ROOT, "benchmarks/polybench")
UTIL = os.path.join(SRC, "utilities")

# MEDIUM 下运行不足 1ms 的程序改用 LARGE，与 phase_build.sh 保持一致
BIG = {"atax", "bicg", "mvt", "gemver", "gesummv", "trisolv", "durbin",
       "jacobi-1d"}


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "build/phase_ir")
    rels = [x.strip() for x in open(os.path.join(UTIL, "benchmark_list"))
            if x.strip()]

    ok = fail = 0
    for rel in rels:
        src = os.path.join(SRC, rel.lstrip("./"))
        name = os.path.splitext(os.path.basename(rel))[0]
        ds = "LARGE_DATASET" if name in BIG else "MEDIUM_DATASET"
        cflags = ["-D" + ds, "-I" + UTIL, "-I" + os.path.dirname(src)]
        try:
            for tag, tf in (("x86", []),
                            ("arm", ["--target=aarch64-linux-gnu", "-static"])):
                build_target(name, [src, os.path.join(UTIL, "polybench.c")],
                             outdir, tag, tf, cflags, ["-lm"])
            ok += 1
            print("[%d/%d] %s OK" % (ok + fail, len(rels), name))
        except RuntimeError as e:
            fail += 1
            print("[%s] 失败\n%s" % (name, str(e)[-600:]))

    print("完成: 成功 %d, 失败 %d -> %s" % (ok, fail, outdir))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

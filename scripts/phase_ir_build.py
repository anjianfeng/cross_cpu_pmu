#!/usr/bin/env python3
"""构建带 IR 层结构插桩的 phase 采集二进制（x86 + aarch64）。

流程（每个源文件）：
  clang -O2 -S -emit-llvm      -> <name>.ll
  ir_instrument.py             -> <name>.inst.ll + table.c
  clang -O2 -c <name>.inst.ll  -> .o
  link + phase_profiler_ir.o + table.o

用法示例：
  python3 scripts/phase_ir_build.py --name gemm --outdir build/phase_ir \
      --cflag -DMEDIUM_DATASET --cflag -Ibenchmarks/polybench/utilities \
      --src benchmarks/polybench/.../gemm.c \
      --src benchmarks/polybench/utilities/polybench.c --ldflag -lm
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
CLEAN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
CC = "clang-18"
CXX = "clang++-18"


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, env=CLEAN_ENV,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("命令失败: %s\n%s" % (" ".join(cmd),
                                                 r.stderr.strip()[-1500:]))
    return r


def build_target(name, srcs, outdir, tag, target_flags, cflags, ldflags,
                 is_cxx=False, cwd=None):
    # 只重建源平台二进制时跳过 ARM（富特征采集只需重跑 x86 侧）
    if tag == "arm" and os.environ.get("PHASE_BUILD_X86_ONLY") == "1":
        return None
    ir_dir = os.path.join(outdir, tag)
    os.makedirs(ir_dir, exist_ok=True)
    link_cc = CXX if is_cxx else CC

    # 1. 生成优化后的 IR。混合工程中的 .c 仍按 C 编译，不能因最终使用
    # C++ 链接器就交给 clang++（会把隐式 int -> enum 转换等当成错误）。
    lls = []
    compilers = []
    for s in srcs:
        src_cxx = s.endswith(".C") or s.lower().endswith((".cc", ".cpp", ".cxx", ".c++"))
        cc = CXX if src_cxx else CC
        src_cflags = cflags
        if is_cxx and not src_cxx:
            src_cflags = [f for f in cflags if not f.startswith("-std=")]
            src_cflags.append("-std=gnu99")
        base = os.path.splitext(os.path.basename(s))[0]
        ll = os.path.join(ir_dir, base + ".ll")
        run([cc] + target_flags + src_cflags +
            ["-O2", "-S", "-emit-llvm", "-o", ll, s], cwd=cwd)
        lls.append(ll)
        compilers.append(cc)

    # 2. IR 插桩 + 静态表
    table = os.path.join(ir_dir, name + "_table.c")
    inst_out = os.path.join(ir_dir, "inst")
    r = run([sys.executable, os.path.join(SCRIPTS, "ir_instrument.py"),
             "--outdir", inst_out, "--table", table] + lls)
    inst_lls = [x for x in r.stdout.split() if x.endswith(".inst.ll")]

    # 3. 编译插桩后的 IR
    objs = []
    for ll, cc in zip(inst_lls, compilers):
        obj = ll[:-len(".inst.ll")] + ".inst.o"
        run([cc] + target_flags + ["-O2", "-w", "-c", ll, "-o", obj])
        objs.append(obj)

    # 4. 运行时与静态表（不加插桩）
    rt_src = os.path.join(SCRIPTS, "phase_profiler_ir.c")
    rt_obj = os.path.join(ir_dir, "prof_ir.o")
    run([CC] + target_flags + ["-O2", "-w", "-I" + SCRIPTS,
                               "-c", rt_src, "-o", rt_obj])
    tbl_obj = os.path.join(ir_dir, "table.o")
    run([CC] + target_flags + ["-O2", "-w", "-I" + SCRIPTS,
                               "-c", table, "-o", tbl_obj])

    # 5. 链接
    exe = os.path.join(outdir, tag, name)
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    run([link_cc] + target_flags + objs + [rt_obj, tbl_obj] + ldflags + ["-o", exe])
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--src", action="append", default=[], required=True)
    ap.add_argument("--cflag", action="append", default=[])
    ap.add_argument("--ldflag", action="append", default=[])
    ap.add_argument("--cxx", action="store_true")
    args = ap.parse_args()

    srcs = [s if os.path.isabs(s) else os.path.join(ROOT, s)
            for s in args.src]
    for s in srcs:
        if not os.path.isfile(s):
            print("缺少源文件 %s" % s)
            return 1

    for tag, tflags in (("x86", []),
                        ("arm", ["--target=aarch64-linux-gnu", "-static"])):
        try:
            exe = build_target(args.name, srcs, args.outdir, tag, tflags,
                               args.cflag, args.ldflag, args.cxx)
            print("[%s] OK %s" % (tag, exe))
        except RuntimeError as e:
            print("[%s] 失败\n%s" % (tag, e))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

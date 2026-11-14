#!/usr/bin/env python3
"""用 clang-18 编译带 phase 插桩的 SPEC CPU2017 程序（x86 + aarch64）。

思路：
  SPEC 的 runcpu 已把每个 benchmark 需要的源码复制到
  benchspec/CPU/<bench>/build/build_base_d2000-64.0000/ 下，并生成
  Makefile.spec（含 SOURCES / BENCH_CFLAGS / EXTRA_PORTABILITY 等）。
  这里直接解析该文件，用 clang 重编每个翻译单元（-fsanitize-coverage=trace-pc），
  再与 phase_profiler.o 链接，得到可用于 phase 采集的二进制。

只处理 C / C++ benchmark（Fortran 需 flang，本机不可用）。

用法:
  python3 scripts/spec_phase_build.py [bench ...]
"""

import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = f"{ROOT}/benchmarks/cpu2017"
OUT = f"{ROOT}/build/phase_spec"
CC = "clang-18"
BUILD_SUBDIR = "build_base_d2000-64.0000"

DEFAULT_BENCHES = [
    "505.mcf_r",
    "519.lbm_r",
    "531.deepsjeng_r",
    "541.leela_r",
    "544.nab_r",
    "557.xz_r",
]

CLEAN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def _get_var(txt, name):
    """取 Makefile 变量值，支持反斜杠续行。"""
    m = re.search(rf"^{name}\s*=\s*((?:.*\\\n)*.*)$", txt, re.M)
    if not m:
        return ""
    return m.group(1).replace("\\\n", " ").strip()


def parse_makefile(path):
    """从 Makefile.spec 抽取 build 需要的变量。"""
    txt = open(path).read()
    info = {}

    srcs = _get_var(txt, "SOURCES").split()
    if not srcs:
        return None
    info["sources"] = srcs

    # SPEC 生成的文件里两种名字都出现过
    info["BENCH_CFLAGS"] = " ".join(
        x for x in (_get_var(txt, "BENCH_CFLAGS"),
                    _get_var(txt, "BENCH_FLAGS")) if x)
    for key in ("EXTRA_COPTIMIZE", "EXTRA_PORTABILITY",
                "OPTIMIZE", "NEED_MATH", "BENCHLANG"):
        info[key] = _get_var(txt, key)
    return info


def build_one(bench, info, tag, clang_args, prof_obj):
    bld = f"{SPEC}/benchspec/CPU/{bench}/build/{BUILD_SUBDIR}"
    srcs = [os.path.join(bld, s) for s in info["sources"]]
    missing = [s for s in srcs if not os.path.isfile(s)]
    if missing:
        return f"缺少源码 {missing[:2]}"

    is_cpp = info["BENCHLANG"].upper().startswith("CXX")
    std = "-std=c++03" if is_cpp else "-std=gnu99"

    common = [std, "-O2", "-w", "-DSPEC", "-DNDEBUG", "-DSPEC_LP64"]
    if info["EXTRA_PORTABILITY"]:
        common.append(info["EXTRA_PORTABILITY"])
    if info["BENCH_CFLAGS"]:
        common += info["BENCH_CFLAGS"].split()
    if info["EXTRA_COPTIMIZE"]:
        # -fgnu89-inline 只对 C 有效，C++ 会直接报错
        skip = ("-fno-strict-aliasing",)
        if is_cpp:
            skip += ("-fgnu89-inline",)
        common += [f for f in info["EXTRA_COPTIMIZE"].split()
                   if f and not f.startswith(skip)]

    objs = []
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    for src in srcs:
        obj = f"{OUT}/{tag}/{bench}_{os.path.basename(src)}.o"
        cmd = [CC] + clang_args + common + \
              ["-fsanitize-coverage=trace-pc", "-c", src, "-o", obj]
        r = subprocess.run(cmd, cwd=bld, env=CLEAN_ENV,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"编译失败 {os.path.basename(src)}: {r.stderr.strip()[-300:]}"
        objs.append(obj)

    exe = f"{OUT}/{tag}/{bench}"
    link_lang = "clang++-18" if is_cpp else CC
    cmd = [link_lang] + clang_args + objs + [prof_obj, "-o", exe]
    if info["NEED_MATH"] == "yes":
        cmd.append("-lm")
    if is_cpp:
        cmd += ["-lstdc++", "-lm"]
    r = subprocess.run(cmd, cwd=bld, env=CLEAN_ENV,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"链接失败: {r.stderr.strip()[-300:]}"
    return None


def main():
    benches = sys.argv[1:] or DEFAULT_BENCHES
    os.makedirs(f"{OUT}/x86", exist_ok=True)
    os.makedirs(f"{OUT}/arm", exist_ok=True)

    subprocess.run([CC, "-O2", "-c", f"{ROOT}/scripts/phase_profiler.c",
                    "-o", f"{OUT}/prof_x86.o"], env=CLEAN_ENV, check=True)
    subprocess.run([CC, "--target=aarch64-linux-gnu", "-static", "-O2",
                    "-c", f"{ROOT}/scripts/phase_profiler.c",
                    "-o", f"{OUT}/prof_arm.o"], env=CLEAN_ENV, check=True)

    for bench in benches:
        mk = f"{SPEC}/benchspec/CPU/{bench}/build/{BUILD_SUBDIR}/Makefile.spec"
        if not os.path.isfile(mk):
            print(f"[{bench}] 无 Makefile.spec，跳过")
            continue
        info = parse_makefile(mk)
        if not info:
            print(f"[{bench}] 无法解析 Makefile.spec")
            continue
        if info["BENCHLANG"].upper().startswith("F"):
            print(f"[{bench}] Fortran，跳过")
            continue

        e1 = build_one(bench, info, "x86", [], f"{OUT}/prof_x86.o")
        print(f"[{bench} x86] {'OK' if e1 is None else e1}")
        e2 = build_one(bench, info, "arm",
                       ["--target=aarch64-linux-gnu", "-static"],
                       f"{OUT}/prof_arm.o")
        print(f"[{bench} arm] {'OK' if e2 is None else e2}")


if __name__ == "__main__":
    main()

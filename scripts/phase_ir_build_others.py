#!/usr/bin/env python3
"""用 IR 插桩管线构建 MiBench / SD-VBS / SPEC CPU2017（x86 + aarch64）。

与 phase_build_mibench.sh / phase_build_sdvbs.sh / spec_phase_build.py 的
程序清单、编译宏保持一致，只是把 -fsanitize-coverage=trace-pc 换成
「优化后 IR 插桩 + 静态结构表」，使 phase 数据带上指令构成特征。

用法: python3 scripts/phase_ir_build_others.py [mibench|sdvbs|spec|all]
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_ir_build import ROOT, build_target
import spec_phase_build as spb

OUT = os.path.join(ROOT, "build/phase_ir")
TARGETS = (("x86", []), ("arm", ["--target=aarch64-linux-gnu", "-static"]))

MIBENCH = {
    "aes": ("security/rijndael", ["aes.c", "aesxam.c"]),
    "crc": ("telecomm/CRC32", ["crc_32.c"]),
    "dijkstra": ("network/dijkstra", ["dijkstra_large.c"]),
    "fft": ("telecomm/FFT", ["main.c", "fftmisc.c", "fourierf.c"]),
    "patricia": ("network/patricia", ["patricia.c", "patricia_test.c"]),
    "qsort": ("automotive/qsort", ["qsort_large.c"]),
    "sha": ("security/sha", ["sha_driver.c", "sha.c"]),
    "basicmath": ("automotive/basicmath",
                  ["basicmath_large.c", "rad2deg.c", "cubic.c", "isqrt.c"]),
    "bitcount": ("automotive/bitcount",
                 ["bitcnt_1.c", "bitcnt_2.c", "bitcnt_3.c", "bitcnt_4.c",
                  "bitcnts.c", "bitfiles.c", "bitstrng.c", "bstr_i.c"]),
    "susan": ("automotive/susan", ["susan.c"]),
    "blowfish": ("security/blowfish",
                 ["bf.c", "bf_skey.c", "bf_ecb.c", "bf_enc.c", "bf_cbc.c",
                  "bf_cfb64.c", "bf_ofb64.c"]),
    "stringsearch": ("office/stringsearch",
                     ["bmhasrch.c", "bmhisrch.c", "bmhsrch.c", "pbmsrch_large.c"]),
    "adpcm_encode": ("telecomm/adpcm/src", ["rawcaudio.c", "adpcm.c"]),
    "adpcm_decode": ("telecomm/adpcm/src", ["rawdaudio.c", "adpcm.c"]),
    # src 目录恰好只含 GSM 库和 toast 前端，在 do_mibench 中展开。
    "gsm": ("telecomm/gsm", []),
}

SDVBS = ["disparity", "localization", "mser", "multi_ncut", "sift", "stitch",
         "svm", "texture_synthesis", "tracking"]

SPEC_EXTRA = ["500.perlbench_r", "502.gcc_r", "508.namd_r", "510.parest_r",
              "520.omnetpp_r", "523.xalancbmk_r"]


def build(name, srcs, cflags, ldflags=(), cxx=False, cwd=None):
    for tag, tf in TARGETS:
        build_target(name, srcs, OUT, tag, tf, list(cflags), list(ldflags),
                     cxx, cwd)


def do_mibench():
    base = os.path.join(ROOT, "benchmarks/mibench")
    for name, (d, files) in MIBENCH.items():
        src_dir = os.path.join(base, d)
        if name == "gsm":
            srcs = sorted(glob.glob(os.path.join(src_dir, "src/*.c")))
        else:
            srcs = [os.path.join(src_dir, f) for f in files]
        if not all(os.path.isfile(s) for s in srcs):
            print("[%s] 缺源文件，跳过" % name)
            continue
        cflags = ["-std=gnu89", "-w", "-I" + src_dir]
        if name == "gsm":
            cflags += ["-I" + os.path.join(src_dir, "inc"), "-DSASR",
                       "-DSTUPID_COMPILER", "-DNeedFunctionPrototypes=1"]
        try:
            build(name, srcs, cflags, ["-lm"])
            print("[mibench %s] OK" % name)
        except RuntimeError as e:
            print("[mibench %s] 失败\n%s" % (name, str(e)[-500:]))


def do_sdvbs():
    src = os.path.join(ROOT, "benchmarks/SD-VBS")
    common = sorted(glob.glob(os.path.join(src, "common/c/*.c")))
    for name in SDVBS:
        d = os.path.join(src, "benchmarks", name, "src/c")
        files = sorted(glob.glob(os.path.join(d, "*.c")))
        if not files:
            print("[%s] 无源码，跳过" % name)
            continue
        ld = ["-lm"]
        if name == "texture_synthesis":
            ld.append("-Wl,--allow-multiple-definition")
        try:
            build(name, common + files,
                  ["-DGCC", "-Dqcif", "-DCHECK", "-w",
                   "-I" + os.path.join(src, "common/c"), "-I" + d], ld)
            print("[sdvbs %s] OK" % name)
        except RuntimeError as e:
            print("[sdvbs %s] 失败\n%s" % (name, str(e)[-500:]))


def do_spec(benches=None):
    for bench in benches or (spb.DEFAULT_BENCHES + SPEC_EXTRA):
        bld = os.path.join(spb.SPEC, "benchspec/CPU", bench,
                           "build", spb.BUILD_SUBDIR)
        mk = os.path.join(bld, "Makefile.spec")
        if not os.path.isfile(mk):
            print("[%s] 无 Makefile.spec" % bench)
            continue
        info = spb.parse_makefile(mk)
        if not info or info["BENCHLANG"].upper().startswith("F"):
            print("[%s] 跳过" % bench)
            continue
        is_cpp = info["BENCHLANG"].upper().startswith("CXX")
        flags = ["-w", "-DSPEC", "-DNDEBUG", "-DSPEC_LP64"]
        if info["EXTRA_PORTABILITY"]:
            flags += info["EXTRA_PORTABILITY"].split()
        if info["BENCH_CFLAGS"]:
            flags += info["BENCH_CFLAGS"].split()
        skip = ("-fno-strict-aliasing",) + (("-fgnu89-inline",) if is_cpp else ())
        flags += [f for f in info["EXTRA_COPTIMIZE"].split()
                  if f and not f.startswith(skip)]
        srcs = [os.path.join(bld, s) for s in info["sources"]]
        if not all(os.path.isfile(s) for s in srcs):
            print("[%s] 缺源码，跳过" % bench)
            continue
        try:
            build("spec_" + bench.split(".", 1)[1].replace("_r", ""),
                  srcs, flags,
                  ["-lm"] + (["-lstdc++"] if is_cpp else []), is_cpp, bld)
            print("[spec %s] OK" % bench)
        except RuntimeError as e:
            print("[spec %s] 失败\n%s" % (bench, str(e)[-500:]))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("mibench", "all"):
        do_mibench()
    if what in ("sdvbs", "all"):
        do_sdvbs()
    if what in ("spec", "all"):
        do_spec(sys.argv[2:] or None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""采集 phase 级 PMU 数据（PolyBench，x86 本机 / D2000 ARM）。

每个程序运行一次，运行时由 phase_profiler 每累计 PHASE_BB 个基本块读一次
6 个通用硬件计数器，写出 phase.csv。

用法:
  python3 scripts/phase_collect.py --platform x86 --bindir build/phase_pb/x86 \
      --outdir data/phase/x86
  python3 scripts/phase_collect.py --platform arm --bindir build/phase_pb/arm \
      --outdir data/phase/arm
"""

import argparse
import os
import subprocess
import sys

REMOTE = os.environ.get("D2000_USER", "soc") + "@" + os.environ.get("D2000_HOST", "")
RPORT = "2000"
RPASS = os.environ.get("D2000_PASS", "")
RROOT = "/home/soc/PMU_bench/phase"

PROGRAMS = [
    "durbin", "cholesky", "ludcmp", "trisolv", "syrk", "mvt", "atax", "bicg",
    "doitgen", "jacobi-1d", "seidel-2d", "heat-3d", "jacobi-2d", "fdtd-2d", "adi",
    "gemver", "gesummv", "symm", "syr2k", "trmm", "gemm", "3mm", "deriche", "2mm",
    "floyd-warshall", "correlation", "gramschmidt", "lu", "covariance", "nussinov",
]


def clean_env():
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C", "LC_ALL": "C",
    }


def ssh(cmd, timeout=900):
    return subprocess.run(
        ["sshpass", "-p", RPASS, "ssh", "-p", RPORT,
         "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
         REMOTE, cmd],
        capture_output=True, text=True, timeout=timeout)


def scp(src, dst, timeout=900):
    return subprocess.run(
        ["sshpass", "-p", RPASS, "scp", "-P", RPORT,
         "-o", "StrictHostKeyChecking=no", "-r", src, dst],
        capture_output=True, text=True, timeout=timeout)


def scp_glob(bindir, dst, timeout=900):
    """scp 不会展开通配符，用 shell 交给远端/本地 shell 处理。"""
    return subprocess.run(
        ["bash", "-c",
         f"sshpass -p {RPASS} scp -P {RPORT} -o StrictHostKeyChecking=no -r "
         f"{bindir}/* {dst}"],
        capture_output=True, text=True, timeout=timeout)


def run_x86(bindir, outdir, prog, phase_bb):
    out = os.path.join(outdir, f"{prog}.csv")
    if os.path.exists(out):
        return out, "skip"
    env = clean_env()
    env["PHASE_OUT"] = out
    env["PHASE_BB"] = str(phase_bb)
    exe = os.path.join(bindir, prog)
    with open(os.devnull, "w") as dn:
        r = subprocess.run(["taskset", "-c", "0", exe], env=env,
                           stdout=dn, stderr=subprocess.PIPE, timeout=900)
    return out, f"rc={r.returncode}"


def run_arm(bindir, outdir, phase_bb, only):
    # 1) 上传全部二进制（一次性）
    r = ssh(f"mkdir -p {RROOT}")
    if r.returncode != 0:
        sys.exit(f"远端建目录失败: {r.stderr}")
    r = scp_glob(bindir, f"{REMOTE}:{RROOT}/")
    if r.returncode != 0:
        sys.exit(f"scp 上传失败: {r.stderr}")

    # 2) 远端逐个运行
    todo = [p for p in PROGRAMS if p in only] if only else PROGRAMS
    names = " ".join(todo)
    remote = (
        f"cd {RROOT} && chmod +x * && "
        f"for p in {names}; do "
        f"  PHASE_OUT=/tmp/$p.csv PHASE_BB={phase_bb} taskset -c 0 ./$p >/dev/null 2>&1; "
        f"  echo \"$p done\"; "
        f"done"
    )
    r = ssh(remote, timeout=3600)
    print(r.stdout.strip())

    # 3) 下载 CSV
    os.makedirs(outdir, exist_ok=True)
    for p in todo:
        rr = scp(f"{REMOTE}:/tmp/{p}.csv", os.path.join(outdir, f"{p}.csv"))
        if rr.returncode != 0:
            print(f"  [下载失败] {p}: {rr.stderr.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["x86", "arm"], required=True)
    ap.add_argument("--bindir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase-bb", type=int, default=5000)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.platform == "arm":
        run_arm(args.bindir, args.outdir, args.phase_bb, args.only)
        return

    todo = [p for p in PROGRAMS if p in args.only] if args.only else PROGRAMS
    for i, p in enumerate(todo, 1):
        try:
            out, st = run_x86(args.bindir, args.outdir, p, args.phase_bb)
            print(f"[{i}/{len(todo)}] {p}: {st}")
        except subprocess.TimeoutExpired:
            print(f"[{i}/{len(todo)}] {p}: 超时")


if __name__ == "__main__":
    main()

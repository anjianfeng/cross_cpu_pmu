#!/usr/bin/env python3
"""采集 SPEC CPU2017 程序的 phase 数据（test 数据集）。

SPEC 已经为每个 benchmark 生成好运行目录
  benchspec/CPU/<bench>/run/run_base_test_d2000-64.0000/
其中包含输入文件与 speccmds.cmd（记录命令行）。本脚本解析该命令行，
在运行目录里用我们的 phase 插桩二进制替换原程序执行。

用法:
  python3 scripts/spec_phase_collect.py --platform x86 --phase-bb 5000
  python3 scripts/spec_phase_collect.py --platform arm --phase-bb 5000
"""

import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import tempfile

from phase_ir_llvm import RPASS, RPORT, REMOTE, RROOT, ssh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "benchmarks/cpu2017")
BINDIR = os.path.join(ROOT, "build/phase_spec")
DATA = os.path.join(ROOT, "data/phase")
RUNSUB = "run_base_test_d2000-64.0000"

BENCHES = ["505.mcf_r", "519.lbm_r", "531.deepsjeng_r",
           "541.leela_r", "544.nab_r", "557.xz_r",
           "500.perlbench_r", "502.gcc_r", "508.namd_r", "510.parest_r",
           "520.omnetpp_r", "523.xalancbmk_r"]

# 程序名（CSV 名）用去掉编号与后缀的短名
def short(bench):
    return bench.split(".", 1)[1].replace("_r", "").replace("_s", "")


def run_args(bench, rundir):
    """从 speccmds.cmd 里解析出真正的程序参数。"""
    cmd_file = os.path.join(rundir, "speccmds.cmd")
    if not os.path.isfile(cmd_file):
        return None
    for line in open(cmd_file):
        line = line.strip()
        if "_base." not in line or line.startswith("-E "):
            continue
        toks = line.split()
        for i, t in enumerate(toks):
            if "_base." in t and "run_base" in t:
                rest = toks[i + 1:]
                out = []
                for t in rest:
                    if t in (">", "2>>", "2>", ">>"):
                        break
                    out.append(t)
                return out
    return None


def run_one_arm(exe, name, rundir, run_args, out, phase_bb):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="phase_ir_spec_arm_") as tmp:
        bindir = os.path.join(tmp, "bin")
        os.mkdir(bindir)
        shutil.copy2(exe, os.path.join(bindir, name))
        rel_run = os.path.relpath(rundir, root)
        tar = subprocess.Popen(
            ["tar", "-czf", "-", "-C", root, rel_run, "-C", tmp, "bin"],
            stdout=subprocess.PIPE)
        try:
            r = ssh("mkdir -p %s/data %s/out && tar -xzf - -C %s" %
                    (RROOT, RROOT, RROOT), timeout=1800, stdin=tar.stdout)
        finally:
            tar.stdout.close()
            tar_rc = tar.wait()
        if tar_rc != 0 or r.returncode != 0:
            return None, 0, "上传失败: " + r.stderr[-180:]

    remote_dir = rundir.replace(root, RROOT, 1)
    rout = "%s/data/%s.csv" % (RROOT, name)
    rerr = "%s/out/%s.err" % (RROOT, name)
    inner = "exec taskset -c 0 %s %s" % (
        shlex.quote("%s/bin/%s" % (RROOT, name)),
        " ".join(shlex.quote(a) for a in run_args))
    cmd = ("cd %s && PHASE_OUT=%s PHASE_BB=%d /bin/sh -c %s "
           ">/dev/null 2>%s") % (shlex.quote(remote_dir), rout, phase_bb,
                                  shlex.quote(inner), rerr)
    try:
        r = ssh(cmd, timeout=3630)
        rc, err = r.returncode, r.stderr.strip()[-180:]
    except subprocess.TimeoutExpired:
        rc, err = None, "timeout"
    rr = subprocess.run(
        ["sshpass", "-p", RPASS, "scp", "-P", RPORT,
         "-o", "StrictHostKeyChecking=no", "%s:%s" % (REMOTE, rout), out],
        capture_output=True, text=True, timeout=900)
    if rr.returncode != 0:
        err = (err + " " + rr.stderr.strip()[-180:]).strip()
    rows = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else 0
    return rc, rows, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["x86", "arm"], required=True)
    ap.add_argument("--phase-bb", type=int, default=5000)
    ap.add_argument("--bindir", default=BINDIR)
    ap.add_argument("--outdir", default=DATA)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    outdir = os.path.join(args.outdir, args.platform)
    os.makedirs(outdir, exist_ok=True)
    bindir = os.path.abspath(os.path.join(args.bindir, args.platform))

    benches = args.only or BENCHES
    for bench in benches:
        name = short(bench)
        out = os.path.join(outdir, f"spec_{name}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 50:
            print(f"[{name}] 已有数据，跳过")
            continue
        exe = os.path.join(bindir, bench)
        if not os.path.isfile(exe):          # IR 插桩版按 spec_<短名> 命名
            exe = os.path.join(bindir, "spec_" + name)
        if not os.path.isfile(exe):
            print(f"[{name}] 缺二进制 {exe}")
            continue
        rundirs = glob.glob(f"{SPEC}/benchspec/CPU/{bench}/run/{RUNSUB}")
        if not rundirs:
            rundirs = sorted(glob.glob(
                f"{SPEC}/benchspec/CPU/{bench}/run/run_base_test_*"))
        if not rundirs:
            print(f"[{name}] 缺运行目录")
            continue
        rundir = rundirs[0]
        a = run_args(bench, rundir)
        if a is None:
            print(f"[{name}] 解析命令行失败")
            continue

        if args.platform == "arm":
            rc, rows, err = run_one_arm(exe, "spec_" + name, rundir, a,
                                        out, args.phase_bb)
            print(f"[{name}] rc={rc} rows={rows} 参数={' '.join(a)}")
            if err:
                print(f"   {err}")
            continue

        env = {"PATH": "/usr/bin:/bin",
               "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu",
               "LANG": "C", "LC_ALL": "C",
               "PHASE_OUT": os.path.abspath(out),
               "PHASE_BB": str(args.phase_bb)}
        try:
            r = subprocess.run(["taskset", "-c", "0", exe] + a, cwd=rundir,
                               env=env, capture_output=True, text=True,
                               timeout=3600)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            print(f"[{name}] 超时")
            continue
        rows = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else 0
        print(f"[{name}] rc={rc} rows={rows} 参数={' '.join(a)}")
        if rc != 0:
            print(f"   stderr: {r.stderr.strip()[-300:]}")


if __name__ == "__main__":
    main()

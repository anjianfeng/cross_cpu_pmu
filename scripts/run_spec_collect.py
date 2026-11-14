#!/usr/bin/env python3
"""SPEC CPU2017 两平台 PMU 采集驱动。

读取 spec_setup.py 生成的 manifest，逐程序调用 collect_pmu.py：
  - x86：本地执行，工作目录为该基准的 run 目录
  - arm：把 run 目录同步到 D2000，通过 ssh 远程执行同款脚本

为什么逐程序顺序执行：早期用外层 shell 循环批量驱动时测量值严重失真
（见 experiment_log.md 第八节），改为逐程序调用后数据自洽。
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

PMU_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECT = os.path.join(PMU_ROOT, "scripts", "collect_pmu.py")
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=15"]
D2000 = os.environ.get("D2000_USER", "soc") + "@" + os.environ.get("D2000_HOST", "")
D2000_PORT = os.environ.get("D2000_PORT", "22")
D2000_PASS = os.environ.get("D2000_PASS", "")
D2000_ROOT = "/home/soc/PMU_bench"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def sync_arm_rundir(bench, rundir):
    """把 run 目录同步到 D2000，返回远端目录路径。"""
    remote = f"{D2000_ROOT}/spec/{bench}"
    sh(["sshpass", "-p", D2000_PASS, "ssh"] + SSH_OPTS + ["-p", D2000_PORT, D2000,
        f"rm -rf {remote} && mkdir -p {remote}"], check=True)
    r = sh(["sshpass", "-p", D2000_PASS, "scp"] + SSH_OPTS +
           ["-P", D2000_PORT, "-r", rundir + "/.", f"{D2000}:{remote}/"],
           capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] scp 失败: {r.stderr[-200:]}")
        return None
    return remote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["x86", "arm"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--events-file", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="只采指定基准")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    if args.only:
        manifest = [e for e in manifest if e["bench"] in args.only]
    os.makedirs(args.outdir, exist_ok=True)

    for i, e in enumerate(manifest, 1):
        bench = e["bench"]
        print(f"\n=== [{args.platform}] ({i}/{len(manifest)}) {bench} ===", flush=True)
        out_csv = os.path.abspath(os.path.join(args.outdir, f"{bench}_{args.platform}.csv"))
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print("  已存在，跳过")
            continue

        if args.platform == "x86":
            cmd = [sys.executable, COLLECT, "--platform", "x86",
                   "--name", bench, "--cwd", e["rundir"],
                   "--events-file", os.path.abspath(args.events_file),
                   "--out", out_csv, "--reps", str(args.reps),
                   "--cmd", e["binary"]] + e["args"]
            r = sh(cmd)
            if r.returncode != 0:
                print(f"  [FAIL] {bench}")
        else:
            remote_dir = sync_arm_rundir(bench, e["rundir"])
            if not remote_dir:
                continue
            remote_csv = f"{D2000_ROOT}/data/spec/{bench}_arm.csv"
            remote_ev = f"{D2000_ROOT}/events_arm.txt"
            inner = ("cd %s && mkdir -p %s/data/spec && python3 %s/collect_pmu.py "
                     "--platform arm --name %s --cwd %s --events-file %s "
                     "--out %s --reps %d --cmd %s") % (
                D2000_ROOT, D2000_ROOT, D2000_ROOT, shlex.quote(bench),
                shlex.quote(remote_dir), remote_ev, remote_csv, args.reps,
                " ".join([shlex.quote(e["binary"])] + [shlex.quote(a) for a in e["args"]]))
            r = sh(["sshpass", "-p", D2000_PASS, "ssh"] + SSH_OPTS +
                   ["-p", D2000_PORT, D2000, inner])
            if r.returncode != 0:
                print(f"  [FAIL] {bench}")
            else:
                sh(["sshpass", "-p", D2000_PASS, "scp"] + SSH_OPTS +
                   ["-P", D2000_PORT, f"{D2000}:{remote_csv}", out_csv])

    print("\n全部完成")


if __name__ == "__main__":
    main()

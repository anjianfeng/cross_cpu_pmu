#!/usr/bin/env python3
"""为 SPEC CPU2017 生成采集清单。

对每个基准：
  1. 若 run 目录不存在，用 runcpu --action=setup 铺出（只复制输入，不执行）
  2. 用 specinvoke -n 取得实际命令行
  3. 解析出 (run目录, 可执行文件名, 参数列表)，写入 manifest JSON

manifest 供 run_spec_collect.sh 使用，x86 与 ARM 各一份。
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

SPEC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks/cpu2017")
SPEC_BIN = os.path.join(SPEC_ROOT, "bin")
BENCHES = [
    "500.perlbench_r", "502.gcc_r", "505.mcf_r", "520.omnetpp_r", "523.xalancbmk_r",
    "531.deepsjeng_r", "541.leela_r", "557.xz_r",
    "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r", "526.blender_r",
    "538.imagick_r", "544.nab_r",
]


def run(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def setup_bench(bench, label, size, config):
    """确保 run 目录存在。"""
    bdir = os.path.join(SPEC_ROOT, "benchspec", "CPU", bench, "run")
    pattern = f"run_base_{size}_{label}.0000"
    if os.path.isdir(os.path.join(bdir, pattern)):
        return os.path.join(bdir, pattern)
    env = dict(os.environ)
    env["PATH"] = SPEC_BIN + ":" + env.get("PATH", "/usr/bin:/bin")
    r = run([os.path.join(SPEC_BIN, "runcpu"), f"--config={config}",
             "--action=setup", f"--size={size}", "--tune=base", bench],
            cwd=SPEC_ROOT, env=env)
    if os.path.isdir(os.path.join(bdir, pattern)):
        return os.path.join(bdir, pattern)
    print(f"  [WARN] {bench} setup 失败: {r.stdout[-300:]}{r.stderr[-300:]}")
    return None


def parse_cmd(rundir):
    """用 specinvoke -n 取命令行，返回 (binary, args) 或 None。"""
    r = run([os.path.join(SPEC_BIN, "specinvoke"), "-n"], cwd=rundir)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("specinvoke exit"):
            continue
        # 形如: ../run_base_x.0000/foo_base.x  arg1 > out 2>> err
        cmdpart = line.split(">")[0].strip()
        try:
            toks = shlex.split(cmdpart)
        except ValueError:
            continue
        if not toks:
            continue
        binary = "./" + os.path.basename(toks[0])
        return binary, toks[1:]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run 目录标签，如 d2000-64 / x86n-64")
    ap.add_argument("--config", required=True, help="SPEC config 文件名，如 d2000 / x86")
    ap.add_argument("--size", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", nargs="*", default=BENCHES)
    args = ap.parse_args()

    manifest = []
    for b in args.bench:
        print(f"[{b}]")
        rundir = setup_bench(b, args.label, args.size, args.config)
        if not rundir:
            continue
        parsed = parse_cmd(rundir)
        if not parsed:
            print(f"  [WARN] 无法解析命令行，跳过")
            continue
        binary, cargs = parsed
        if not os.access(os.path.join(rundir, binary), os.X_OK):
            print(f"  [WARN] {binary} 不可执行，跳过")
            continue
        manifest.append({"bench": b, "rundir": rundir,
                         "binary": binary, "args": cargs})
        print(f"  OK  {rundir}  ->  {binary} {' '.join(cargs)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n共 {len(manifest)} 个基准 -> {args.out}")


if __name__ == "__main__":
    main()

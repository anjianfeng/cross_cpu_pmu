#!/usr/bin/env python3
"""x86 富特征 phase 采集（LACross 的 host 特征扩展）。

背景：D2000 只能打开 6 个通用硬件事件，但**特征只需要在源平台可测**，
目标平台只提供标签。因此可以在 x86 侧按 phase 采集 topdown、L2/LLC、
TLB 等微架构事件，ARM 侧沿用已有数据。

做法：
  1. 复用 data/phase_ir/x86 里已有的基础 6 事件轨迹（不重跑）；
  2. 用 --groups 指定的若干事件组分别重跑程序，每组一趟（论文在 Intel
     上同样因计数器限制跑了 5 趟）；
  3. 各趟按 phase 编号对齐后合并到一份 CSV，写入 data/phase_ir_rich/x86；
  4. data/phase_ir_rich/arm 指向原 ARM 数据，方便 analyze_phase.py 直接使用。

用法：
  python3 scripts/phase_collect_rich.py --only gemm qsort spec_mcf
  python3 scripts/phase_collect_rich.py                 # 全量
  python3 scripts/phase_collect_rich.py --list
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase_ir_llvm as pil
from phase_collect_others import CMDS, CWD, STDIN
from spec_phase_collect import BENCHES, RUNSUB, short, run_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_X86 = os.path.join(ROOT, "data/phase_ir/x86")
BASE_ARM = os.path.join(ROOT, "data/phase_ir/arm")
OUT = os.path.join(ROOT, "data/phase_ir_rich")
X86_OUT = os.path.join(OUT, "x86")
RAW_OUT = os.path.join(OUT, "raw")
LLVM_BIN = os.path.join(ROOT, "build/phase_ir_llvm/x86")
OTHER_BIN = os.path.join(ROOT, "build/phase_ir/x86")

# 第一组：topdown 主分类 + 各级缓存/TLB miss
# 第二组：topdown 细分（mem-bound 等）+ 指令侧/存储侧补充
BASE_EVENTS = ["cycles", "instructions", "cache_refs", "cache_misses",
               "branches", "branch_misses"]

# topdown 指标事件必须挂在 slots 事件所在组里（slots 作组长），
# 否则 perf_event_open 返回 EINVAL；这也是 perf stat 的实际做法。
DEFAULT_GROUPS = [
    ["slots", "cycles", "instructions",
     "topdown-retiring", "topdown-bad-spec", "topdown-fe-bound",
     "topdown-be-bound",
     "L1-dcache-load-misses", "LLC-load-misses",
     "dTLB-load-misses", "iTLB-load-misses"],
    ["slots", "cycles", "instructions",
     "topdown-mem-bound", "topdown-br-mispredict", "topdown-heavy-ops",
     "topdown-fetch-lat", "L1-icache-load-misses", "mem-stores"],
]


def clean_env():
    return {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": "C", "LC_ALL": "C", "TMPDIR": "/tmp"}


def manifest():
    """name -> {exe, cwd, args, stdin}；只保留已有基础 x86 数据的程序。"""
    m = {}
    try:
        for p in pil.discover():
            # run_options 里可能带 shell 重定向（如 "< inputs"），保留原文
            m[p["name"]] = dict(exe=os.path.join(LLVM_BIN, p["name"]),
                                cwd=p["cwd"],
                                cmdline=p["run_options"],
                                stdin=None)
    except Exception as e:                       # 清单解析失败不影响其它套件
        print("[manifest] LLVM 发现失败: %s" % e)

    for name in CMDS:
        m[name] = dict(exe=os.path.join(OTHER_BIN, name), cwd=CWD[name],
                       cmdline=" ".join(shlex.quote(a) for a in CMDS[name]),
                       stdin=STDIN.get(name))

    for bench in BENCHES:
        name = "spec_" + short(bench)
        rundirs = glob.glob(
            os.path.join(ROOT, "benchmarks/cpu2017/benchspec/CPU", bench,
                         "run", RUNSUB))
        if not rundirs:
            rundirs = sorted(glob.glob(os.path.join(
                ROOT, "benchmarks/cpu2017/benchspec/CPU", bench,
                "run/run_base_test_*")))
        if not rundirs:
            continue
        args = run_args(bench, rundirs[0])
        if args is None:
            continue
        m[name] = dict(exe=os.path.join(OTHER_BIN, name), cwd=rundirs[0],
                       cmdline=" ".join(shlex.quote(a) for a in args),
                       stdin=None)

    # PolyBench 及其它直接以程序名命名的构建产物（无参数、cwd 在仓库根）
    for xf in sorted(glob.glob(os.path.join(BASE_X86, "*.csv"))):
        name = os.path.basename(xf)[:-4]
        if name in m:
            continue
        m[name] = dict(exe=os.path.join(OTHER_BIN, name), cwd=ROOT,
                       cmdline="", stdin=None)

    return {k: v for k, v in m.items()
            if os.path.isfile(os.path.join(BASE_X86, k + ".csv"))
            and os.path.isfile(v["exe"])}


def run_pass(entry, out, events, phase_bb, timeout):
    env = clean_env()
    env.update({"PHASE_OUT": os.path.abspath(out),
                "PHASE_BB": str(phase_bb),
                "PHASE_EVENTS": ",".join(events)})
    inner = "exec taskset -c 0 %s %s" % (
        shlex.quote(entry["exe"]), entry.get("cmdline", ""))
    if entry.get("stdin"):
        inner += " < " + shlex.quote(entry["stdin"])
    cmd = "/bin/sh -c " + shlex.quote(inner)
    try:
        r = subprocess.run(["/bin/sh", "-c", cmd], cwd=entry["cwd"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    rows = sum(1 for _ in open(out)) - 1 if os.path.isfile(out) else 0
    return rc, rows


def merge(base_csv, pass_csvs):
    base = pd.read_csv(base_csv)
    out = base.copy()
    notes = []
    for path in pass_csvs:
        d = pd.read_csv(path)
        if len(d) != len(base):
            return None, ["phase 行数不一致 base=%d pass=%d" %
                          (len(base), len(d))]
        if not (d["phase"].values == base["phase"].values).all():
            return None, ["phase 编号错位"]
        if "instructions" in d and base["instructions"].sum() > 0:
            # phase 边界由基本块数决定，行数一致即对齐；指令总量会因
            # 每趟读取的事件数不同而带上不同的剖析器自身开销（~1%），
            # 只作为粗检，偏差超过 10% 才认为执行流不一致。
            ratio = d["instructions"].sum() / base["instructions"].sum()
            if abs(ratio - 1.0) > 0.10:
                return None, ["指令总量偏差 %.3f" % ratio]
            if abs(ratio - 1.0) > 0.02:
                notes.append("指令总量偏差 %.1f%%(剖析开销)" %
                             ((ratio - 1) * 100))
        drop = [c for c in d.columns
                if c in ("phase", "cycles", "instructions", "t_enabled",
                         "t_running") or c.startswith("s_")]
        add = d.drop(columns=drop)
        for c in add.columns:
            if add[c].sum() == 0:
                notes.append("%s 全零" % c)
        out = pd.concat([out, add], axis=1)
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="不采集的程序（如 multi_ncut 这类超长程序）")
    ap.add_argument("--phase-bb", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--groups", help="JSON 事件组列表，默认内置两组")
    ap.add_argument("--rebase", action="store_true",
                    help="对齐失败时重采一趟基础 6 事件作为基准")
    args = ap.parse_args()

    groups = json.loads(args.groups) if args.groups else DEFAULT_GROUPS
    m = manifest()
    names = sorted(m)
    if args.only:
        names = [n for n in names if n in set(args.only)]
    if args.exclude:
        names = [n for n in names if n not in set(args.exclude)]
    if args.list:
        for n in names:
            print("%-34s %s %s" % (n, m[n]["exe"], m[n]["cmdline"]))
        print("共 %d 个程序，%d 个事件组" % (len(names), len(groups)))
        return 0
    if not names:
        print("没有可采集的程序")
        return 1

    os.makedirs(X86_OUT, exist_ok=True)
    os.makedirs(RAW_OUT, exist_ok=True)
    arm_link = os.path.join(OUT, "arm")
    if not os.path.exists(arm_link):
        os.symlink(BASE_ARM, arm_link)

    status_path = os.path.join(OUT, "rich_status.json")
    status = {}
    if os.path.isfile(status_path):
        status = json.load(open(status_path))

    for i, name in enumerate(names, 1):
        dst = os.path.join(X86_OUT, name + ".csv")
        if args.skip_existing and os.path.isfile(dst) and \
                os.path.getsize(dst) > 50:
            print("[%d/%d] %s 已有合并数据" % (i, len(names), name), flush=True)
            continue
        rawdir = os.path.join(RAW_OUT, name)
        os.makedirs(rawdir, exist_ok=True)
        pass_csvs, infos = [], []
        for gi, events in enumerate(groups, 1):
            out = os.path.join(rawdir, "g%d.csv" % gi)
            rc, rows = run_pass(m[name], out, events, args.phase_bb,
                                args.timeout)
            # 计数器未复用时 time_running/time_enabled ≈ 1
            scale = 1.0
            if rows > 0:
                d = pd.read_csv(out)
                if "t_enabled" in d and d["t_enabled"].sum() > 0:
                    scale = float(d["t_running"].sum() / d["t_enabled"].sum())
            infos.append(dict(group=gi, rc=str(rc), rows=rows,
                              running_ratio=round(scale, 4)))
            print("[%d/%d] %s g%d %s rows=%d running_ratio=%.3f" %
                  (i, len(names), name, gi, rc, rows, scale), flush=True)
            if rows > 0:
                pass_csvs.append(out)

        if len(pass_csvs) != len(groups):
            status[name] = {"status": "incomplete", "passes": infos}
        else:
            base_csv = os.path.join(BASE_X86, name + ".csv")
            merged, notes = merge(base_csv, pass_csvs)
            if merged is None and args.rebase:
                # 旧基础数据可能是不同构建产生的，用当前二进制补采一趟
                rebase = os.path.join(rawdir, "base.csv")
                rc, rows = run_pass(m[name], rebase,
                                    BASE_EVENTS, args.phase_bb, args.timeout)
                print("    -> 重采基准 rc=%s rows=%d" % (rc, rows), flush=True)
                if rows > 0:
                    base_csv = rebase
                    merged, notes = merge(base_csv, pass_csvs)
            if merged is None:
                status[name] = {"status": "misaligned", "notes": notes,
                                "passes": infos}
                print("    -> 对齐失败: %s" % "; ".join(notes), flush=True)
            else:
                merged.to_csv(dst, index=False)
                status[name] = {"status": "ok", "rows": len(merged),
                                "cols": len(merged.columns),
                                "passes": infos, "notes": notes}
                print("    -> 合并完成 cols=%d%s" %
                      (len(merged.columns),
                       ("; " + "; ".join(notes)) if notes else ""),
                      flush=True)
                if not args.keep_raw:
                    for p in pass_csvs:
                        os.remove(p)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)

    ok = sum(1 for v in status.values() if v.get("status") == "ok")
    print("完成：成功 %d / 目标 %d，状态见 %s" % (ok, len(names), status_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

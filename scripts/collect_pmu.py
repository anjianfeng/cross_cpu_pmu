#!/usr/bin/env python3
"""
PMU 全事件分组采集脚本（LACross 复现实验）

用途：在被测平台上枚举 perf 支持的全部 PMU 事件，按物理计数器上限分组，
对被测程序逐组采集，最终合并为宽表 CSV（每行 = 一次运行的全部事件值）。

设计要点（见 doc/experiment_plan.md 3.2 节）：
- 事件数超过物理计数器时 perf 会时分复用产生缩放估计值，故按组上限切分
- 每组必带锚点事件 cycles + instructions（固定计数器不占可编程名额），
  用于跨组一致性校验与数据对齐
- x86 (i9-12900)：绑 P-core（taskset -c 0，逻辑 CPU 0-15），事件加 cpu_core/ 前缀
- ARM (D2000)：绑核 taskset -c 0，事件加 armv8_pmuv3_0/ 前缀或用通用事件名

用法示例：
  python3 collect_pmu.py --platform x86 --name qsort_large \
      --cmd "./qsort_large input_large.dat" --stdout-null \
      --out data/qsort_x86.csv --reps 3
  python3 collect_pmu.py --platform arm --name qsort_large \
      --cmd "./qsort_large input_large.dat" --stdout-null \
      --out data/qsort_arm.csv --reps 3
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

# 物理可编程计数器上限之外的余量：每组事件数（不含固定计数器锚点）
GROUP_SIZE = {"x86": 10, "arm": 6}  # 保守取值：x86 实测>=7 可用，arm 实测 6 上限
ANCHOR = {
    "x86": ["cpu_core/cycles/", "cpu_core/instructions/"],
    "arm": ["armv8_pmuv3_0/cpu_cycles/", "armv8_pmuv3_0/inst_retired/"],
}
PMU_PREFIX = {"x86": "cpu_core/", "arm": "armv8_pmuv3_0/"}
# 已知不支持/不稳定事件排除清单（按子串匹配）
EXCLUDE_SUBSTR = [
    "stalled-cycles",          # Alder Lake 不支持
    "cpu_atom",                # 只采 P-core
    "ref-cycles",              # 与 cycles 冗余（x86 固定计数器之一，避免歧义）
]
# ARM 通用硬件事件与 PMU 原生事件的等价映射：
# 通用名不加前缀（perf 自行解析），且与原生事件重复，统一保留原生名避免同义列
ARM_GENERIC_ALIAS = {
    "cycles": "cpu_cycles",
    "cpu-cycles": "cpu_cycles",
    "instructions": "inst_retired",
    "branch-misses": "br_mis_pred",
    "bus-cycles": "bus_cycles",
}
# 探测用短负载：纯 CPU 循环，保证各事件产生非零计数（避免用 true/sleep 导致空计数）
PROBE_WORKLOAD = ["bash", "-c", "x=0; for i in $(seq 1 30000); do x=$((x+i)); done"]


def run(cmd, **kw):
    """运行命令并返回 stdout（文本）。"""
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def list_events(platform):
    """枚举 perf 支持的事件。

    关键：只有 Kernel PMU 事件需要加 PMU 前缀；通用硬件事件与硬件缓存事件
    （cycles / L1-dcache-loads 等）必须保持裸名，加前缀会变成非法事件名，
    导致 perf 整条命令失败（此前 ARM 端事件大量丢失的根因）。
    """
    out = run(["perf", "list"])
    events = set()
    prefix = PMU_PREFIX[platform]

    # 1) 通用硬件事件与硬件缓存事件：保持裸名
    for m in re.finditer(
            r"^\s{2}([\w.-]+)\s+(?:OR\s+([\w./-]+)\s+)?\[(Hardware event|Hardware cache event)\]",
            out, re.M):
        name = m.group(2) or m.group(1)  # 优先取 OR 后的规范名（如 cpu-cycles OR cycles → cycles）
        if any(s in name for s in EXCLUDE_SUBSTR):
            continue
        events.add(name)

    # 2) 内核 PMU 原生事件：加 PMU 前缀
    for m in re.finditer(r"^\s{2}([\w.-]+)\s+OR\s+([\w./-]+)\s+\[Kernel PMU event\]", out, re.M):
        raw = m.group(2)  # 形如 armv8_pmuv3_0/br_mis_pred/
        if any(s in raw for s in EXCLUDE_SUBSTR):
            continue
        events.add(raw if "/" in raw else f"{prefix}{raw}/")

    # 3) ARM：通用事件与原生事件等价，去重（保留原生名，避免同义列占用计数槽）
    if platform == "arm":
        for generic, native in ARM_GENERIC_ALIAS.items():
            if f"{prefix}{native}/" in events:
                events.discard(generic)
            # 若原生事件不存在，则保留通用名

    # 4) x86：Golden Cove 微架构 JSON 命名事件
    if platform == "x86":
        out_pmu = run(["perf", "list", "pmu"])
        for m in re.finditer(r"^\s{2}([a-z0-9_]+(?:\.[a-z0-9_-]+)+)\s", out_pmu, re.M | re.I):
            name = m.group(1)
            if any(s in name for s in EXCLUDE_SUBSTR):
                continue
            events.add(f"{prefix}{name}/")

    return sorted(events)


def _probe_batch(batch):
    """探测一批事件，返回可用事件列表。整条命令失败时返回 None。"""
    r = subprocess.run(
        ["taskset", "-c", "0", "perf", "stat", "-e", ",".join(batch), "-x", ",", "--"]
        + PROBE_WORKLOAD, capture_output=True, text=True, env=clean_env())
    if "Run 'perf list'" in r.stderr:
        return None  # 批次含非法事件名，整条命令未执行
    ok = []
    for line in r.stderr.splitlines():
        if "<not supported>" in line or "<not counted>" in line:
            continue
        m = re.match(r"\s*([\d,]+)\s*,+\s*([^,]+)", line)
        if m:
            ok.append(m.group(2).strip())
    return ok or None


def probe_events(events):
    """探测事件可用性。

    - 分批（4 个）快速探测；某批整体失败时退化为逐事件探测，
      避免一个非法事件拖垮整批（ARM 端曾因此丢失 8 个有效事件）。
    - 必须绑核，否则进程落到其他核导致事件被误判不可用。
    """
    ok = []
    for i in range(0, len(events), 4):
        batch = events[i:i + 4]
        res = _probe_batch(batch)
        if res is None:
            # 退化：逐个探测
            for ev in batch:
                single = _probe_batch([ev])
                if single:
                    ok.extend(single)
        else:
            ok.extend(res)
    return sorted(set(ok))


def make_groups(events, platform):
    """锚点事件 + 其余事件按组上限切分。"""
    anchors = ANCHOR[platform]
    rest = [e for e in events if e not in anchors]
    size = GROUP_SIZE[platform]
    groups = []
    for i in range(0, len(rest), size):
        groups.append(anchors + rest[i:i + size])
    return groups


def clean_env():
    """净化子进程环境。

    IDE/工具链会往 LD_LIBRARY_PATH 注入自带的 libstdc++.so / libselinux.so，
    导致被测程序加载到错误版本（如 GLIBCXX_3.4.32 not found）或工具本身报
    "no version information available"。采集必须使用干净环境。
    """
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    return env


def collect_one(platform, group, cmd, stdout_null, cwd=None):
    """单组单次采集，返回 {event: value}。"""
    full_cmd = ["taskset", "-c", "0", "perf", "stat", "-e",
                ",".join(group), "-x", ",", "--"] + cmd
    if stdout_null:
        full_cmd = ["bash", "-c",
                    " ".join(full_cmd) + " >/dev/null", ]
    r = subprocess.run(full_cmd, capture_output=True, text=True, cwd=cwd,
                       env=clean_env())
    res = {}
    for line in r.stderr.splitlines():
        m = re.match(r"\s*([\d,]+)\s*,+\s*([^,]+)", line)
        if m:
            val = int(m.group(1).replace(",", ""))
            ev = m.group(2).strip()
            res[ev] = val
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["x86", "arm"], required=True)
    ap.add_argument("--name", required=True, help="被测程序名（用于 CSV 标识）")
    ap.add_argument("--cmd", required=True, nargs=argparse.REMAINDER,
                    help="被测命令（必须放在最后，可包含以 - 开头的参数）")
    ap.add_argument("--out", required=True, help="输出宽表 CSV 路径")
    ap.add_argument("--reps", type=int, default=3, help="每组重复次数")
    ap.add_argument("--stdout-null", action="store_true",
                    help="将被测程序 stdout 重定向到 /dev/null（如 MiBench qsort）")
    ap.add_argument("--max-groups", type=int, default=0, help="调试用：只采前 N 组")
    ap.add_argument("--cwd", default=None, help="被测程序的工作目录（如 SPEC run 目录）")
    ap.add_argument("--events-file", default=None,
                    help="事件清单缓存文件：存在则直接读取，不存在则在探测后写入")
    args = ap.parse_args()

    if args.events_file and os.path.exists(args.events_file):
        with open(args.events_file) as f:
            events = [ln.strip() for ln in f if ln.strip()]
        print(f"[1-2/4] 从 {args.events_file} 读取 {len(events)} 个事件（跳过枚举与探测）")
    else:
        print(f"[1/4] 枚举 {args.platform} 平台事件...")
        events = list_events(args.platform)
        print(f"      枚举到 {len(events)} 个事件")

        print("[2/4] 探测事件可用性（剔除 not supported）...")
        events = probe_events(events)
        print(f"      可用 {len(events)} 个事件")
        if args.events_file:
            with open(args.events_file, "w") as f:
                f.write("\n".join(events) + "\n")
            print(f"      已缓存事件清单 -> {args.events_file}")

    groups = make_groups(events, args.platform)
    if args.max_groups:
        groups = groups[:args.max_groups]
    print(f"[3/4] 共 {len(groups)} 组 × {args.reps} 次重复 = {len(groups) * args.reps} 次运行")

    # rows[(group_idx, rep)] = {event: value}
    rows = {}
    all_events = set()
    for gi, g in enumerate(groups):
        for rep in range(args.reps):
            res = collect_one(args.platform, g, args.cmd, args.stdout_null, cwd=args.cwd)
            rows[(gi, rep)] = res
            all_events.update(res)
            if (gi * args.reps + rep + 1) % 10 == 0:
                print(f"      进度 {gi * args.reps + rep + 1}/{len(groups) * args.reps}")

    print("[4/4] 写出宽表 CSV...")
    header = ["program", "group", "rep"] + sorted(all_events)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (gi, rep), res in sorted(rows.items()):
            w.writerow([args.name, gi, rep] + [res.get(e, "") for e in header[3:]])

    # 锚点一致性校验：cycles 在各组间的变异系数
    anchor = ANCHOR[args.platform][0]
    vals = [res.get(anchor) for res in rows.values() if res.get(anchor)]
    if len(vals) > 1:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = (var ** 0.5) / mean * 100
        print(f"      锚点 {anchor} 跨组变异系数 CV = {cv:.2f}%（>5% 需检查运行一致性）")
    print(f"完成 -> {args.out}（{len(rows)} 行 × {len(all_events)} 事件）")


if __name__ == "__main__":
    main()

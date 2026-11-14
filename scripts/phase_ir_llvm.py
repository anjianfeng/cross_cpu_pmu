#!/usr/bin/env python3
"""构建并采集 LLVM MultiSource/Benchmarks 的 IR phase 数据。

程序源码、编译选项和运行参数直接取自各目录的 CMakeLists.txt；只选择
独立的 C/C++ 基准，不纳入同一算法的 float/double 或不同输入变体。

用法：
  python3 scripts/phase_ir_llvm.py build [--only NAME ...]
  python3 scripts/phase_ir_llvm.py collect [--only NAME ...]
  python3 scripts/phase_ir_llvm.py collect-arm [--only NAME ...]
  python3 scripts/phase_ir_llvm.py list
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_ir_build import ROOT, build_target

BASE = os.path.join(ROOT, "benchmarks/llvm-test-suite/MultiSource/Benchmarks")
OUT = os.path.join(ROOT, "build/phase_ir_llvm")
DATA = os.path.join(ROOT, "data/phase_ir/x86")
TARGETS = (("x86", []), ("arm", ["--target=aarch64-linux-gnu", "-static"]))
REMOTE = os.environ.get("D2000_USER", "soc") + "@" + os.environ.get("D2000_HOST", "")
RPORT = "2000"
RPASS = os.environ.get("D2000_PASS", "")
RROOT = "/home/soc/PMU_bench/phase_ir_expand"

# 同一算法只保留一个版本：不选 Fhourstones 新旧两份、TSVC 的 flt/dbl 对、
# mediabench 与本地 MiBench 重复的 adpcm/gsm/jpeg。
ROOTS = [
    "7zip", "ASC_Sequoia", "ASCI_Purple/SMG2000", "BitBench", "Bullet",
    "DOE-ProxyApps-C", "DOE-ProxyApps-C++", "Fhourstones-3.1",
    "FreeBench", "MallocBench", "McCat", "NPB-serial", "Olden",
    "OptimizerEval", "PAQ8p", "Prolangs-C", "Prolangs-C++", "Ptrdist",
    "Rodinia", "SciMark2-C", "Trimaran", "VersaBench", "nbench", "sim",
    "llubenchmark", "mafft", "tramp3d-v4",
    # TSVC 每个计算类别只保留 double 版本，不纳入对应 float 变体。
    "TSVC/ControlFlow-dbl", "TSVC/ControlLoops-dbl",
    "TSVC/Equivalencing-dbl", "TSVC/IndirectAddressing-dbl",
    "TSVC/LinearDependence-dbl", "TSVC/Recurrences-dbl",
    "TSVC/Reductions-dbl", "TSVC/Symbolics-dbl",
    "mediabench/g721/g721encode", "mediabench/mpeg2/mpeg2dec",
]
LEGACY_C = {"BitBench", "FreeBench", "MallocBench", "McCat", "Olden",
            "OptimizerEval", "Prolangs-C", "Ptrdist", "Trimaran", "VersaBench",
            "sim"}
REQUIRED_DIRS = {"llvm_sim_sim": ("Output",)}


def cmake_commands(text):
    """生成有效 (命令名, 参数文本)；默认不启用 LARGE/SMALL 输入分支。"""
    text = re.sub(r"(?m)#.*$", "", text)
    active = True
    branches = []
    for m in re.finditer(r"(?ms)^\s*([A-Za-z_][\w]*)\s*\((.*?)\)", text):
        cmd, args = m.group(1).lower(), m.group(2).strip()
        if cmd == "if":
            cond = False if args in ("LARGE_PROBLEM_SIZE", "SMALL_PROBLEM_SIZE") else None
            branches.append((active, cond))
            if cond is False:
                active = False
        elif cmd == "else":
            parent, cond = branches[-1]
            if cond is not None:
                active = parent if cond is False else False
        elif cmd == "endif":
            active = branches.pop()[0]
        elif active:
            yield cmd, args


def expand(text, variables):
    text = text.replace("${CMAKE_CURRENT_SOURCE_DIR}", variables["cwd"])
    for _ in range(8):
        new = re.sub(r"\$\{([A-Za-z_][\w]*)\}",
                     lambda m: variables.get(m.group(1), m.group(0)), text)
        if new == text:
            break
        text = new
    return text


def name_for(path, target):
    rel = os.path.relpath(path, BASE)
    top = rel.split(os.sep)[0].lower().replace("+", "p")
    top = re.sub(r"[^a-z0-9]+", "", top)
    return "llvm_%s_%s" % (top, re.sub(r"[^A-Za-z0-9_]+", "_", target))


def parse_program(path):
    mk = os.path.join(path, "CMakeLists.txt")
    text = open(mk, encoding="utf-8", errors="surrogateescape").read()
    variables = {"cwd": path}
    calls = list(cmake_commands(text))
    target = sources = None
    for cmd, args in calls:
        if cmd == "set":
            parts = args.split(None, 1)
            if len(parts) == 1:
                variables[parts[0]] = ""
            else:
                variables[parts[0]] = expand(parts[1], variables)
        elif cmd == "list":
            parts = args.split(None, 2)
            if len(parts) == 3 and parts[0].lower() == "append":
                variables[parts[1]] = (variables.get(parts[1], "") + " " +
                                       expand(parts[2], variables)).strip()
        elif cmd == "add_definitions":
            variables["CPPFLAGS"] = (variables.get("CPPFLAGS", "") + " " +
                                     expand(args, variables)).strip()
        elif cmd == "llvm_multisource":
            toks = expand(args, variables).split()
            if toks:
                target = toks[0]
                sources = toks[1:]
    if not target:
        return None
    if not sources:
        sources = sorted(os.path.basename(s) for ext in ("c", "cc", "cpp", "cxx")
                         for s in glob.glob(os.path.join(path, "*." + ext)))
    srcs = [os.path.normpath(os.path.join(path, s)) for s in sources]
    srcs = [s for s in srcs if os.path.isfile(s)]
    if not srcs:
        return None
    top = os.path.relpath(path, BASE).split(os.sep)[0]
    is_cpp = any(s.endswith((".cc", ".cpp", ".cxx")) for s in srcs)
    cxx_std = variables.get("CMAKE_CXX_STANDARD", "03")
    std = "-std=gnu++" + cxx_std if is_cpp else (
        "-std=gnu89" if top in LEGACY_C else "-std=gnu99")
    cflags = [std, "-w", "-fcommon", "-D_GNU_SOURCE", "-I" + path]
    for key in ("CPPFLAGS", "CFLAGS", "CXXFLAGS"):
        cflags += variables.get(key, "").split()
    if top == "NPB-serial" and target == "is":
        # 该源码默认选择 B 类大输入；沿用已验证的 W 类小规模配置。
        cflags.append("-DSMALL_PROBLEM_SIZE")
    cflags = [f for f in cflags if f != "-mxgot"]
    ldflags = variables.get("LDFLAGS", "").split()
    if target == "archie":
        # CMake 保留的是 Solaris  socket 库；Linux 所需接口已由 libc 提供。
        ldflags = [f for f in ldflags if f not in ("-lsocket", "-lnsl")]
    if target == "espresso":
        # 程序自带 getopt，静态 glibc 也含 optind；保留程序内定义即可。
        ldflags.append("-Wl,--allow-multiple-definition")
    if is_cpp:
        ldflags.append("-lstdc++")
    run_options = expand(variables.get("RUN_OPTIONS", ""), variables)
    if top == "mediabench" and target == "encode":
        run_options = run_options.replace("clinton.pcm", "../data/clinton.pcm")
    if top == "mediabench" and target == "mpeg2decode":
        run_options = run_options.replace("mei16v2.m2v", "../data/mei16v2.m2v")
    return {
        "name": name_for(path, target), "target": target,
        "path": os.path.relpath(path, BASE), "cwd": path, "srcs": srcs,
        "cflags": cflags, "ldflags": ldflags, "cxx": is_cpp,
        "run_options": run_options,
    }


def discover():
    programs = []
    seen = set()

    def walk(path):
        mk = os.path.join(path, "CMakeLists.txt")
        if not os.path.isfile(mk):
            return
        p = parse_program(path)
        if p:
            if p["name"] not in seen:
                programs.append(p)
                seen.add(p["name"])
        text = open(mk, encoding="utf-8", errors="surrogateescape").read()
        for cmd, args in cmake_commands(text):
            if cmd == "add_subdirectory":
                sub = expand(args, {"cwd": path}).split()[0]
                walk(os.path.normpath(os.path.join(path, sub)))

    for rel in ROOTS:
        walk(os.path.join(BASE, rel))
    return programs


def selected(programs, only):
    if not only:
        return programs
    keys = set(only)
    return [p for p in programs if p["name"] in keys or p["target"] in keys
            or p["path"] in keys]


def save_manifest(programs):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "manifest.json")
    with open(path, "w") as f:
        json.dump(programs, f, indent=2, ensure_ascii=False)
    return path


def do_build(programs):
    ok = fail = skip = 0
    x86_only = os.environ.get("PHASE_BUILD_X86_ONLY") == "1"
    force = os.environ.get("PHASE_FORCE_REBUILD") == "1"
    tags = [t for t in TARGETS if not (x86_only and t[0] == "arm")]
    for i, p in enumerate(programs, 1):
        exes = [os.path.join(OUT, tag, p["name"]) for tag, _ in tags]
        if not force and all(os.path.isfile(e) for e in exes):
            skip += 1
            print("[%d/%d] %s 已构建" % (i, len(programs), p["name"]), flush=True)
            continue
        try:
            for tag, tf in TARGETS:
                build_target(p["name"], p["srcs"], OUT, tag, tf,
                             p["cflags"], p["ldflags"], p["cxx"], p["cwd"])
            ok += 1
            print("[%d/%d] %s OK" % (i, len(programs), p["name"]), flush=True)
        except RuntimeError as e:
            fail += 1
            print("[%d/%d] %s 失败\n%s" %
                  (i, len(programs), p["name"], str(e)[-900:]), flush=True)
    print("构建完成：成功 %d，已有 %d，失败 %d" % (ok, skip, fail))


def do_collect(programs, outdir, phase_bb, timeout):
    os.makedirs(outdir, exist_ok=True)
    status_path = os.path.join(OUT, "collect_x86_status.json")
    status = json.load(open(status_path)) if os.path.isfile(status_path) else {}
    for i, p in enumerate(programs, 1):
        name = p["name"]
        out = os.path.join(os.path.abspath(outdir), name + ".csv")
        exe = os.path.join(OUT, "x86", name)
        if not os.path.isfile(exe):
            print("[%d/%d] %s 缺二进制" % (i, len(programs), name), flush=True)
            status[name] = "no_binary"
            continue
        if os.path.exists(out) and os.path.getsize(out) > 50:
            print("[%d/%d] %s 已有数据" % (i, len(programs), name), flush=True)
            status[name] = "skip"
            continue
        for d in REQUIRED_DIRS.get(name, ()):
            os.makedirs(os.path.join(p["cwd"], d), exist_ok=True)
        env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
               "HOME": os.environ.get("HOME", "/tmp"),
               "PHASE_OUT": out, "PHASE_BB": str(phase_bb)}
        cmd = "exec taskset -c 0 %s %s" % (shlex.quote(exe), p["run_options"])
        try:
            r = subprocess.run(["/bin/sh", "-c", cmd], cwd=p["cwd"], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=timeout)
            st = "rc=%d" % r.returncode
            err = r.stderr.decode(errors="replace").strip()[-180:]
        except subprocess.TimeoutExpired:
            st, err = "timeout", ""
        rows = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else 0
        status[name] = {"status": st, "rows": rows, "stderr": err}
        print("[%d/%d] %s %s rows=%d%s" %
              (i, len(programs), name, st, rows,
               (" " + err) if err else ""), flush=True)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


def ssh(cmd, timeout=900, stdin=None):
    return subprocess.run(
        ["sshpass", "-p", RPASS, "ssh", "-p", RPORT,
         "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
         REMOTE, cmd], stdin=stdin, capture_output=True, text=True,
        timeout=timeout)


def upload_arm(programs):
    """上传输入树和待采集的静态 ARM 二进制，保留各程序相对 cwd。"""
    with tempfile.TemporaryDirectory(prefix="phase_ir_llvm_arm_") as tmp:
        bindir = os.path.join(tmp, "bin")
        os.mkdir(bindir)
        for p in programs:
            src = os.path.join(OUT, "arm", p["name"])
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(bindir, p["name"]))
        tar = subprocess.Popen(
            ["tar", "-czf", "-", "-C", ROOT,
             "benchmarks/llvm-test-suite/MultiSource/Benchmarks",
             "-C", tmp, "bin"], stdout=subprocess.PIPE)
        try:
            r = ssh("mkdir -p %s/data %s/out && tar -xzf - -C %s" %
                    (RROOT, RROOT, RROOT), timeout=1800, stdin=tar.stdout)
        finally:
            tar.stdout.close()
            tar_rc = tar.wait()
        if tar_rc != 0 or r.returncode != 0:
            raise RuntimeError("ARM 上传失败: tar=%d ssh=%d %s" %
                               (tar_rc, r.returncode, r.stderr[-300:]))


def do_collect_arm(programs, outdir, phase_bb, timeout):
    os.makedirs(outdir, exist_ok=True)
    status_path = os.path.join(OUT, "collect_arm_status.json")
    status = json.load(open(status_path)) if os.path.isfile(status_path) else {}
    xstatus_path = os.path.join(OUT, "collect_x86_status.json")
    xstatus = json.load(open(xstatus_path)) if os.path.isfile(xstatus_path) else {}
    paired = []
    for p in programs:
        xf = os.path.join(DATA, p["name"] + ".csv")
        rows = sum(1 for _ in open(xf)) - 1 if os.path.isfile(xf) else 0
        xs = xstatus.get(p["name"], {})
        if rows < 10 or (isinstance(xs, dict) and xs.get("status") == "timeout"):
            print("%s 跳过：x86 rows=%d status=%s" %
                  (p["name"], rows, xs.get("status") if isinstance(xs, dict) else xs),
                  flush=True)
            continue
        paired.append(p)
    programs = paired
    upload_arm(programs)
    for i, p in enumerate(programs, 1):
        name = p["name"]
        out = os.path.join(os.path.abspath(outdir), name + ".csv")
        exe = os.path.join(OUT, "arm", name)
        if not os.path.isfile(exe):
            print("[%d/%d] %s 缺 ARM 二进制" % (i, len(programs), name),
                  flush=True)
            status[name] = "no_binary"
            continue
        if os.path.exists(out) and os.path.getsize(out) > 50:
            print("[%d/%d] %s 已有数据" % (i, len(programs), name), flush=True)
            status[name] = "skip"
            continue
        cwd = p["cwd"].replace(ROOT, RROOT, 1)
        rout = "%s/data/%s.csv" % (RROOT, name)
        rerr = "%s/out/%s.err" % (RROOT, name)
        inner = "exec taskset -c 0 %s %s" % (
            shlex.quote("%s/bin/%s" % (RROOT, name)), p["run_options"])
        mkdir = "".join("mkdir -p %s && " % shlex.quote(os.path.join(cwd, d))
                        for d in REQUIRED_DIRS.get(name, ()))
        cmd = ("cd %s && %sPHASE_OUT=%s PHASE_BB=%d /bin/sh -c %s "
               ">/dev/null 2>%s") % (shlex.quote(cwd), mkdir, rout, phase_bb,
                                      shlex.quote(inner), rerr)
        try:
            r = ssh(cmd, timeout=timeout + 30)
            st = "rc=%d" % r.returncode
            err = r.stderr.strip()[-180:]
        except subprocess.TimeoutExpired:
            st, err = "timeout", ""
        rr = subprocess.run(
            ["sshpass", "-p", RPASS, "scp", "-P", RPORT,
             "-o", "StrictHostKeyChecking=no",
             "%s:%s" % (REMOTE, rout), out], capture_output=True, text=True,
            timeout=900)
        if rr.returncode != 0:
            err = (err + " " + rr.stderr.strip()[-180:]).strip()
        rows = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else 0
        status[name] = {"status": st, "rows": rows, "stderr": err}
        print("[%d/%d] %s %s rows=%d%s" %
              (i, len(programs), name, st, rows,
               (" " + err) if err else ""), flush=True)
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "collect", "collect-arm", "list"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--outdir")
    ap.add_argument("--phase-bb", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    all_programs = discover()
    programs = selected(all_programs, args.only)
    manifest = save_manifest(all_programs)
    if args.action == "list":
        for p in programs:
            print("%-34s %-45s %s" % (p["name"], p["path"], p["run_options"]))
        print("共 %d 个独立候选；清单 %s" % (len(programs), manifest))
    elif args.action == "build":
        do_build(programs)
    elif args.action == "collect":
        do_collect(programs, args.outdir or DATA, args.phase_bb, args.timeout)
    else:
        arm_data = os.path.join(ROOT, "data/phase_ir/arm")
        do_collect_arm(programs, args.outdir or arm_data,
                       args.phase_bb, args.timeout)


if __name__ == "__main__":
    main()

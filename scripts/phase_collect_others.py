#!/usr/bin/env python3
"""采集 MiBench + SD-VBS + PolyBench 的 phase 数据（x86 本机 / D2000 ARM）。

用法:
  python3 scripts/phase_collect_others.py --platform x86 \
      --bindir build/phase_mibench/x86:build/phase_pb/x86 \
      --outdir data/phase/x86

  python3 scripts/phase_collect_others.py --platform arm \
      --bindir build/phase_mibench/arm:build/phase_pb/arm \
      --outdir data/phase/arm
"""

import argparse, csv, json, os, shlex, shutil, subprocess, sys, tempfile

from phase_ir_llvm import RROOT, ssh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOTE = os.environ.get("D2000_USER", "soc") + "@" + os.environ.get("D2000_HOST", "")
RPORT  = "2000"
RPASS  = os.environ.get("D2000_PASS", "")
RROOT  = "/home/soc/PMU_bench/phase_other"

# 每个程序的运行命令（相对 bindir）。key = 程序名（同 CSV 前缀）
# tuple 第一个元素起就是命令行参数（二进制本身由 bindir+name 决定）
# MiBench 用 large 版本，与整程序采集一致
CMDS = {
    # MiBench（参数列表：去掉冗余的程序名）
    "aes":      ("input_large.asc", "/tmp/aes_out.enc", "e",
                 "1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321"),
    "crc":      ("../adpcm/data/large.pcm",),
    "dijkstra": ("input.dat",),
    "fft":      ("8", "32768"),
    "patricia": ("large.udp",),
    "qsort":    ("input_large.dat",),
    "sha":      ("input_large.asc",),
    "basicmath": (),
    "bitcount": ("1125000",),
    "susan":    ("input_large.pgm", "/tmp/susan_large.pgm", "-s"),
    "blowfish": ("e", "input_large.asc", "/tmp/blowfish_large.enc",
                 "1234567890abcdeffedcba0987654321"),
    "stringsearch": (),
    "adpcm_encode": (),
    "adpcm_decode": (),
    "gsm":      ("-fps", "-c", "data/large.au"),
    # SD-VBS：两个参数 (data_dir, result_dir)，都用绝对路径
    # cwd 在 SD-VBS 根目录，BMARK 是程序名
    # 统一使用 qcif 数据集：9 个程序中只有 qcif 全集可用
    # （localization/svm 无 fullhd），且 fullhd 单程运行时长达 10 分钟、
    # phase 轨迹达数十万行，不适合做训练集。qcif 在两侧平台口径一致即可。
    "disparity":  ("benchmarks/disparity/data/qcif/",
                   "benchmarks/disparity/result/"),
    "localization":("benchmarks/localization/data/qcif/",
                   "benchmarks/localization/result/"),
    "mser":       ("benchmarks/mser/data/qcif/",
                   "benchmarks/mser/result/"),
    "multi_ncut": ("benchmarks/multi_ncut/data/qcif/",
                   "benchmarks/multi_ncut/result/"),
    "sift":       ("benchmarks/sift/data/qcif/",
                   "benchmarks/sift/result/"),
    "stitch":     ("benchmarks/stitch/data/qcif/",
                   "benchmarks/stitch/result/"),
    "svm":        ("benchmarks/svm/data/qcif/",
                   "benchmarks/svm/result/"),
    "texture_synthesis": ("benchmarks/texture_synthesis/data/qcif/",
                   "benchmarks/texture_synthesis/result/"),
    "tracking":   ("benchmarks/tracking/data/qcif/",
                   "benchmarks/tracking/result/"),
}

# 各程序的 cwd（需要在源码目录运行以访问相对路径输入文件）
CWD = {
    "aes":      ROOT + "/benchmarks/mibench/security/rijndael",
    "crc":      ROOT + "/benchmarks/mibench/telecomm/CRC32",
    "dijkstra": ROOT + "/benchmarks/mibench/network/dijkstra",
    "fft":      ROOT + "/benchmarks/mibench/telecomm/FFT",
    "patricia": ROOT + "/benchmarks/mibench/network/patricia",
    "qsort":    ROOT + "/benchmarks/mibench/automotive/qsort",
    "sha":      ROOT + "/benchmarks/mibench/security/sha",
    "basicmath": ROOT + "/benchmarks/mibench/automotive/basicmath",
    "bitcount": ROOT + "/benchmarks/mibench/automotive/bitcount",
    "susan":    ROOT + "/benchmarks/mibench/automotive/susan",
    "blowfish": ROOT + "/benchmarks/mibench/security/blowfish",
    "stringsearch": ROOT + "/benchmarks/mibench/office/stringsearch",
    "adpcm_encode": ROOT + "/benchmarks/mibench/telecomm/adpcm",
    "adpcm_decode": ROOT + "/benchmarks/mibench/telecomm/adpcm",
    "gsm":      ROOT + "/benchmarks/mibench/telecomm/gsm",
    # SD-VBS 数据和结果路径用绝对路径（从 SD-VBS 根目录运行）
    "disparity":  ROOT + "/benchmarks/SD-VBS",
    "localization":ROOT + "/benchmarks/SD-VBS",
    "mser":       ROOT + "/benchmarks/SD-VBS",
    "multi_ncut": ROOT + "/benchmarks/SD-VBS",
    "sift":       ROOT + "/benchmarks/SD-VBS",
    "stitch":     ROOT + "/benchmarks/SD-VBS",
    "svm":        ROOT + "/benchmarks/SD-VBS",
    "texture_synthesis": ROOT + "/benchmarks/SD-VBS",
    "tracking":   ROOT + "/benchmarks/SD-VBS",
}

# 实际二进制名（有些程序 make 出来的名字不同）
# phase 构建脚本输出名取自 PROGS 第一列（dijkstra、qsort），
# 而源码是 dijkstra_large.c、qsort_large.c → 输出名简化
BINARY_NAME = {}

# 需要从文件重定向标准输入的程序，路径相对于各自 CWD。
STDIN = {
    "adpcm_encode": "data/large.pcm",
    "adpcm_decode": "data/large.adpcm",
}


def clean_env():
    return {"PATH": "/usr/bin:/bin:/usr/local/bin",
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu",
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C", "LC_ALL": "C"}


def resolve_bindir(bindirs, name):
    """在 bindirs 列表里找含有对应二进制的目录（返回绝对路径）。"""
    bname = BINARY_NAME.get(name, name)
    for d in bindirs:
        abs_d = os.path.abspath(d)
        cand = os.path.join(abs_d, bname)
        if os.path.isfile(cand):
            return abs_d, cand
    return None, None


def run_one_x86(bindirs, outdir, name, phase_bb):
    # outdir 可能是相对路径，但子进程 cwd 是各程序自己的目录，
    # 必须转成绝对路径，否则 PHASE_OUT 会落到错误位置而静默失败
    out = os.path.join(os.path.abspath(outdir), f"{name}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 50:
        return out, "skip"
    bindir, exe = resolve_bindir(bindirs, name)
    if exe is None:
        return None, "no_binary"
    cmd = [exe] + list(CMDS.get(name, []))
    cwd = CWD.get(name, bindir)
    env = clean_env()
    env["PHASE_OUT"] = out
    env["PHASE_BB"]  = str(phase_bb)
    dn = open(os.devnull, "w")
    inf = None
    try:
        if name in STDIN:
            inf = open(os.path.join(cwd, STDIN[name]), "rb")
        r = subprocess.run(["taskset", "-c", "0"] + cmd,
                           cwd=cwd, env=env, stdin=inf,
                           stdout=dn, stderr=subprocess.PIPE, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        dn.close()
        if inf is not None:
            inf.close()
    return out, f"rc={r.returncode}"


def run_arm(bindirs, outdir, todo, phase_bb, timeout):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="phase_ir_others_arm_") as tmp:
        remote_bin = os.path.join(tmp, "bin")
        os.mkdir(remote_bin)
        for name in todo:
            _, exe = resolve_bindir(bindirs, name)
            if exe:
                shutil.copy2(exe, os.path.join(remote_bin, name))
        tar = subprocess.Popen(
            ["tar", "-czf", "-", "-C", root, "benchmarks/mibench",
             "benchmarks/SD-VBS", "-C", tmp, "bin"], stdout=subprocess.PIPE)
        try:
            r = ssh("mkdir -p %s/data %s/out && tar -xzf - -C %s" %
                    (RROOT, RROOT, RROOT), timeout=1800, stdin=tar.stdout)
        finally:
            tar.stdout.close()
            tar_rc = tar.wait()
        if tar_rc != 0 or r.returncode != 0:
            raise RuntimeError("ARM 上传失败: tar=%d ssh=%d %s" %
                               (tar_rc, r.returncode, r.stderr[-300:]))

    status = {}
    for i, name in enumerate(todo, 1):
        out = os.path.join(os.path.abspath(outdir), name + ".csv")
        if os.path.exists(out) and os.path.getsize(out) > 50:
            print("[%d/%d] %s 已有数据" % (i, len(todo), name), flush=True)
            status[name] = "skip"
            continue
        _, exe = resolve_bindir(bindirs, name)
        if not exe:
            print("[%d/%d] %s 缺 ARM 二进制" % (i, len(todo), name), flush=True)
            status[name] = "no_binary"
            continue
        cwd = CWD.get(name, root).replace(root, RROOT, 1)
        inner = "exec taskset -c 0 %s %s" % (
            shlex.quote("%s/bin/%s" % (RROOT, name)),
            " ".join(shlex.quote(a) for a in CMDS.get(name, ())))
        if name in STDIN:
            inner += " < " + shlex.quote(os.path.join(cwd, STDIN[name]))
        rout = "%s/data/%s.csv" % (RROOT, name)
        rerr = "%s/out/%s.err" % (RROOT, name)
        cmd = ("cd %s && PHASE_OUT=%s PHASE_BB=%d /bin/sh -c %s "
               ">/dev/null 2>%s") % (shlex.quote(cwd), rout, phase_bb,
                                      shlex.quote(inner), rerr)
        try:
            r = ssh(cmd, timeout=timeout + 30)
            st, err = "rc=%d" % r.returncode, r.stderr.strip()[-180:]
        except subprocess.TimeoutExpired:
            st, err = "timeout", ""
        rr = subprocess.run(
            ["sshpass", "-p", RPASS, "scp", "-P", RPORT,
             "-o", "StrictHostKeyChecking=no", "%s:%s" % (REMOTE, rout), out],
            capture_output=True, text=True, timeout=900)
        if rr.returncode != 0:
            err = (err + " " + rr.stderr.strip()[-180:]).strip()
        rows = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else 0
        status[name] = {"status": st, "rows": rows, "stderr": err}
        print("[%d/%d] %s %s rows=%d%s" %
              (i, len(todo), name, st, rows,
               (" " + err) if err else ""), flush=True)
        status_path = os.path.join(root, "build/phase_ir",
                                   "collect_arm_others_status.json")
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["x86", "arm"], required=True)
    ap.add_argument("--bindir", required=True,
                    help="二进制目录列表，用 : 分隔（e.g. build/a:build/b）")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase-bb", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    bindirs = args.bindir.split(":")
    for d in bindirs:
        os.makedirs(d, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    todo = sorted(CMDS.keys())
    if args.only:
        todo = [p for p in todo if p in args.only]

    if args.platform == "arm":
        run_arm(bindirs, args.outdir, todo, args.phase_bb, args.timeout)
        return

    for i, name in enumerate(todo, 1):
        out, st = run_one_x86(bindirs, args.outdir, name, args.phase_bb)
        print(f"[{i}/{len(todo)}] {name}: {st}")
        if out and os.path.exists(out):
            try:
                with open(out) as f:
                    phase_count = sum(1 for _ in f) - 1
                if phase_count < 3:
                    print(f"  ⚠️ 只有 {phase_count} 个 phase（{out}），可能输入不对")
            except Exception:
                pass


if __name__ == "__main__":
    main()

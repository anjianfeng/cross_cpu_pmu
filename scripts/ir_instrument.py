#!/usr/bin/env python3
"""在 LLVM IR 层做 phase 插桩，导出每个基本块的结构特征。

为什么要在 IR 层做：
  只用 PMU 通用事件（6 个）无法区分一个 phase 里到底在执行什么代码——
  浮点密集、访存密集、分支密集的块在计数器上可以长得一模一样。
  LACross 论文的精度优势正来自这类「架构无关的结构特征」。
  这里用 IR 静态分类 + 运行时按执行次数加权，得到不依赖任何特定 ISA
  的 phase 级指令构成，x86 与 aarch64 两平台语义一致。

做法：
  1. clang -O2 -S -emit-llvm 得到优化后的 .ll；
  2. 给每个基本块分配全局唯一 id，在块首（phi / landingpad 之后）插入
         call void @__phase_bb(i32 <id>)
  3. 统计该块的指令类型构成与后继块，写成 C 表（phase_bb_table.h 的结构）。

用法:
  python3 scripts/ir_instrument.py --outdir DIR --table DIR/table.c a.ll b.ll
产物:
  DIR/a.inst.ll, DIR/b.inst.ll, DIR/table.c
"""

import argparse
import os
import re
import sys

MDECL = "declare void @__phase_bb(i32)\n"

LABEL_RE = re.compile(r'^(?:"([^"]+)"|([-a-zA-Z$._0-9]+)):')
DEFINE_RE = re.compile(r'^define\b')
RESULT_RE = re.compile(
    r'^%[-\w.$"]+\s*=\s*(?:(?:tail|musttail|notail)\s+)?([a-z][\w.]*)')
OP_RE = re.compile(r'^([a-z][\w.]*)')

# opcode -> 结构类别
OP_CLASS = {
    "load": "ld", "store": "st", "atomicrmw": "st", "cmpxchg": "st",
    "fadd": "fp", "fsub": "fp", "fmul": "fp", "fdiv": "fp", "frem": "fp",
    "fneg": "fp", "fcmp": "fp",
    "sdiv": "alu", "udiv": "alu", "srem": "alu", "urem": "alu",
    "call": "call", "invoke": "call", "callbr": "call",
    "br": "br", "switch": "br", "indirectbr": "br", "ret": "br",
    "resume": "br", "unreachable": "br", "cleanupret": "br",
    "catchret": "br",
    "phi": "phi",
}
DEFAULT_CLASS = "alu"
DIV_OPS = {"sdiv", "udiv", "srem", "urem", "fdiv", "frem"}
# 这些指令必须是块内第一条非 phi 指令，探针要插在它们后面
PAD_OPS = {"landingpad", "catchpad", "cleanuppad", "catchswitch"}

FIELDS = ["ld", "st", "alu", "fp", "vec", "div", "call", "br", "inst"]


class Block:
    __slots__ = ("label", "label_idx", "insns", "id", "feat", "succ")

    def __init__(self, label, label_idx):
        self.label = label
        self.label_idx = label_idx
        self.insns = []      # [(line_idx, line)]
        self.id = -1
        self.feat = dict.fromkeys(FIELDS, 0)
        self.succ = []


def opcode_of(line):
    s = line.strip()
    if not s or s.startswith(";"):
        return None
    m = RESULT_RE.match(s)
    if m:
        return m.group(1)
    m = OP_RE.match(s)
    return m.group(1) if m else None


def classify(line):
    """返回 (opcode, 类别, 是否向量指令)。"""
    op = opcode_of(line)
    if op is None or op.startswith("llvm.dbg"):
        return None, None, False
    cls = OP_CLASS.get(op, DEFAULT_CLASS)
    is_vec = bool(re.search(r"<\s*\d+\s*x\s*", line))
    return op, cls, is_vec


def split_blocks(lines, start, end):
    """把 [start, end) 区间的函数体切成基本块。

    insns 中每项为 (起始行号, 结束行号, 指令文本)。LLVM 文本 IR 里少数指令
    会跨行（如 landingpad 的 cleanup 子句、跨行 switch），缩进大于 2 个空格
    的行视为上一条指令的续行。
    """
    blocks = []
    cur = None
    for idx in range(start, end):
        line = lines[idx]
        m = LABEL_RE.match(line)
        if m:
            cur = Block(m.group(1) or m.group(2), idx)
            blocks.append(cur)
            continue
        if cur is None:                      # 入口块没有标签
            cur = Block(None, start - 1)
            blocks.append(cur)
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue                         # 空行/注释不参与统计与终结指令判定
        indent = len(line) - len(line.lstrip())
        if indent > 2 and cur.insns:         # 续行
            s, e, t = cur.insns[-1]
            cur.insns[-1] = (s, idx, t + line)
        else:
            cur.insns.append((idx, idx, line))
    for b in blocks:
        for _, _, text in b.insns:
            op, cls, is_vec = classify(text)
            if op is None or cls is None:
                continue
            if cls != "phi":             # phi 在寄存器分配后基本消失，不计入
                b.feat[cls] += 1
                b.feat["inst"] += 1
            if is_vec:
                b.feat["vec"] += 1
            if op in DIV_OPS:
                b.feat["div"] += 1
    return blocks


def terminator_succ_labels(block):
    """从终结指令里抽取后继块标签（按 IR 文本顺序，succ[0] 为条件为真分支）。"""
    if not block.insns:
        return []
    last = block.insns[-1][2]
    if opcode_of(last) not in ("br", "switch", "indirectbr", "invoke", "callbr"):
        return []
    return [a or b for a, b in
            re.findall(r'label\s+%(?:"([^"]+)"|([-a-zA-Z$._0-9]+))', last)]


def process_file(path, outdir, next_id):
    lines = open(path, encoding="utf-8", errors="surrogateescape").readlines()

    # ---- 1. 找函数体区间并切块 ----
    funcs = []
    i, n = 0, len(lines)
    while i < n:
        if DEFINE_RE.match(lines[i]):
            j = i + 1
            while j < n and lines[j].rstrip("\n") != "}":
                j += 1
            funcs.append((i + 1, j))
            i = j + 1
        else:
            i += 1

    all_blocks = []
    for start, end in funcs:
        blocks = split_blocks(lines, start, end)
        # ---- 2. 分配 id 并解析块内后继 ----
        by_label = {}
        for b in blocks:
            b.id = next_id
            next_id += 1
            if b.label:
                by_label[b.label] = b.id
            all_blocks.append(b)
        for b in blocks:
            b.succ = [by_label.get(l, -1) for l in terminator_succ_labels(b)]

    # ---- 3. 生成插桩后的 .ll ----
    inserts = {}
    for b in all_blocks:
        pos = 0
        while pos < len(b.insns):
            op = opcode_of(b.insns[pos][2])
            if op in PAD_OPS or op == "phi":
                pos += 1
                continue
            break
        # 插在 phi / landingpad 之后（多行指令用其结束行号）
        anchor = b.insns[pos - 1][1] if pos > 0 else b.label_idx
        inserts[anchor] = "  call void @__phase_bb(i32 %d)\n" % b.id

    out = []
    declared = False
    for idx, line in enumerate(lines):
        if not declared and DEFINE_RE.match(line):
            out.append(MDECL)
            declared = True
        out.append(line)
        if idx in inserts:
            out.append(inserts[idx])
    if not declared:
        out.append(MDECL)

    base = os.path.basename(path)
    if base.endswith(".ll"):
        base = base[:-3]
    dst = os.path.join(outdir, base + ".inst.ll")
    with open(dst, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.writelines(out)
    return all_blocks, next_id, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("ll", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    table = []          # 按 id 顺序的 (feat, succ)
    next_id = 0
    outs = []
    for path in args.ll:
        blocks, next_id, dst = process_file(path, args.outdir, next_id)
        outs.append(dst)
        # process_file 内部按文件返回块列表，这里需要按 id 排序后追加
        for b in sorted(blocks, key=lambda x: x.id):
            table.append((b.feat, b.succ))

    with open(args.table, "w") as f:
        f.write('#include "phase_bb_table.h"\n\n')
        f.write("const struct phase_bb_info phase_bb_table[] = {\n")
        for feat, succ in table:
            s = list(succ) + [-1, -1]
            f.write("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,{%d,%d}},\n" % (
                feat["inst"], feat["ld"], feat["st"], feat["alu"], feat["fp"],
                feat["vec"], feat["div"], feat["call"], feat["br"],
                len(succ) if len(succ) <= 2 else 3, s[0], s[1]))
        f.write("};\n\n")
        f.write("const unsigned int phase_bb_table_size = %d;\n" % len(table))

    print("\n".join(outs))
    print("blocks=%d" % len(table), file=sys.stderr)


if __name__ == "__main__":
    main()

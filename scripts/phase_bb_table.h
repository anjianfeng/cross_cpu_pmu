/* 基本块静态结构表：由 scripts/ir_instrument.py 生成，
 * 供 phase_profiler_ir.c 在运行时按执行次数加权累加。
 *
 * 字段含义（IR 层计数，与 ISA 无关）：
 *   n_inst  该块 IR 指令总数（不含 phi）
 *   n_ld/n_st  访存指令数
 *   n_alu   整数/地址运算、比较、类型转换
 *   n_fp    浮点运算（含 fcmp/fneg）
 *   n_vec   操作向量类型的指令数（叠加计数，不计入 n_inst）
 *   n_div   除法/取余（跨平台代价差异最大的操作）
 *   n_call  调用
 *   n_br    分支/返回等终结指令
 *   nsucc   后继块个数（2 表示条件分支）
 *   succ[]  后继块 id，-1 表示无；succ[0] 为 IR 文本里第一个目标（条件为真）
 */
#ifndef PHASE_BB_TABLE_H
#define PHASE_BB_TABLE_H

#include <stdint.h>

struct phase_bb_info {
    uint16_t n_inst;
    uint16_t n_ld;
    uint16_t n_st;
    uint16_t n_alu;
    uint16_t n_fp;
    uint16_t n_vec;
    uint16_t n_div;
    uint16_t n_call;
    uint16_t n_br;
    uint16_t nsucc;
    int32_t succ[2];
};

#endif

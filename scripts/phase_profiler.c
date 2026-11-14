/* LACross 风格 phase 级性能剖析运行时。
 *
 * 原理：
 *   1. 编译时用 `-fsanitize-coverage=trace-pc` 让 clang 在每个基本块插入
 *      对 __sanitizer_cov_trace_pc() 的调用；
 *   2. 本文件实现该回调，做基本块计数，每累计 PHASE_BB 个基本块就读取一次
 *      硬件性能计数器，把增量写成一行 CSV（即一个 phase）；
 *   3. 只用 PERF_TYPE_HARDWARE / PERF_TYPE_HW_CACHE 里的通用事件，
 *      因此 x86 与 ARM 得到的是语义一致的特征向量；
 *   4. 目标平台（D2000）上读到的 cycles 就是该 phase 的真实 ARM 周期数，
 *      即监督学习的标签。
 *
 * 编译时必须单独编译本文件且不要加 -fsanitize-coverage，否则回调会自递归。
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAX_EVENTS 8

static int g_fds[MAX_EVENTS];
static int g_pos[MAX_EVENTS];   /* 该事件在组读数 values[] 中的下标，-1 表示未打开 */
static int g_nevents = 0;
static long long g_phase_bb = 5000;   /* 每个 phase 的基本块数 */
static long long g_bb_count = 0;
static long long g_phase_id = 0;
static FILE *g_out = NULL;

/* 通用事件集合：x86 与 ARM 都有对应实现 */
#define N_EVENTS 6
static const char *g_names[N_EVENTS] = {
    "cycles", "instructions", "cache_refs", "cache_misses",
    "branches", "branch_misses",
};
static const unsigned g_types[N_EVENTS] = {
    PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
    PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
};
static const uint64_t g_configs[N_EVENTS] = {
    PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
    PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES,
    PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
};

struct read_data {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    struct {
        uint64_t value;
    } values[MAX_EVENTS];
};

static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int open_event(unsigned type, uint64_t config, int group_fd) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = (group_fd == -1) ? 1 : 0;
    pe.exclude_kernel = getenv("PHASE_KERNEL") ? 0 : 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                     PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)perf_event_open(&pe, 0, -1, group_fd, 0);
}

static void prof_init(void) {
    const char *outpath = getenv("PHASE_OUT");
    const char *kstr = getenv("PHASE_BB");
    if (kstr && atoll(kstr) > 0) g_phase_bb = atoll(kstr);
    if (!outpath) outpath = "phase.csv";

    g_out = fopen(outpath, "w");
    if (!g_out) {
        /* 无法写文件时直接放弃剖析，不影响被测程序自身运行 */
        return;
    }
    fprintf(g_out,
            "phase,cycles,instructions,cache_refs,cache_misses,"
            "branches,branch_misses,t_enabled,t_running\n");

    int total = (int)(sizeof(g_configs) / sizeof(g_configs[0]));
    int leader = -1;
    int pos = 0;
    for (int i = 0; i < total; i++) {
        int fd = open_event(g_types[i], g_configs[i], leader);
        if (fd < 0) {
            /* 某些平台不支持个别通用事件（如 D2000 无分支事件）。
               跳过该事件但保持输出列固定，未打开的事件恒为 0。 */
            fprintf(stderr, "[phase_profiler] 事件 %s 不可用: %s\n",
                    g_names[i], strerror(errno));
            g_pos[i] = -1;
            if (i == 0) return;
            continue;
        }
        g_pos[i] = pos++;
        g_fds[i] = fd;
        if (leader == -1) {
            leader = fd;
            ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        }
        g_nevents++;   /* 已成功打开的事件数 */
    }
}

static struct read_data g_prev;

static void prof_emit(void) {
    struct read_data now;
    memset(&now, 0, sizeof(now));
    ssize_t n = read(g_fds[0], &now, sizeof(now));
    if (n <= 0 || g_out == NULL) return;

    fprintf(g_out, "%lld", g_phase_id);
    for (int i = 0; i < N_EVENTS; i++) {
        uint64_t v = 0;
        if (g_pos[i] >= 0) {
            v = now.values[g_pos[i]].value - g_prev.values[g_pos[i]].value;
        }
        fprintf(g_out, ",%llu", (unsigned long long)v);
    }
    /* 时间字段是自启用以来的累计值，这里输出本 phase 的增量；
       running < enabled 表示分组被时分复用，下游应按 enabled/running 还原。 */
    fprintf(g_out, ",%llu,%llu\n",
            (unsigned long long)(now.time_enabled - g_prev.time_enabled),
            (unsigned long long)(now.time_running - g_prev.time_running));
    g_prev = now;
    g_phase_id++;
}

static void prof_fini(void) {
    if (g_out == NULL) return;
    prof_emit();       /* 收尾：最后不足一个 phase 的部分 */
    fclose(g_out);
    g_out = NULL;
}

__attribute__((constructor)) static void prof_ctor(void) {
    prof_init();
    atexit(prof_fini);
}

/* clang -fsanitize-coverage=trace-pc 在每个基本块插入对本函数的调用 */
void __sanitizer_cov_trace_pc(void) {
    if (g_out == NULL) return;
    if (++g_bb_count >= g_phase_bb) {
        g_bb_count = 0;
        prof_emit();
    }
}

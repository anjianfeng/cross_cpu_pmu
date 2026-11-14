/* LACross 风格 phase 剖析运行时（IR 插桩版）。
 *
 * 与 phase_profiler.c 的区别：
 *   - 探针是 IR 层插入的 __phase_bb(id)，id 对应 phase_bb_table[] 里的
 *     静态结构（指令类型构成、后继块），因此除了 PMU 事件外，
 *     还能得到「这一 phase 在执行什么代码」的架构无关特征；
 *   - 每条基本块只在采样点累加静态特征（PHASE_SAMPLE，默认 8），
 *     把插桩开销压到可忽略，避免测得的总线/周期被探针本身主导；
 *   - 用相邻两次基本块编号判断条件分支的走向（succ[0]=真 / succ[1]=假），
 *     得到每 phase 的分支可预测性特征。
 *
 * 事件集：
 *   默认仍是 6 个 PERF_TYPE_HARDWARE 通用事件（ARM 侧采集口径不变）。
 *   设置环境变量 PHASE_EVENTS 可改用任意事件列表，元素支持三种写法：
 *     cycles / instructions / cache_refs / cache_misses / branches /
 *     branch_misses                      -> PERF_TYPE_HARDWARE
 *     L1-dcache-load-misses / LLC-...    -> PERF_TYPE_HW_CACHE 通用缓存事件
 *     topdown-be-bound / mem-stores ...  -> cpu_core PMU 的 sysfs 别名（原始编码）
 *   事件名写入 CSV 时统一加 x_ 前缀并把非字母数字换成下划线。
 *   某个事件打不开只会在该列写 0，不会中断其它事件。
 *
 * 编译：必须单独编译本文件，且不要加 -fsanitize-coverage，
 * 也不能让 ir_instrument.py 给它插入探针（否则自递归）。
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "phase_bb_table.h"

#define MAX_EVENTS 32
#define NAME_LEN 72
#define N_ACC 9       /* ld st alu fp vec div call br inst */

extern const struct phase_bb_info phase_bb_table[];
extern const unsigned int phase_bb_table_size;

struct ev_spec {
    char name[NAME_LEN];
    unsigned int type;
    uint64_t config;
    uint64_t config1;
};

static struct ev_spec g_ev[MAX_EVENTS];
static int g_nev = 0;
static int g_fds[MAX_EVENTS];
static int g_pos[MAX_EVENTS];
static int g_leader = -1;
static int g_rich = 0;

static long long g_phase_bb = 5000;
static long long g_bb_count = 0;
static long long g_phase_id = 0;
static long long g_sample_mask = 7;   /* 每 8 个基本块采样一次 */
static FILE *g_out = NULL;

static uint64_t g_acc[N_ACC];
static uint64_t g_acc_bb = 0;
static uint64_t g_acc_taken = 0;
static uint64_t g_acc_notaken = 0;
static int g_prev_id = -1;

/* 默认（旧口径）事件表，ARM 侧继续使用 */
static const char *BASE_NAMES[6] = {
    "cycles", "instructions", "cache_refs", "cache_misses",
    "branches", "branch_misses",
};
static const uint64_t BASE_CONFIGS[6] = {
    PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
    PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES,
    PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
};

/* perf 的通用缓存事件别名 -> (level | op<<8 | result<<16) */
static const struct {
    const char *name;
    uint64_t config;
} CACHE_ALIASES[] = {
#define CE(n, l, o, r) {n, (l) | ((o) << 8) | ((r) << 16)}
    CE("L1-dcache-loads", PERF_COUNT_HW_CACHE_L1D,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("L1-dcache-load-misses", PERF_COUNT_HW_CACHE_L1D,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("L1-dcache-stores", PERF_COUNT_HW_CACHE_L1D,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("L1-dcache-store-misses", PERF_COUNT_HW_CACHE_L1D,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("L1-icache-loads", PERF_COUNT_HW_CACHE_L1I,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("L1-icache-load-misses", PERF_COUNT_HW_CACHE_L1I,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("LLC-loads", PERF_COUNT_HW_CACHE_LL,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("LLC-load-misses", PERF_COUNT_HW_CACHE_LL,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("LLC-stores", PERF_COUNT_HW_CACHE_LL,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("LLC-store-misses", PERF_COUNT_HW_CACHE_LL,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("dTLB-loads", PERF_COUNT_HW_CACHE_DTLB,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("dTLB-load-misses", PERF_COUNT_HW_CACHE_DTLB,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("dTLB-stores", PERF_COUNT_HW_CACHE_DTLB,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("dTLB-store-misses", PERF_COUNT_HW_CACHE_DTLB,
       PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS),
    CE("iTLB-loads", PERF_COUNT_HW_CACHE_ITLB,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
    CE("iTLB-load-misses", PERF_COUNT_HW_CACHE_ITLB,
       PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
#undef CE
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

static int read_small_file(const char *path, char *buf, int n) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t got = fread(buf, 1, (size_t)n - 1, f);
    fclose(f);
    buf[got] = '\0';
    while (got > 0 && (buf[got - 1] == '\n' || buf[got - 1] == ' '))
        buf[--got] = '\0';
    return (int)got;
}

static uint64_t parse_num(const char *s) {
    return (uint64_t)strtoull(s, NULL, 0);
}

/* 解析 /sys/bus/event_source/devices/<pmu>/events/<alias>
 * 形如 "event=0x3c,umask=0x11,ldlat=3"。 */
static int resolve_pmu_alias(const char *pmu, const char *alias,
                             unsigned int *type, uint64_t *config,
                             uint64_t *config1) {
    char path[256], buf[512], typebuf[64];
    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/type", pmu);
    if (read_small_file(path, typebuf, sizeof(typebuf)) <= 0) return -1;
    *type = (unsigned int)parse_num(typebuf);

    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/events/%s",
             pmu, alias);
    if (read_small_file(path, buf, sizeof(buf)) <= 0) return -1;

    *config = 0;
    *config1 = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok;
         tok = strtok_r(NULL, ",", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) continue;
        *eq = '\0';
        uint64_t v = parse_num(eq + 1);
        if (!strcmp(tok, "event"))          *config |= v;
        else if (!strcmp(tok, "umask"))     *config |= (v & 0xff) << 8;
        else if (!strcmp(tok, "cmask"))     *config |= (v & 0xff) << 24;
        else if (!strcmp(tok, "edge")) { if (v) *config |= 1ULL << 18; }
        else if (!strcmp(tok, "inv"))  { if (v) *config |= 1ULL << 23; }
        else if (!strcmp(tok, "ldlat"))     *config1 |= v & 0xffff;
        else if (!strcmp(tok, "offcore_rsp")) *config1 = v;
        /* period / anythread 等对计数模式无影响，忽略 */
    }
    return 0;
}

static void csv_name(const char *src, char *dst, int n) {
    int k = snprintf(dst, n, "x_");
    for (const char *p = src; *p && k < n - 1; p++)
        dst[k++] = (isalnum((unsigned char)*p) || *p == '_') ? *p : '_';
    dst[k] = '\0';
}

static void add_spec(const char *name, unsigned int type, uint64_t config,
                     uint64_t config1) {
    if (g_nev >= MAX_EVENTS) return;
    struct ev_spec *e = &g_ev[g_nev++];
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->type = type;
    e->config = config;
    e->config1 = config1;
}

static int add_base_event(const char *tok) {
    for (int i = 0; i < 6; i++) {
        if (!strcmp(tok, BASE_NAMES[i])) {
            add_spec(BASE_NAMES[i], PERF_TYPE_HARDWARE, BASE_CONFIGS[i], 0);
            return 1;
        }
    }
    return 0;
}

static int add_cache_event(const char *tok) {
    for (unsigned i = 0; i < sizeof(CACHE_ALIASES) / sizeof(CACHE_ALIASES[0]); i++) {
        if (!strcmp(tok, CACHE_ALIASES[i].name)) {
            char name[NAME_LEN];
            csv_name(CACHE_ALIASES[i].name, name, sizeof(name));
            add_spec(name, PERF_TYPE_HW_CACHE, CACHE_ALIASES[i].config, 0);
            return 1;
        }
    }
    return 0;
}

static int add_pmu_event(const char *tok) {
    char pmu[64], alias[128], name[NAME_LEN];
    unsigned int type;
    uint64_t config, config1;
    const char *colon = strchr(tok, ':');
    if (colon) {
        int n = (int)(colon - tok);
        if (n <= 0 || n >= (int)sizeof(pmu)) return 0;
        memcpy(pmu, tok, n);
        pmu[n] = '\0';
        snprintf(alias, sizeof(alias), "%s", colon + 1);
    } else {
        snprintf(pmu, sizeof(pmu), "cpu_core");
        snprintf(alias, sizeof(alias), "%s", tok);
    }
    if (resolve_pmu_alias(pmu, alias, &type, &config, &config1) != 0)
        return 0;
    csv_name(alias, name, sizeof(name));
    add_spec(name, type, config, config1);
    return 1;
}

static void parse_events(const char *env) {
    char buf[2048];
    snprintf(buf, sizeof(buf), "%s", env);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok;
         tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ') tok++;
        if (!*tok) continue;
        if (add_base_event(tok) || add_cache_event(tok) || add_pmu_event(tok))
            continue;
        fprintf(stderr, "[phase_profiler_ir] 未知事件 %s（已跳过）\n", tok);
    }
}

static int open_event(unsigned type, uint64_t config, uint64_t config1,
                      int group_fd) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.config1 = config1;
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
    const char *sstr = getenv("PHASE_SAMPLE");
    const char *evstr = getenv("PHASE_EVENTS");
    if (kstr && atoll(kstr) > 0) g_phase_bb = atoll(kstr);
    if (sstr && atoll(sstr) > 0) g_sample_mask = atoll(sstr) - 1;
    if (!outpath) outpath = "phase.csv";

    if (evstr && *evstr) {
        g_rich = 1;
        parse_events(evstr);
        if (g_nev == 0) {
            fprintf(stderr, "[phase_profiler_ir] PHASE_EVENTS 无有效事件，"
                            "退回默认 6 事件\n");
            g_rich = 0;
        }
    }
    if (!g_rich)
        for (int i = 0; i < 6; i++)
            add_spec(BASE_NAMES[i], PERF_TYPE_HARDWARE, BASE_CONFIGS[i], 0);

    g_out = fopen(outpath, "w");
    if (!g_out) return;

    fprintf(g_out, "phase");
    for (int i = 0; i < g_nev; i++) fprintf(g_out, ",%s", g_ev[i].name);
    fprintf(g_out, ",t_enabled,t_running,"
                   "s_ld,s_st,s_alu,s_fp,s_vec,s_div,s_call,s_br,s_inst,"
                   "s_bb,s_taken,s_notaken\n");

    int npos = 0;
    for (int i = 0; i < g_nev; i++) {
        int fd = open_event(g_ev[i].type, g_ev[i].config, g_ev[i].config1,
                            g_leader);
        if (fd < 0) {
            fprintf(stderr, "[phase_profiler_ir] 事件 %s 不可用: %s\n",
                    g_ev[i].name, strerror(errno));
            g_pos[i] = -1;
            continue;
        }
        g_fds[i] = fd;
        /* 组读数按成功加入组的顺序排列：leader 为 0，之后依次递增 */
        g_pos[i] = npos++;
        if (g_leader < 0) g_leader = fd;
    }

    if (g_leader < 0) {
        fclose(g_out);
        g_out = NULL;
        return;
    }
    ioctl(g_leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(g_leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

static struct read_data g_prev;

static void prof_emit(void) {
    struct read_data now;
    memset(&now, 0, sizeof(now));
    if (g_leader < 0 || g_out == NULL) return;
    ssize_t n = read(g_leader, &now, sizeof(now));
    if (n <= 0) return;

    fprintf(g_out, "%lld", g_phase_id);
    for (int i = 0; i < g_nev; i++) {
        uint64_t v = 0;
        if (g_pos[i] >= 0 && now.values[g_pos[i]].value >=
                             g_prev.values[g_pos[i]].value)
            v = now.values[g_pos[i]].value - g_prev.values[g_pos[i]].value;
        fprintf(g_out, ",%llu", (unsigned long long)v);
    }
    fprintf(g_out, ",%llu,%llu",
            (unsigned long long)(now.time_enabled - g_prev.time_enabled),
            (unsigned long long)(now.time_running - g_prev.time_running));
    for (int i = 0; i < N_ACC; i++)
        fprintf(g_out, ",%llu", (unsigned long long)g_acc[i]);
    fprintf(g_out, ",%llu,%llu,%llu\n",
            (unsigned long long)g_acc_bb,
            (unsigned long long)g_acc_taken,
            (unsigned long long)g_acc_notaken);

    g_prev = now;
    g_phase_id++;
    memset(g_acc, 0, sizeof(g_acc));
    g_acc_bb = g_acc_taken = g_acc_notaken = 0;
}

static void prof_fini(void) {
    if (g_out == NULL) return;
    prof_emit();
    fclose(g_out);
    g_out = NULL;
}

__attribute__((constructor)) static void prof_ctor(void) {
    prof_init();
    atexit(prof_fini);
}

/* ir_instrument.py 在每个基本块首插入对本函数的调用。
 * id 由构建期分配，保证 < phase_bb_table_size。 */
void __phase_bb(int id) {
    if (g_out == NULL) return;
    int prev = g_prev_id;
    g_prev_id = id;

    if (((g_bb_count + 1) & g_sample_mask) == 0) {
        if ((unsigned)id < phase_bb_table_size) {
            const struct phase_bb_info *b = &phase_bb_table[id];
            g_acc[0] += b->n_ld;
            g_acc[1] += b->n_st;
            g_acc[2] += b->n_alu;
            g_acc[3] += b->n_fp;
            g_acc[4] += b->n_vec;
            g_acc[5] += b->n_div;
            g_acc[6] += b->n_call;
            g_acc[7] += b->n_br;
            g_acc[8] += b->n_inst;
            g_acc_bb++;
            if ((unsigned)prev < phase_bb_table_size) {
                const struct phase_bb_info *p = &phase_bb_table[prev];
                if (p->nsucc == 2) {
                    if (p->succ[0] == id)
                        g_acc_taken++;          /* 条件为真 */
                    else if (p->succ[1] == id)
                        g_acc_notaken++;        /* 条件为假 */
                }
            }
        }
    }

    if (++g_bb_count >= g_phase_bb) {
        g_bb_count = 0;
        prof_emit();
    }
}

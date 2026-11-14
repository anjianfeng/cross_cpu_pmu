# LACross 跨平台性能预测 — 实验复现代码

本目录包含复现论文实验所需的全部脚本与目录结构。实验目标：在源平台（x86）
采集被测程序的 PMU 硬件性能计数，训练学习模型，预测同一程序在目标平台
（飞腾 D2000 ARM）上的运行周期数（cycles）。

> **核心思路**：将程序切分为 phase（每累计 5000 个基本块），在每个 phase
> 上用 clang `-fsanitize-coverage=trace-pc` 插桩采集 6 个通用硬件事件
> （cycles / instructions / cache_refs / cache_misses / branches /
> branch_misses），再按动态指令进度对齐两平台数据，训练 GBDT 模型预测
> ARM cycles / x86 cycles 的比值。

> **分工**：本目录（`openlab/`）只放复现实验所需的脚本与数据；论文本身的
> Markdown 源文件、插图与 Word/PDF 构建脚本位于同级目录 `paper/`
> （`paper/paper_MDPI_Computers.md`、`paper/figures/`、`paper/scripts/make_*.py`）。

---

## 目录结构

```
openlab/
├── README.md                  # 本文件
├── benchmarks/                # 基准程序源码（需自行下载，见下文）
│   ├── polybench/             # PolyBench/C 4.2.1（30 程序，MIT）
│   ├── mibench/               # MiBench（7 个目标程序）
│   ├── SD-VBS/                # San Diego Vision Benchmark Suite（9 程序）
│   └── cpu2017/               # SPEC CPU2017（需授权，6 个 C/C++ 程序）
├── build/                     # 编译产物（插桩二进制，运行时生成）
├── data/                      # 采集的 PMU 数据
│   ├── phase_ir/x86/          #   phase 级基础 6 事件轨迹（x86）  [需重新采集]
│   ├── phase_ir/arm/          #   phase 级基础 6 事件轨迹（ARM）  [需重新采集]
│   ├── phase_ir_rich/x86/     #   phase 级 x86 富特征（topdown/TLB 等）[需重新采集]
│   ├── phase_ir_rich/arm/     #   指向 ARM 基础数据（软链接或复制）
│   ├── phase/x86|arm/         #   早期 phase 采集（非 IR 插桩版）[需重新采集]
│   ├── phase_base.csv.gz      #   phase 级预聚合表（161 程序，18 特征，已包含）
│   ├── phase_rich.csv.gz      #   phase 级预聚合表（159 程序，33 特征，已包含）
│   ├── mibench_v2/            #   整程序级 MiBench PMU 数据（7 程序，已包含）
│   ├── mibench/               #   整程序级 MiBench 首轮采集（7 程序，已包含）
│   ├── sdvbs/                 #   整程序级 SD-VBS PMU 数据（9 程序，已包含）
│   ├── polybench/             #   整程序级 PolyBench PMU 数据（30 程序，已包含）
│   └── spec/                  #   整程序级 SPEC PMU 数据（9 程序，已包含）
├── scripts/                   # 全部实验脚本
└── doc/                       # 过程记录（reviews/ 三批实验日志、gap_analysis.md、
                               # experiment_log.md 等）
```

> **图与结果 CSV**：`paper/figures/` 下同时放 7 张论文插图（.png）和 9 个逐程序
> 结果 CSV（.csv）。`make_paper_figures.py` / `make_paper_docx.py` 都从这里读。
> 读者如果想重建图或 PDF，只需 `cd paper/ && python3 scripts/make_paper_figures.py`，
> 无需重新采集数据。

> **数据说明**：
> - `data/{mibench_v2,mibench,sdvbs,polybench,spec}/` 已随仓库提供（共 124 个 CSV，
>   约 2.8 MB），每行是一次整程序采集的 perf 事件组计数（`program,group,rep,...`），
>   `*_x86.csv` / `*_arm.csv` 成对出现。`analyze_model.py`、`predict_pmu.py`、
>   `program_level_eval.py` 可直接运行，**无需任何采集**。
> - `data/phase_base.csv.gz`（2.5 MB）与 `data/phase_rich.csv.gz`（4.3 MB）是 **phase
>   级预聚合表**：161 / 159 个程序 × 100 个等进度区间，含 18 个基础特征与 15 个
>   top-down 富特征，外加 `y = log10(arm_cycles / x86_cycles)`。用
>   `load_aggregated("base")` / `load_aggregated("rich")` 直接加载，即可复现
>   Table 5–9 的全部 phase 级结果，**同样无需采集**；两份表由
>   `make_aggregated.py` 从原始轨迹生成。
> - `data/{phase,phase_ir,phase_ir_rich}/` 为空（仅含 `.gitkeep`）。这三套 phase 级
>   原始轨迹合计约 16 GB，超出仓库容量，需按下文说明重新采集。
> - `data/events_x86.txt`（perf 事件组定义）、`data/spec_manifest_*.json`（SPEC
>   运行清单）为采集元数据，同样已包含。
>
> `benchmarks/` 下的子目录为空（仅含 `.gitkeep`），需按下文说明下载基准程序。

---

## 环境要求

### 硬件

| 平台 | 要求 |
|---|---|
| 源平台 x86 | Intel CPU（实测为 i9-12900），支持 `perf`，`perf_event_paranoid=-1` |
| 目标平台 ARM | 飞腾 D2000（aarch64），支持 `perf`，SSH 可达 |

### 软件

**x86 源平台：**
- Python 3.8+，依赖：`numpy pandas scikit-learn`
- `clang-18`（含 aarch64 交叉编译目标 `--target=aarch64-linux-gnu`）
- `perf`（Linux perf tools）
- `sshpass`（用于远程执行 D2000 命令）

```bash
pip install numpy pandas scikit-learn
# clang-18 安装示例（Ubuntu/Debian）：
# wget https://apt.llvm.org/llvm.sh && bash llvm.sh 18
```

**ARM 目标平台（D2000）：**
- `perf`、`python3`
- 能接收 ssh/scp 上传的静态 ELF 并执行

---

## 第一步：获取基准程序

各套件需自行下载到 `benchmarks/` 对应目录：

| 套件 | 目录 | 获取方式 |
|---|---|---|
| PolyBench/C 4.2.1 | `benchmarks/polybench/` | `git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git benchmarks/polybench` |
| MiBench | `benchmarks/mibench/` | `git clone https://github.com/embecosm/mibench.git benchmarks/mibench` |
| SD-VBS | `benchmarks/SD-VBS/` | `git clone https://github.com/leochanj105/SD-VBS.git benchmarks/SD-VBS` |
| SPEC CPU2017 | `benchmarks/cpu2017/` | 需商业授权，安装后将 `benchspec/` 置于此目录 |

> 目标程序清单与路径映射见 `scripts/phase_collect_others.py` 的 `CMDS` / `CWD`。

---

## 第二步：构建 phase 插桩二进制

用 clang-18 + `-fsanitize-coverage=trace-pc` 为每个程序编译 x86 与 aarch64
两份插桩二进制。运行时每累计 `PHASE_BB`（默认 5000）个基本块读取一次
硬件计数器。

### PolyBench（30 程序）

```bash
bash scripts/phase_build.sh            # 输出到 build/phase_pb/{x86,arm}/
# 可选：DATASET=MEDIUM_DATASET 或 LARGE_DATASET
```

### MiBench（7 程序）

```bash
bash scripts/phase_build_mibench.sh    # 输出到 build/phase_mibench/{x86,arm}/
```

### SD-VBS（9 程序）

```bash
bash scripts/phase_build_sdvbs.sh      # 输出到 build/phase_sdvbs/{x86,arm}/
```

### SPEC CPU2017（6 程序）

```bash
# 先解析 SPEC 的 Makefile.spec 并重编为插桩版
python3 scripts/spec_phase_build.py    # 输出到 build/phase_spec/{x86,arm}/
```

### LLVM test-suite（可选扩展训练集）

```bash
python3 scripts/phase_ir_llvm.py build
```

> 所有构建脚本都需要 `clang-18` 在 PATH 中。构建产物为静态 ELF，
> ARM 版可直接上传 D2000 运行。

---

## 第三步：采集 phase 级 PMU 数据

### 基础 6 事件（两平台都采集）

```bash
# PolyBench
python3 scripts/phase_collect.py --platform x86 \
    --bindir build/phase_pb/x86 --outdir data/phase/x86
python3 scripts/phase_collect.py --platform arm \
    --bindir build/phase_pb/arm --outdir data/phase/arm

# MiBench + SD-VBS
python3 scripts/phase_collect_others.py --platform x86 \
    --bindir build/phase_mibench/x86:build/phase_pb/x86:build/phase_sdvbs/x86 \
    --outdir data/phase/x86
python3 scripts/phase_collect_others.py --platform arm \
    --bindir build/phase_mibench/arm:build/phase_pb/arm:build/phase_sdvbs/arm \
    --outdir data/phase/arm

# SPEC
python3 scripts/spec_phase_collect.py --platform x86 --phase-bb 5000
python3 scripts/spec_phase_collect.py --platform arm --phase-bb 5000
```

> ARM 采集通过 ssh 在 D2000 上执行。**采集前需配置 D2000 的 SSH 连接信息**
> （`D2000_USER` / `D2000_HOST` / `D2000_PORT` / `D2000_PASS` 环境变量，或直接在
> `scripts/phase_ir_llvm.py`、`scripts/phase_collect.py` 等顶部修改默认值）。

### IR 结构特征 + x86 富特征（提升精度）

```bash
# 1) 用 ir_instrument.py 在 LLVM IR 层插桩，采集结构特征
#    （phase_ir_build.py / phase_ir_build_pb.py / phase_ir_build_others.py）
python3 scripts/phase_ir_build_pb.py    # PolyBench IR 插桩版

# 2) 采集基础 6 事件 + 结构特征 → data/phase_ir/
python3 scripts/phase_collect.py --platform x86 \
    --bindir build/phase_ir/x86 --outdir data/phase_ir/x86
python3 scripts/phase_collect.py --platform arm \
    --bindir build/phase_ir/arm --outdir data/phase_ir/arm

# 3) x86 侧追加 topdown / TLB / LLC 等富特征 → data/phase_ir_rich/
python3 scripts/phase_collect_rich.py
```

### 整程序级 PMU（可选，用于对比）

```bash
# 各套件两平台全事件采集
bash scripts/run_polybench_collect.sh x86 build/polybench_x86 data/polybench 3
bash scripts/run_polybench_collect.sh arm build/polybench_arm data/polybench 3
bash scripts/run_mibench_collect.sh x86 build/mibench_x86 data/mibench 3
bash scripts/run_sdvbs_collect.sh x86 build/sdvbs_x86 data/sdvbs 3
python3 scripts/run_spec_collect.py
```

---

## 第四步：分析与建模

### 主分析脚本（phase 级 LOPO 交叉验证）

```bash
# 基础 6 事件
python3 -W ignore scripts/analyze_phase.py --data data/phase_ir

# 基础 + x86 富特征（推荐，论文主结果）
python3 -W ignore scripts/analyze_phase.py --data data/phase_ir_rich --rich
```

输出内容：
- 常数比值基线、岭回归、Lasso、GBDT 的 MAPE / log 相关 / RMSLE
- GBDT 超参搜索结果
- 各程序的预测误差明细

> **无需采集即可复现**：`data/phase_base.csv.gz` 与 `data/phase_rich.csv.gz` 是
> phase 级预聚合表，`holdout_eval.py`、`lopo_table.py`、`m3_controls.py` 均支持
> `--agg base|rich` 直接读取，跳过 16 GB 原始轨迹的采集与重采样：
>
> ```bash
> python3 -W ignore scripts/holdout_eval.py --agg rich --rich   # Table 7 Rich 列
> python3 -W ignore scripts/lopo_table.py  --agg rich           # Table 7 LOPO 列
> ```
>
> 实测本包内 held-out 为 14.51%（与论文一致），LOPO 为 17.38%（论文 17.40%，
> 0.02 的差异来自预聚合 CSV 的浮点舍入）。LOPO 需逐程序训练 159 折，约 8 分钟。

### 其他分析脚本

| 脚本 | 用途 |
|---|---|
| `holdout_eval.py` | 留出套件验证（train on 3 suites, test on 1） |
| `lopo_table.py` | LOPO 逐程序结果表 |
| `m3_controls.py` | 训练池组成控制实验 + 安慰剂检验 |
| `matched_ci.py` | 配对 LOPO 置信区间 |
| `pclsrr.py` | 论文 PCLSLR 方法复现（局部加权回归） |
| `clslr_rerun.py` | CLSLR 重跑实验 |
| `program_level_eval.py` | 程序级基线与 epsilon-PCLSLR |
| `dtw_per_program.py` | DTW 对齐审计 |
| `domain_adapt.py` | 域适配实验 |
| `within_sd.py` | 同套件内标准差分析 |
| `test_dtw_quality.py` | DTW 对齐质量测试 |

---

## 第五步：验证结果（可选）

`doc/figures/` 中已包含论文各图背后的逐程序结果 CSV。用上面带 `--agg` 的命令
重新计算，再与归档 CSV 比对，即可确认本机复现的数与论文一致：

```bash
python3 -W ignore scripts/lopo_table.py --agg rich --dump-csv /tmp/lopo.csv
python3 -c "
import csv
a=[float(r['ape']) for r in csv.DictReader(open('/tmp/lopo.csv'))]
b=[float(r['ape']) for r in csv.DictReader(open('doc/figures/lopo_per_program_rich.csv'))]
print('逐程序平均 APE: 复算 %.2f%%  归档 %.2f%%' % (sum(a)/len(a), sum(b)/len(b)))
"
```

期望两者一致（LOPO 17.40%）。

---

## 脚本清单

### 构建与插桩

| 脚本 | 说明 |
|---|---|
| `phase_build.sh` | PolyBench phase 插桩构建（clang-18 + trace-pc） |
| `phase_build_mibench.sh` | MiBench phase 插桩构建 |
| `phase_build_sdvbs.sh` | SD-VBS phase 插桩构建 |
| `phase_ir_build.py` | IR 层结构插桩构建（通用框架） |
| `phase_ir_build_pb.py` | PolyBench IR 插桩构建 |
| `phase_ir_build_others.py` | MiBench + SD-VBS IR 插桩构建 |
| `phase_ir_llvm.py` | LLVM test-suite IR 插桩构建与采集 |
| `ir_instrument.py` | LLVM IR 基本块插桩 + 结构特征表生成 |
| `spec_phase_build.py` | SPEC CPU2017 phase 插桩构建 |
| `phase_profiler.c` | phase 运行时 profiler（6 通用事件） |
| `phase_profiler_ir.c` | phase 运行时 profiler（含 IR 结构特征） |
| `phase_bb_table.h` | 基本块结构特征表（由 ir_instrument.py 生成） |

### 数据采集

| 脚本 | 说明 |
|---|---|
| `collect_pmu.py` | 整程序级全 PMU 事件采集驱动 |
| `phase_collect.py` | PolyBench phase 级采集 |
| `phase_collect_others.py` | MiBench + SD-VBS phase 级采集 |
| `phase_collect_rich.py` | x86 富特征 phase 采集（topdown/TLB/LLC） |
| `spec_phase_collect.py` | SPEC phase 级采集 |
| `spec_setup.py` | SPEC 运行目录准备 |
| `run_mibench_collect.sh` | MiBench 整程序采集驱动 |
| `run_sdvbs_collect.sh` | SD-VBS 整程序采集驱动 |
| `run_polybench_collect.sh` | PolyBench 整程序采集驱动 |
| `run_spec_collect.py` | SPEC 整程序采集驱动 |

### 分析与建模

| 脚本 | 说明 |
|---|---|
| `analyze_phase.py` | phase 级主分析（LOPO + GBDT 超参搜索） |
| `analyze_model.py` | 整程序级建模分析 |
| `holdout_eval.py` | 留出套件评估 |
| `homog_eval.py` | 同质性评估 |
| `lopo_table.py` | LOPO 结果表 |
| `m3_controls.py` | 训练池控制 + 安慰剂 |
| `matched_ci.py` | 配对 LOPO CI |
| `pclsrr.py` | PCLSLR 局部回归 |
| `clslr_rerun.py` | CLSLR 重跑 |
| `predict_pmu.py` | PMU 预测工具 |
| `program_level_eval.py` | 程序级评估 |
| `domain_adapt.py` | 域适配 |
| `within_sd.py` | 套件内标准差 |
| `dtw_per_program.py` | DTW 对齐审计 |
| `test_dtw_quality.py` | DTW 质量测试 |

---

## 注意事项

1. **D2000 连接信息**：SSH 采集脚本在文件顶部定义了 `REMOTE` / `RPASS` /
   `D2000_*` 等常量。运行前请替换为自己 D2000 板的实际用户、主机、端口
   与密码；也可通过同名环境变量覆盖（如 `export D2000_PASS=...`）。

2. **绑核**：x86 侧必须绑 P-core（`taskset -c 0`），避免 E-core 导致
   `cpu_core/*` 事件 `<not counted>`。脚本已内置。

3. **静态链接**：所有 ARM 二进制必须 `-static`，避免 D2000 上 glibc 版本
   差异。构建脚本已处理。

4. **数据集尺寸**：PolyBench 用 `MEDIUM_DATASET`（8 个短程序用
   `LARGE_DATASET`）；SD-VBS 用 `qcif`；SPEC 用 `test`。详见各构建脚本。

5. **相位对齐**：两平台机器级基本块数不同，`analyze_phase.py` 按动态
   指令进度重采样到 100 个等进度区间后对齐，而非按行号对齐。

6. **Python 环境**：分析脚本依赖 `numpy`、`pandas`、`scikit-learn`，
   建议使用虚拟环境。

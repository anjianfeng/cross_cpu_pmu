#!/usr/bin/env bash
# 为 MiBench 构建 phase 插桩版本（x86 + aarch64，同 clang-18 -O2）
#
# 用法: bash scripts/phase_build_mibench.sh [输出目录]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/benchmarks/mibench"
OUT="${1:-$ROOT/build/phase_mibench}"
CC="${CC:-clang-18}"
CLEAN_ENV=(env -i PATH=/usr/bin:/bin)

mkdir -p "$OUT/x86" "$OUT/arm"

# 预编译 profiler 运行时
"${CLEAN_ENV[@]}" "$CC" -O2 -c "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_x86.o"
"${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static -c \
    "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_arm.o"

# 条目：(输出名 源码目录 源文件列表...)
PROGS=(
  "aes       security/rijndael        aes.c aesxam.c"
  "crc       telecomm/CRC32           crc_32.c"
  "dijkstra  network/dijkstra         dijkstra_large.c"
  "fft       telecomm/FFT             main.c fftmisc.c fourierf.c"
  "patricia  network/patricia         patricia.c patricia_test.c"
  "qsort     automotive/qsort          qsort_large.c"
  "sha       security/sha             sha_driver.c sha.c"
)

ok=0; fail=0
for entry in "${PROGS[@]}"; do
  read -r name dir rest <<< "$entry"
  src_dir="$SRC/$dir"
  src_files=($rest)

  echo "=== $name ($dir) ==="

  # x86: 每源文件单独编译（带插桩），再链接
  # -std=gnu89 -w 模拟原 gcc 的宽松模式（MiBench 老代码依赖隐式声明）
  obj_x86=()
  for sf in "${src_files[@]}"; do
    "${CLEAN_ENV[@]}" "$CC" -O2 -std=gnu89 -w -c \
        -fsanitize-coverage=trace-pc \
        "$src_dir/$sf" -o "$OUT/${name}_${sf%.c}.x86.o" 2>"$OUT/${name}.x86.err" \
        || { echo "  [x86 编译失败] $sf"; tail -3 "$OUT/${name}.x86.err"; fail=$((fail+1)); continue 2; }
    obj_x86+=("$OUT/${name}_${sf%.c}.x86.o")
  done
  "${CLEAN_ENV[@]}" "$CC" "${obj_x86[@]}" "$OUT/prof_x86.o" \
      -lm -o "$OUT/x86/$name" 2>>"$OUT/${name}.x86.err" \
      && { echo "  [x86 OK]"; ok=$((ok+1)); } \
      || { echo "  [x86 链接失败]"; tail -3 "$OUT/${name}.x86.err"; fail=$((fail+1)); }

  # arm: 同上，加 --target=aarch64-linux-gnu -static
  obj_arm=()
  for sf in "${src_files[@]}"; do
    "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static \
        -std=gnu89 -w -c -fsanitize-coverage=trace-pc \
        "$src_dir/$sf" -o "$OUT/${name}_${sf%.c}.arm.o" 2>"$OUT/${name}.arm.err" \
        || { echo "  [arm 编译失败] $sf"; tail -3 "$OUT/${name}.arm.err"; fail=$((fail+1)); continue; }
    obj_arm+=("$OUT/${name}_${sf%.c}.arm.o")
  done
  "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -static \
      "${obj_arm[@]}" "$OUT/prof_arm.o" \
      -lm -o "$OUT/arm/$name" 2>>"$OUT/${name}.arm.err" \
      && { echo "  [arm OK]"; ok=$((ok+1)); } \
      || { echo "  [arm 链接失败]"; tail -3 "$OUT/${name}.arm.err"; fail=$((fail+1)); }
done

echo "=== 完成: 成功 $ok, 失败 $fail ==="

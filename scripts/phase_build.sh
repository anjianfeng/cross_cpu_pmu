#!/usr/bin/env bash
# 构建 phase 插桩版 PolyBench（x86 与 aarch64 同用 clang-18 + -fsanitize-coverage=trace-pc）
#
# 用法:
#   bash scripts/phase_build.sh [输出目录]
#   DATASET=MEDIUM_DATASET bash scripts/phase_build.sh
#
# 产物:
#   <out>/x86/<prog>   本机可执行
#   <out>/arm/<prog>   aarch64 静态可执行（可直接上传 D2000）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/benchmarks/polybench"
UTIL="$SRC/utilities"
DATASET="${DATASET:-MEDIUM_DATASET}"
# 这些程序 MEDIUM 尺寸下运行不足 1ms（phase 数过少），改用 LARGE 数据集
BIG_PROGS="${BIG_PROGS:-atax bicg mvt gemver gesummv trisolv durbin jacobi-1d}"
OUT="${1:-$ROOT/build/phase_pb}"
CC="${CC:-clang-18}"

CLEAN_ENV=(env -i PATH=/usr/bin:/bin)

mkdir -p "$OUT/x86" "$OUT/arm"

echo "[1/3] 编译 polybench 公共库与 profiler 运行时..."
"${CLEAN_ENV[@]}" "$CC" -O2 -c -I"$UTIL" "$UTIL/polybench.c" -o "$OUT/pb_x86.o"
"${CLEAN_ENV[@]}" "$CC" -O2 -c "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_x86.o"
"${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static -c -I"$UTIL" "$UTIL/polybench.c" -o "$OUT/pb_arm.o"
"${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static -c "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_arm.o"

echo "[2/3] 构建 30 个 PolyBench（数据集 $DATASET）..."
ok=0; fail=0
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  src="$SRC/${rel#./}"
  name="$(basename "$rel" .c)"
  dir="$(dirname "$src")"
  ds="$DATASET"
  for big in $BIG_PROGS; do
    [ "$name" = "$big" ] && ds="LARGE_DATASET"
  done
  # 插桩放到编译阶段；链接阶段不带该 flag，否则 clang 会要求 ubsan 运行时
  if "${CLEAN_ENV[@]}" "$CC" -O2 -c -fsanitize-coverage=trace-pc -D"$ds" \
        -I"$UTIL" -I"$dir" "$src" -o "$OUT/$name.x86.o" 2>"$OUT/$name.x86.err" \
     && "${CLEAN_ENV[@]}" "$CC" "$OUT/$name.x86.o" "$OUT/pb_x86.o" "$OUT/prof_x86.o" -lm \
        -o "$OUT/x86/$name" 2>>"$OUT/$name.x86.err"; then
    ok=$((ok+1))
  else
    echo "  [x86 失败] $name"; tail -3 "$OUT/$name.x86.err"; fail=$((fail+1))
  fi
  if "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static -c \
        -fsanitize-coverage=trace-pc -D"$ds" \
        -I"$UTIL" -I"$dir" "$src" -o "$OUT/$name.arm.o" 2>"$OUT/$name.arm.err" \
     && "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -static \
        "$OUT/$name.arm.o" "$OUT/pb_arm.o" "$OUT/prof_arm.o" -lm \
        -o "$OUT/arm/$name" 2>>"$OUT/$name.arm.err"; then
    ok=$((ok+1))
  else
    echo "  [arm 失败] $name"; tail -3 "$OUT/$name.arm.err"; fail=$((fail+1))
  fi
done < "$UTIL/benchmark_list"

echo "[3/3] 完成：成功 $ok 个（30 程序 × 2 平台），失败 $fail 个 -> $OUT"

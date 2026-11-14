#!/usr/bin/env bash
# 为 SD-VBS 构建 phase 插桩版本（x86 + aarch64）
#
# 思路：每源文件单独 clang-18 -fsanitize-coverage=trace-pc 编译，
# 然后和 phase_profiler.o 一起链接（链接阶段不带 -fsanitize-coverage，
# 避免链接 libclang_rt.ubsan）。
#
# SD-VBS 通用编译命令（见 Makefile.common）：
#   COMPILE_C = $(CC) $(CFLAGS) -lm -O2 $(INCLUDES)
#   其中 CFLAGS += -DGCC -D$(INPUT) -DCHECK，INCLUDES = -Icommon/c -I$(BMARK)/src/c
#   实际编译：$(COMPILE_C) $(COMMON_SRC) $(C_SRC) -lm -w -o $(BMARK)
#
# INPUT 用 fullhd（最高分辨率，与整程序采集一致）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/benchmarks/SD-VBS"
OUT="${1:-$ROOT/build/phase_sdvbs}"
CC="${CC:-clang-18}"
CLEAN_ENV=(env -i PATH=/usr/bin:/bin)

INPUT="${INPUT:-fullhd}"
mkdir -p "$OUT/x86" "$OUT/arm"

echo "[1/3] 预编译 profiler 运行时..."
"${CLEAN_ENV[@]}" "$CC" -O2 -c "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_x86.o"
"${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static -c \
    "$ROOT/scripts/phase_profiler.c" -o "$OUT/prof_arm.o"

echo "[2/3] 编译 SD-VBS common/c/*.c ..."
for sf in "$SRC/common/c"/*.c; do
  "${CLEAN_ENV[@]}" "$CC" -O2 -DGCC -D"$INPUT" -DCHECK -w \
      -c -fsanitize-coverage=trace-pc \
      -I"$SRC/common/c" \
      "$sf" -o "$OUT/common_$(basename $sf .c).x86.o" 2>/dev/null || true
  "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static \
      -DGCC -D"$INPUT" -DCHECK -w -c -fsanitize-coverage=trace-pc \
      -I"$SRC/common/c" \
      "$sf" -o "$OUT/common_$(basename $sf .c).arm.o" 2>/dev/null || true
done
COMMON_X86=$(ls "$OUT"/common_*.x86.o)
COMMON_ARM=$(ls "$OUT"/common_*.arm.o)

PROGRAMS=(disparity localization mser multi_ncut sift stitch svm texture_synthesis tracking)
ok=0; fail=0

echo "[3/3] 构建 9 个 SD-VBS 程序（INPUT=$INPUT）..."
for name in "${PROGRAMS[@]}"; do
  src_dir="$SRC/benchmarks/$name/src/c"
  src_files=($(ls "$src_dir"/*.c 2>/dev/null))
  if [ ${#src_files[@]} -eq 0 ]; then
    echo "  [$name] 无 C 源文件"
    fail=$((fail+1)); continue
  fi

  obj_x86=()
  obj_arm=()
  for sf in "${src_files[@]}"; do
    "${CLEAN_ENV[@]}" "$CC" -O2 -DGCC -D"$INPUT" -DCHECK -w \
        -c -fsanitize-coverage=trace-pc \
        -I"$SRC/common/c" -I"$src_dir" \
        "$sf" -o "$OUT/${name}_$(basename $sf .c).x86.o" 2>"$OUT/${name}.x86.err" \
        || { echo "  [$name x86 编译失败] $(basename $sf)"; fail=$((fail+1)); continue 2; }
    obj_x86+=("$OUT/${name}_$(basename $sf .c).x86.o")

    "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -O2 -static \
        -DGCC -D"$INPUT" -DCHECK -w -c -fsanitize-coverage=trace-pc \
        -I"$SRC/common/c" -I"$src_dir" \
        "$sf" -o "$OUT/${name}_$(basename $sf .c).arm.o" 2>"$OUT/${name}.arm.err" \
        || { echo "  [$name arm 编译失败] $(basename $sf)"; fail=$((fail+1)); continue 2; }
    obj_arm+=("$OUT/${name}_$(basename $sf .c).arm.o")
  done

  # texture_synthesis 的 xloopin/yloopin 在 texture.c 和 script_*.c 里重复定义
  extra_ld=""
  if [ "$name" = "texture_synthesis" ]; then
    extra_ld="-Wl,--allow-multiple-definition"
  fi

  # x86 链接（动态）
  "${CLEAN_ENV[@]}" "$CC" ${COMMON_X86[@]} ${obj_x86[@]} "$OUT/prof_x86.o" \
      $extra_ld -lm -o "$OUT/x86/$name" 2>>"$OUT/${name}.x86.err" \
      && { echo "  [$name x86 OK]"; ok=$((ok+1)); } \
      || { echo "  [$name x86 链接失败]"; tail -3 "$OUT/${name}.x86.err"; fail=$((fail+1)); }

  # arm 链接（静态）
  "${CLEAN_ENV[@]}" "$CC" --target=aarch64-linux-gnu -static \
      ${COMMON_ARM[@]} ${obj_arm[@]} "$OUT/prof_arm.o" \
      $extra_ld -lm -o "$OUT/arm/$name" 2>>"$OUT/${name}.arm.err" \
      && { echo "  [$name arm OK]"; ok=$((ok+1)); } \
      || { echo "  [$name arm 链接失败]"; tail -3 "$OUT/${name}.arm.err"; fail=$((fail+1)); }
done

echo "=== 完成: 成功 $ok, 失败 $fail ==="

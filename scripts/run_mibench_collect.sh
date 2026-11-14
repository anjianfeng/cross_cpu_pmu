#!/usr/bin/env bash
# MiBench 7 程序全事件批量采集驱动
# 用法: bash run_mibench_collect.sh <platform: x86|arm> <mibench根目录> <输出目录> [每组重复次数]
set -u
PLATFORM=$1
ROOT=$(readlink -f "$2")
OUTDIR=$(readlink -f "$3")
REPS=${4:-3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# name|相对目录|命令（不含输出重定向，统一走 --stdout-null）
PROGRAMS=(
  "aes|security/rijndael|./rijndael input_large.asc output_large.enc e 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321"
  "crc|telecomm/CRC32|./crc ../adpcm/data/large.pcm"
  "dijkstra|network/dijkstra|./dijkstra_large input.dat"
  "fft|telecomm/FFT|./fft 8 32768"
  "patricia|network/patricia|./patricia large.udp"
  "qsort|automotive/qsort|./qsort_large input_large.dat"
  "sha|security/sha|./sha input_large.asc"
)

mkdir -p "$OUTDIR"
for entry in "${PROGRAMS[@]}"; do
  IFS='|' read -r name dir cmd <<< "$entry"
  # ARM 平台使用 x86 本机交叉编译的 <prog>_arm 二进制（统一工具链 GCC 13.3）
  if [ "$PLATFORM" = "arm" ]; then
    cmd=$(echo "$cmd" | sed 's|^\./\([a-z_0-9]*\)|./\1_arm|')
  fi
  echo "=== [$PLATFORM] $name ($dir): $cmd ==="
  (cd "$ROOT/$dir" && python3 "$SCRIPT_DIR/collect_pmu.py" \
      --platform "$PLATFORM" --name "$name" \
      --cmd $cmd --stdout-null \
      --out "$OUTDIR/${name}_${PLATFORM}.csv" --reps "$REPS")
done
echo "全部完成 -> $OUTDIR"

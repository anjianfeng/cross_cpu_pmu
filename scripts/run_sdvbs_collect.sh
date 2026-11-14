#!/usr/bin/env bash
# SD-VBS 9 程序全事件批量采集驱动
# 用法: bash run_sdvbs_collect.sh <platform: x86|arm> <bin目录> <数据根目录> <输出目录> [重复次数]
#   数据根目录下每个程序一个子目录（内容为 data/sim 的数据），例如 <数据根>/sift/1.bmp
set -u
PLATFORM=$1
BIN_DIR=$(readlink -f "$2")
DATA_ROOT=$(readlink -f "$3")
OUTDIR=$(readlink -f "$4")
REPS=${5:-3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR=/tmp/sdvbs_res
mkdir -p "$RESULT_DIR" "$OUTDIR"

# 9 个 SD-VBS 程序，统一使用 sim 数据集（pca 需特殊参数，不纳入）
PROGRAMS=(disparity localization mser multi_ncut sift stitch svm texture_synthesis tracking)

for p in "${PROGRAMS[@]}"; do
  if [ "$PLATFORM" = "arm" ]; then
    BIN="$BIN_DIR/${p}_arm"
  else
    BIN="$BIN_DIR/$p"
  fi
  echo "=== [$PLATFORM] $p: $BIN $DATA_ROOT/$p ==="
  python3 "$SCRIPT_DIR/collect_pmu.py" \
      --platform "$PLATFORM" --name "$p" \
      --cmd "$BIN" "$DATA_ROOT/$p" "$RESULT_DIR" --stdout-null \
      --out "$OUTDIR/${p}_${PLATFORM}.csv" --reps "$REPS"
done
echo "全部完成 -> $OUTDIR"

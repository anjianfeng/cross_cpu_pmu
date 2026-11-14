#!/usr/bin/env bash
# PolyBench 30 程序两平台全事件采集驱动
# 用法: bash run_polybench_collect.sh <platform> <bin_dir> <输出目录> [重复次数]
set -u
PLATFORM=$1
BIN_DIR=$(readlink -f "$2")
OUTDIR=$(readlink -f "$3")
REPS=${4:-3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTDIR"

# 30 个程序按运行时长排序（先跑快的把慢的留给后台）
PROGRAMS=(
  durbin cholesky ludcmp trisolv syrk mvt atax bicg doitgen jacobi-1d seidel-2d heat-3d jacobi-2d fdtd-2d adi
  gemver gesummv symm syr2k trmm gemm 3mm deriche 2mm floyd-warshall
  correlation gramschmidt covar covariance nussinov
)

i=0
for p in "${PROGRAMS[@]}"; do
  i=$((i+1))
  if [ "$PLATFORM" = "arm" ]; then
    BIN="$BIN_DIR/${p}_arm"
  else
    BIN="$BIN_DIR/$p"
  fi
  echo "=== [$PLATFORM] ($i/${#PROGRAMS[@]}) $p ==="
  python3 "$SCRIPT_DIR/collect_pmu.py" \
      --platform "$PLATFORM" --name "$p" \
      --stdout-null \
      --out "$OUTDIR/${p}_${PLATFORM}.csv" --reps "$REPS" \
      --cmd "$BIN"
done
echo "全部完成 -> $OUTDIR"

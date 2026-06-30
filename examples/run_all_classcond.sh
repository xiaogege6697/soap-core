#!/usr/bin/env bash
# v0.7.6 class-conditional 38 runs 并行调度（CPU）
# A18 Pro 6核：每 run OMP_NUM_THREADS=1 限单线程避免抢核，并行度默认 5（留1核给系统）
# 用法: bash examples/run_all_classcond.sh [并行度]
#   断点续跑：跳过已存在的 examples/cc_seed*_*.csv
set -u
cd "$(dirname "$0")/.." || exit 1   # cd 到 soap-core 项目根
PY="$(pwd)/.venv/bin/python"

run_one() {
  local seed="$1" cond="$2"
  local out="examples/cc_seed${seed}_${cond}.csv"
  local nlines
  nlines=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ -s "$out" ] && [ "$nlines" -ge 281 ]; then
    echo "[skip] seed=$seed $cond (完整 ${nlines} 行)"; return 0
  fi
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" -m soap.apps.training.hf_classcond_experiment \
      --seed "$seed" --condition "$cond" --steps 120 --batch-size 16 \
      --record-every 3 --probe-size 128 --device cpu --output "$out" \
      > "examples/_log_seed${seed}_${cond}.txt" 2>&1
  if [ -s "$out" ]; then
    echo "[done] seed=$seed $cond"
  else
    echo "[FAIL] seed=$seed $cond (见 examples/_log_seed${seed}_${cond}.txt)"
  fi
}
export -f run_one
export PY

# 任务清单：normal 42-73 (32) + condition_b 42-47 (6) = 38 runs
{ for s in $(seq 42 73); do echo "$s normal"; done
  for s in $(seq 42 47); do echo "$s condition_b"; done
} | xargs -n 2 -P "${1:-5}" bash -c 'run_one "$@"' _

echo "=== 调度结束，CSV 计数（预期 38）==="
ls examples/cc_seed*_*.csv 2>/dev/null | wc -l

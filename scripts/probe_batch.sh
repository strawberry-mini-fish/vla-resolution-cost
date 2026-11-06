#!/usr/bin/env bash
# 显存探针：测出这张卡能跑多大 batch。
#
# 用法:
#   bash scripts/probe_batch.sh              # 默认测 4 6 8
#   bash scripts/probe_batch.sh 2 4 6 8 12   # 指定候选值
#
# 输出每个 batch 是 OK / OOM / FAIL，以及显存峰值。
# 把结果填进项目手册 Week 0 的表格里。

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

BATCHES=("$@")
if [ ${#BATCHES[@]} -eq 0 ]; then
  BATCHES=(4 6 8)
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "=============================================="
echo " 显存探针 · $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " 候选 batch: ${BATCHES[*]}"
echo "=============================================="
echo

RESULTS=()

for BS in "${BATCHES[@]}"; do
  echo "------ batch_size=$BS ------"
  LOG="$TMP/train_$BS.log"
  MEMLOG="$TMP/mem_$BS.log"
  : > "$MEMLOG"

  # 后台记录显存，每秒一个采样
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$MEMLOG" 2>/dev/null &
  POLL=$!

  timeout 900 lerobot-train \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/libero \
    --output_dir="$TMP/run_$BS" \
    --batch_size="$BS" \
    --steps=30 \
    --save_checkpoint=false \
    --save_freq=0 \
    --env_eval_freq=0 \
    --num_workers=2 \
    --accelerator.mixed_precision=bf16 \
    > "$LOG" 2>&1
  RC=$?

  kill $POLL 2>/dev/null
  wait $POLL 2>/dev/null
  sleep 1

  PEAK=$(sort -n "$MEMLOG" 2>/dev/null | tail -1)
  PEAK=${PEAK:-0}

  if [ $RC -eq 0 ]; then
    echo "  → OK    peak ${PEAK} MiB"
    RESULTS+=("$BS|OK|$PEAK")
  elif grep -qi "out of memory" "$LOG"; then
    echo "  → OOM   peak ${PEAK} MiB"
    RESULTS+=("$BS|OOM|$PEAK")
  else
    echo "  → FAIL  (rc=$RC)  日志尾部:"
    tail -n 12 "$LOG" | sed 's/^/      /'
    RESULTS+=("$BS|FAIL|$PEAK")
  fi
  echo

  # 清掉临时 checkpoint，避免占满磁盘
  rm -rf "$TMP/run_$BS" 2>/dev/null || true
done

echo "=============================================="
printf "%-8s %-8s %s\n" "batch" "状态" "显存峰值(MiB)"
echo "----------------------------------------------"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r b s p <<< "$r"
  printf "%-8s %-8s %s\n" "$b" "$s" "$p"
done
echo "=============================================="
echo
echo "把最大的 OK 值填进 configs/experiments.yaml 的 batch_size。"
echo "如果连最小 batch 都 OOM，见项目手册第 11 节预案。"

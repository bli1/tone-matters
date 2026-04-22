#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/jianzhiy/Final"
BENCH="$BASE_DIR/Final Benchmark/benchmark_qwen2_5_vl.py"
ANN="$BASE_DIR/final_annotations_800.json"
IMG_DIR="$BASE_DIR/Final Dataset"
OUT_DIR="$BASE_DIR/results/qwen2_5_vl"
LOG_DIR="$OUT_DIR/logs"

NUM_SHARDS="${NUM_SHARDS:-8}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -f "$ANN" ]]; then
  python "$BASE_DIR/Final Benchmark/build_final_annotations.py" \
    --image_dir "$IMG_DIR" \
    --out_json "$ANN"
fi

for ((s=0; s<NUM_SHARDS; s++)); do
  out_csv="$OUT_DIR/results_qwen2_5_vl_shard_${s}_of_${NUM_SHARDS}.csv"
  log_file="$LOG_DIR/shard_${s}.log"

  echo "[RUN] shard $s/$NUM_SHARDS -> $out_csv"

  cmd=(python "$BENCH"
    --image_dir "$IMG_DIR"
    --ann_json "$ANN"
    --model_path "/data/jianzhiy/models/Qwen2.5-VL-7B-Instruct"
    --out_csv "$out_csv"
    --num_shards "$NUM_SHARDS"
    --shard "$s"
    --resume
    --max_new_tokens "$MAX_NEW_TOKENS"
    --device_map "$DEVICE_MAP"
  )

  if [[ "$LOCAL_ONLY" == "1" ]]; then
    cmd+=(--local_only)
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"
done

echo "[DONE] all shards finished under $OUT_DIR"

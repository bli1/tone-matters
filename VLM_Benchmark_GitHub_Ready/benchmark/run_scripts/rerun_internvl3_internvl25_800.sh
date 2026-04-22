#!/usr/bin/env bash
set -u
set -o pipefail

BASE='/data/jianzhiy'
BENCH_DIR="$BASE/Final/Final Benchmark"
ANN_JSON="$BASE/Final/final_annotations_800.json"
IMG_DIR="$BASE/Final/Final Dataset"
FINAL_RESULTS_DIR="$BASE/Final/results/Final_results"
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$BASE/Final/results/rerun_internvl3_internvl25_800_${RUN_TS}"
mkdir -p "$RUN_ROOT" "$FINAL_RESULTS_DIR"

echo "$RUN_ROOT" > "$BASE/Final/results/active_rerun_iv3_iv25.txt"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_ROOT/master.log"
}

count_rows_no_header() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo 0
    return
  fi
  python - <<'PY' "$f"
import sys
p=sys.argv[1]
with open(p,'r',encoding='utf-8',errors='ignore') as f:
    n=sum(1 for _ in f)
print(max(n-1,0))
PY
}

merge_csvs() {
  local out_csv="$1"; shift
  python - <<'PY' "$out_csv" "$@"
import csv, os, sys
out=sys.argv[1]
ins=[p for p in sys.argv[2:] if os.path.exists(p) and os.path.getsize(p)>0]
if not ins:
    raise SystemExit(1)
header=None
rows=[]
for p in ins:
    with open(p,newline='',encoding='utf-8',errors='ignore') as f:
        r=csv.reader(f)
        h=next(r,None)
        if h is None:
            continue
        if header is None:
            header=h
        for row in r:
            if row:
                rows.append(row)
if header is None:
    raise SystemExit(2)
idx_img = header.index('Image_Num') if 'Image_Num' in header else None
idx_tone = header.index('Tone') if 'Tone' in header else None
if idx_img is not None and idx_tone is not None:
    def key(r):
        try: return (int(r[idx_img]), int(r[idx_tone]))
        except: return (10**9,10**9)
    rows.sort(key=key)
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(len(rows))
PY
}

run_internvl3() {
  local name='internvl3'
  local model_path='/data/jianzhiy/models/InternVL3-8B-hf'
  local model_dir="$RUN_ROOT/${name}_800_8shard"
  mkdir -p "$model_dir"
  log "==== START $name ===="

  local pids=()
  local shard gpu out lg

  # first pass: shards 1..7 in parallel on GPU 1..7
  for shard in 1 2 3 4 5 6 7; do
    gpu="$shard"
    out="$model_dir/results_shard_${shard}.csv"
    lg="$model_dir/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python "$BENCH_DIR/benchmark_internvl3_hf_yesno.py" \
      --image_dir "$IMG_DIR" \
      --ann_json "$ANN_JSON" \
      --model_path "$model_path" \
      --out_csv "$out" \
      --local_only \
      --device_map cuda \
      --dtype bf16 \
      --max_new_tokens 128 \
      --shard "$shard" \
      --num_shards 8 > "$lg" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done

  # run shard 0 on GPU1 to avoid busy GPU0
  out="$model_dir/results_shard_0.csv"
  lg="$model_dir/shard_0.log"
  CUDA_VISIBLE_DEVICES=1 python "$BENCH_DIR/benchmark_internvl3_hf_yesno.py" \
    --image_dir "$IMG_DIR" \
    --ann_json "$ANN_JSON" \
    --model_path "$model_path" \
    --out_csv "$out" \
    --local_only \
    --device_map cuda \
    --dtype bf16 \
    --max_new_tokens 128 \
    --shard 0 \
    --num_shards 8 > "$lg" 2>&1 || true

  # retry incomplete shards sequentially on GPU1
  for shard in 0 1 2 3 4 5 6 7; do
    out="$model_dir/results_shard_${shard}.csv"
    local got
    got=$(count_rows_no_header "$out")
    if (( got < 500 )); then
      log "$name shard=$shard incomplete ($got/500), retry on GPU1"
      lg="$model_dir/shard_${shard}_retry.log"
      CUDA_VISIBLE_DEVICES=1 python "$BENCH_DIR/benchmark_internvl3_hf_yesno.py" \
        --image_dir "$IMG_DIR" \
        --ann_json "$ANN_JSON" \
        --model_path "$model_path" \
        --out_csv "$out" \
        --local_only \
        --device_map cuda \
        --dtype bf16 \
        --max_new_tokens 128 \
        --shard "$shard" \
        --num_shards 8 > "$lg" 2>&1 || true
      got=$(count_rows_no_header "$out")
      log "$name shard=$shard after retry rows=$got/500"
    fi
  done

  local merged="$model_dir/results_merged_800.csv"
  merge_csvs "$merged" "$model_dir"/results_shard_*.csv >/dev/null 2>&1 || true
  local total
  total=$(count_rows_no_header "$merged")
  log "$name merged rows=$total"
  cp -f "$merged" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_TS}__results_merged_800.csv" 2>/dev/null || true
  log "==== DONE $name ===="
}

run_internvl25() {
  local name='internvl25'
  local model_path='/data/jianzhiy/hf_cache/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463'
  local model_dir="$RUN_ROOT/${name}_800_8shard"
  mkdir -p "$model_dir"
  log "==== START $name ===="

  local pids=()
  local shard gpu out lg

  # first pass: shards 1..7 in parallel on GPU 1..7
  for shard in 1 2 3 4 5 6 7; do
    gpu="$shard"
    out="$model_dir/results_shard_${shard}.csv"
    lg="$model_dir/shard_${shard}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python "$BENCH_DIR/benchmark_internvl25_freeform_with_judge.py" \
      --image_dir "$IMG_DIR" \
      --ann_json "$ANN_JSON" \
      --model_path "$model_path" \
      --out_csv "$out" \
      --local_only \
      --trust_remote_code \
      --device cuda \
      --dtype bf16 \
      --max_new_tokens 96 \
      --max_words 25 \
      --max_num_patches 6 \
      --shard "$shard" \
      --num_shards 8 > "$lg" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done

  # run shard 0 on GPU1 to avoid busy GPU0
  out="$model_dir/results_shard_0.csv"
  lg="$model_dir/shard_0.log"
  CUDA_VISIBLE_DEVICES=1 python "$BENCH_DIR/benchmark_internvl25_freeform_with_judge.py" \
    --image_dir "$IMG_DIR" \
    --ann_json "$ANN_JSON" \
    --model_path "$model_path" \
    --out_csv "$out" \
    --local_only \
    --trust_remote_code \
    --device cuda \
    --dtype bf16 \
    --max_new_tokens 96 \
    --max_words 25 \
    --max_num_patches 6 \
    --shard 0 \
    --num_shards 8 > "$lg" 2>&1 || true

  # retry incomplete shards sequentially on GPU1
  for shard in 0 1 2 3 4 5 6 7; do
    out="$model_dir/results_shard_${shard}.csv"
    local got
    got=$(count_rows_no_header "$out")
    if (( got < 500 )); then
      log "$name shard=$shard incomplete ($got/500), retry on GPU1"
      lg="$model_dir/shard_${shard}_retry.log"
      CUDA_VISIBLE_DEVICES=1 python "$BENCH_DIR/benchmark_internvl25_freeform_with_judge.py" \
        --image_dir "$IMG_DIR" \
        --ann_json "$ANN_JSON" \
        --model_path "$model_path" \
        --out_csv "$out" \
        --local_only \
        --trust_remote_code \
        --device cuda \
        --dtype bf16 \
        --max_new_tokens 96 \
        --max_words 25 \
        --max_num_patches 6 \
        --shard "$shard" \
        --num_shards 8 > "$lg" 2>&1 || true
      got=$(count_rows_no_header "$out")
      log "$name shard=$shard after retry rows=$got/500"
    fi
  done

  local merged="$model_dir/results_merged_800.csv"
  merge_csvs "$merged" "$model_dir"/results_shard_*.csv >/dev/null 2>&1 || true
  local total
  total=$(count_rows_no_header "$merged")
  log "$name merged rows=$total"
  cp -f "$merged" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_TS}__results_merged_800.csv" 2>/dev/null || true
  log "==== DONE $name ===="
}

log "Run root: $RUN_ROOT"
run_internvl3
run_internvl25
log "ALL DONE"

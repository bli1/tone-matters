#!/usr/bin/env bash
set -u

BASE='/data/jianzhiy'
BENCH_DIR="$BASE/Final/Final Benchmark"
ANN_800="$BASE/Final/final_annotations_800.json"
IMG_DIR="$BASE/Final/Final Dataset"
RESULTS_BASE="$BASE/Final/results"
FINAL_RESULTS_DIR="$RESULTS_BASE/Final_results"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$RESULTS_BASE/runs_800_${TS}"
SHARD_DIR="$RUN_ROOT/ann_shards"
CHUNK_DIR="$RUN_ROOT/ann_chunks"
MASTER_LOG="$RUN_ROOT/master.log"

mkdir -p "$RUN_ROOT" "$SHARD_DIR" "$CHUNK_DIR" "$FINAL_RESULTS_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

log "Run root: $RUN_ROOT"

python - <<'PY' "$ANN_800" "$SHARD_DIR" "$CHUNK_DIR"
import json, os, sys
ann_path, shard_dir, chunk_dir = sys.argv[1:4]
os.makedirs(shard_dir, exist_ok=True)
os.makedirs(chunk_dir, exist_ok=True)
with open(ann_path, 'r', encoding='utf-8') as f:
    ann = json.load(f)
items = ann['items']
# 8-way round-robin shards for single-GPU parallel
for i in range(8):
    d = dict(ann)
    d['items'] = [it for idx, it in enumerate(items) if idx % 8 == i]
    with open(os.path.join(shard_dir, f'ann_shard_{i}.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
# 8 contiguous chunks (100 each for 800) for Gemma model-parallel resume-friendly runs
chunk_size = (len(items) + 7) // 8
for i in range(8):
    d = dict(ann)
    d['items'] = items[i*chunk_size:(i+1)*chunk_size]
    with open(os.path.join(chunk_dir, f'ann_chunk_{i}.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
print('shards/chunks generated')
PY

merge_csvs() {
  local out_csv="$1"; shift
  python - <<'PY' "$out_csv" "$@"
import csv, os, sys
out = sys.argv[1]
ins = [p for p in sys.argv[2:] if os.path.exists(p) and os.path.getsize(p) > 0]
if not ins:
    print('no input csv for', out)
    raise SystemExit(1)
rows = []
header = None
for p in ins:
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        h = next(r, None)
        if h is None:
            continue
        if header is None:
            header = h
        for row in r:
            if row:
                rows.append(row)
if header is None:
    raise SystemExit(2)
idx_img = header.index('Image_Num') if 'Image_Num' in header else None
idx_tone = header.index('Tone') if 'Tone' in header else None
if idx_img is not None and idx_tone is not None:
    def key(row):
        try:
            return (int(row[idx_img]), int(row[idx_tone]))
        except Exception:
            return (10**9, 10**9)
    rows.sort(key=key)
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print('merged', len(rows), 'rows to', out)
PY
}

run_sharded_model() {
  local name="$1"; shift
  local script="$1"; shift
  local model_path="$1"; shift
  local model_dir="$RUN_ROOT/${name}_800_8gpu_${TS}"
  mkdir -p "$model_dir"
  log "==== START $name ===="

  local pids=()
  local shard
  for shard in 0 1 2 3 4 5 6 7; do
    local ann="$SHARD_DIR/ann_shard_${shard}.json"
    local out="$model_dir/results_shard_${shard}.csv"
    local lg="$model_dir/shard_${shard}.log"

    CUDA_VISIBLE_DEVICES="$shard" python "$script" \
      --image_dir "$IMG_DIR" \
      --ann_json "$ann" \
      --model_path "$model_path" \
      --out_csv "$out" \
      "$@" > "$lg" 2>&1 &
    pids+=("$!")
    echo "$!" > "$model_dir/shard_${shard}.pid"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done

  local merged="$model_dir/results_merged_800.csv"
  local shards=("$model_dir"/results_shard_*.csv)
  merge_csvs "$merged" "${shards[@]}" || failed=1

  cp -f "$merged" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${TS}__results_merged_800.csv" 2>/dev/null || true

  if [[ "$failed" -eq 0 ]]; then
    log "==== DONE $name (ok) ===="
  else
    log "==== DONE $name (with failures, check logs) ===="
  fi
}

run_gemma_chunked() {
  local name='gemma_vl'
  local script="$BENCH_DIR/benchmark_gemma.py"
  local model_path='/data/jianzhiy/hf_cache/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80'
  local model_dir="$RUN_ROOT/${name}_800_8gpu_${TS}"
  mkdir -p "$model_dir"
  log "==== START $name ===="

  local failed=0
  local i
  for i in 0 1 2 3 4 5 6 7; do
    local ann="$CHUNK_DIR/ann_chunk_${i}.json"
    local out="$model_dir/results_chunk_${i}.csv"
    local lg="$model_dir/chunk_${i}.log"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$script" \
      --image_dir "$IMG_DIR" \
      --ann_json "$ann" \
      --model_path "$model_path" \
      --out_csv "$out" \
      --local_only \
      --device_map auto \
      --dtype bf16 \
      --max_new_tokens 128 > "$lg" 2>&1 || failed=1

    log "gemma chunk $i finished"
  done

  local merged="$model_dir/results_merged_800.csv"
  local parts=("$model_dir"/results_chunk_*.csv)
  merge_csvs "$merged" "${parts[@]}" || failed=1
  cp -f "$merged" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${TS}__results_merged_800.csv" 2>/dev/null || true

  if [[ "$failed" -eq 0 ]]; then
    log "==== DONE $name (ok) ===="
  else
    log "==== DONE $name (with failures, check logs) ===="
  fi
}

# -------------------------------
# Model queue (all 800)
# -------------------------------
run_sharded_model 'qwen2_5_vl' \
  "$BENCH_DIR/benchmark_qwen2_5_vl.py" \
  '/data/jianzhiy/models/Qwen2.5-VL-7B-Instruct' \
  --local_only --device_map cuda --max_new_tokens 128

run_sharded_model 'qwen3_vl' \
  "$BENCH_DIR/benchmark_qwen3_vl.py" \
  '/data/jianzhiy/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b' \
  --local_only --device_map cuda --max_new_tokens 128

run_sharded_model 'internvl3' \
  "$BENCH_DIR/benchmark_internvl3_hf_yesno.py" \
  '/data/jianzhiy/models/InternVL3-8B-hf' \
  --local_only --device_map cuda --dtype bf16 --max_new_tokens 128

run_sharded_model 'llama32v' \
  "$BENCH_DIR/benchmark_llama32v.py" \
  '/data/jianzhiy/models/llama32v-11b-vision-instruct' \
  --local_only --device_map cuda:0 --dtype bfloat16

run_sharded_model 'llava_onevision' \
  "$BENCH_DIR/benchmark_llava_onevision.py" \
  '/data/jianzhiy/hf_cache/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068' \
  --local_only --device_map cuda --dtype bf16 --max_new_tokens 128

run_sharded_model 'deepseek_vl' \
  "$BENCH_DIR/benchmark_deepseek_vl_v2.py" \
  '/data/jianzhiy/hf_cache/hub/models--deepseek-ai--deepseek-vl-7b-chat/snapshots/6f16f00805f45b5249f709ce21820122eeb43556' \
  --device cuda --dtype bf16 --max_new_tokens 128

run_sharded_model 'internvl25' \
  "$BENCH_DIR/benchmark_internvl25_freeform_with_judge.py" \
  '/data/jianzhiy/hf_cache/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463' \
  --local_only --trust_remote_code --device cuda --dtype bf16 --max_new_tokens 64 --max_words 20

run_gemma_chunked

log "ALL MODELS FINISHED"

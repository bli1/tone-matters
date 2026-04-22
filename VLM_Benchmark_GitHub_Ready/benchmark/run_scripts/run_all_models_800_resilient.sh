#!/usr/bin/env bash
set -u

BASE='/data/jianzhiy'
BENCH_DIR="$BASE/Final/Final Benchmark"
ANN_800="$BASE/Final/final_annotations_800.json"
IMG_DIR="$BASE/Final/Final Dataset"
RESULTS_BASE="$BASE/Final/results"
FINAL_RESULTS_DIR="$RESULTS_BASE/Final_results"
RUN_ID_FILE="$RESULTS_BASE/active_800_run_id.txt"

mkdir -p "$RESULTS_BASE" "$FINAL_RESULTS_DIR"

if [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID=$(cat "$RUN_ID_FILE")
else
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  echo "$RUN_ID" > "$RUN_ID_FILE"
fi

RUN_ROOT="$RESULTS_BASE/runs_800_${RUN_ID}"
SHARD_DIR="$RUN_ROOT/ann_shards"
CHUNK_DIR="$RUN_ROOT/ann_chunks"
MASTER_LOG="$RUN_ROOT/master.log"

mkdir -p "$RUN_ROOT" "$SHARD_DIR" "$CHUNK_DIR"
ln -sfn "$RUN_ROOT" "$RESULTS_BASE/current_800_run"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
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

expected_rows_from_ann() {
  local ann="$1"
  python - <<'PY' "$ann"
import json,sys
with open(sys.argv[1],'r',encoding='utf-8') as f:
    d=json.load(f)
print(len(d.get('items',[]))*5)
PY
}

prepare_ann_splits() {
  if [[ -f "$SHARD_DIR/ann_shard_0.json" && -f "$CHUNK_DIR/ann_chunk_0.json" ]]; then
    return
  fi
  python - <<'PY' "$ANN_800" "$SHARD_DIR" "$CHUNK_DIR"
import json, os, sys
ann_path, shard_dir, chunk_dir = sys.argv[1:4]
os.makedirs(shard_dir, exist_ok=True)
os.makedirs(chunk_dir, exist_ok=True)
with open(ann_path,'r',encoding='utf-8') as f:
    ann=json.load(f)
items=ann['items']
for i in range(8):
    d=dict(ann)
    d['items']=[it for idx,it in enumerate(items) if idx%8==i]
    with open(os.path.join(shard_dir,f'ann_shard_{i}.json'),'w',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,indent=2)
chunk_size=(len(items)+7)//8
for i in range(8):
    d=dict(ann)
    d['items']=items[i*chunk_size:(i+1)*chunk_size]
    with open(os.path.join(chunk_dir,f'ann_chunk_{i}.json'),'w',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,indent=2)
print('ok')
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

run_model_sharded() {
  local name="$1"; shift
  local script="$1"; shift
  local model_path="$1"; shift
  local model_dir="$RUN_ROOT/${name}_800_8gpu_${RUN_ID}"
  local final_csv="$model_dir/results_merged_800.csv"
  mkdir -p "$model_dir"

  if [[ -f "$final_csv" ]]; then
    log "$name already done, skip"
    cp -f "$final_csv" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_ID}__results_merged_800.csv" 2>/dev/null || true
    return
  fi

  log "==== START $name ===="
  local round=0
  while true; do
    round=$((round+1))
    local pending=()
    local shard
    for shard in 0 1 2 3 4 5 6 7; do
      local ann="$SHARD_DIR/ann_shard_${shard}.json"
      local csv="$model_dir/results_shard_${shard}.csv"
      local exp
      exp=$(expected_rows_from_ann "$ann")
      local got
      got=$(count_rows_no_header "$csv")
      if (( got < exp )); then
        pending+=("$shard")
      fi
    done

    if (( ${#pending[@]} == 0 )); then
      break
    fi

    log "$name round=$round pending_shards=${pending[*]}"

    local pids=()
    local pid_to_shard_file="$model_dir/pid_map_round_${round}.txt"
    : > "$pid_to_shard_file"

    for shard in "${pending[@]}"; do
      local ann="$SHARD_DIR/ann_shard_${shard}.json"
      local tmp="$model_dir/results_shard_${shard}.tmp.csv"
      local lg="$model_dir/shard_${shard}.log"
      rm -f "$tmp"

      CUDA_VISIBLE_DEVICES="$shard" python "$script" \
        --image_dir "$IMG_DIR" \
        --ann_json "$ann" \
        --model_path "$model_path" \
        --out_csv "$tmp" \
        "$@" > "$lg" 2>&1 &
      local pid=$!
      echo "$pid $shard" >> "$pid_to_shard_file"
      pids+=("$pid")
    done

    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done

    # validate and commit tmp -> csv if complete
    for shard in "${pending[@]}"; do
      local ann="$SHARD_DIR/ann_shard_${shard}.json"
      local exp
      exp=$(expected_rows_from_ann "$ann")
      local tmp="$model_dir/results_shard_${shard}.tmp.csv"
      local csv="$model_dir/results_shard_${shard}.csv"
      if [[ -f "$tmp" ]]; then
        local got
        got=$(count_rows_no_header "$tmp")
        if (( got >= exp )); then
          mv -f "$tmp" "$csv"
          log "$name shard=$shard committed rows=$got/$exp"
        else
          log "$name shard=$shard incomplete rows=$got/$exp (will retry)"
        fi
      else
        log "$name shard=$shard missing tmp output (will retry)"
      fi
    done

    sleep 5
  done

  local shards=("$model_dir"/results_shard_*.csv)
  if merge_csvs "$final_csv" "${shards[@]}" >/dev/null 2>&1; then
    cp -f "$final_csv" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_ID}__results_merged_800.csv" 2>/dev/null || true
    log "==== DONE $name ===="
  else
    log "==== FAIL MERGE $name ===="
  fi
}

run_gemma_chunked() {
  local name='gemma_vl'
  local script="$BENCH_DIR/benchmark_gemma.py"
  local model_path='/data/jianzhiy/hf_cache/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80'
  local model_dir="$RUN_ROOT/${name}_800_8gpu_${RUN_ID}"
  local final_csv="$model_dir/results_merged_800.csv"
  mkdir -p "$model_dir"

  if [[ -f "$final_csv" ]]; then
    log "$name already done, skip"
    cp -f "$final_csv" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_ID}__results_merged_800.csv" 2>/dev/null || true
    return
  fi

  log "==== START $name ===="

  local round=0
  while true; do
    round=$((round+1))
    local pending=()
    local i
    for i in 0 1 2 3 4 5 6 7; do
      local ann="$CHUNK_DIR/ann_chunk_${i}.json"
      local csv="$model_dir/results_chunk_${i}.csv"
      local exp
      exp=$(expected_rows_from_ann "$ann")
      local got
      got=$(count_rows_no_header "$csv")
      if (( got < exp )); then
        pending+=("$i")
      fi
    done

    if (( ${#pending[@]} == 0 )); then
      break
    fi

    log "$name round=$round pending_chunks=${pending[*]}"

    for i in "${pending[@]}"; do
      local ann="$CHUNK_DIR/ann_chunk_${i}.json"
      local exp
      exp=$(expected_rows_from_ann "$ann")
      local tmp="$model_dir/results_chunk_${i}.tmp.csv"
      local csv="$model_dir/results_chunk_${i}.csv"
      local lg="$model_dir/chunk_${i}.log"
      rm -f "$tmp"

      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python "$script" \
        --image_dir "$IMG_DIR" \
        --ann_json "$ann" \
        --model_path "$model_path" \
        --out_csv "$tmp" \
        --local_only \
        --device_map auto \
        --dtype bf16 \
        --max_new_tokens 128 > "$lg" 2>&1 || true

      if [[ -f "$tmp" ]]; then
        local got
        got=$(count_rows_no_header "$tmp")
        if (( got >= exp )); then
          mv -f "$tmp" "$csv"
          log "$name chunk=$i committed rows=$got/$exp"
        else
          log "$name chunk=$i incomplete rows=$got/$exp (will retry)"
        fi
      else
        log "$name chunk=$i missing tmp output (will retry)"
      fi
    done

    sleep 5
  done

  local parts=("$model_dir"/results_chunk_*.csv)
  if merge_csvs "$final_csv" "${parts[@]}" >/dev/null 2>&1; then
    cp -f "$final_csv" "$FINAL_RESULTS_DIR/${name}_800_8gpu_${RUN_ID}__results_merged_800.csv" 2>/dev/null || true
    log "==== DONE $name ===="
  else
    log "==== FAIL MERGE $name ===="
  fi
}

prepare_ann_splits
log "Run root: $RUN_ROOT"

run_model_sharded 'qwen2_5_vl' \
  "$BENCH_DIR/benchmark_qwen2_5_vl.py" \
  '/data/jianzhiy/models/Qwen2.5-VL-7B-Instruct' \
  --local_only --device_map cuda --max_new_tokens 128

run_model_sharded 'qwen3_vl' \
  "$BENCH_DIR/benchmark_qwen3_vl.py" \
  '/data/jianzhiy/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b' \
  --local_only --device_map cuda --max_new_tokens 128

run_model_sharded 'internvl3' \
  "$BENCH_DIR/benchmark_internvl3_hf_yesno.py" \
  '/data/jianzhiy/models/InternVL3-8B-hf' \
  --local_only --device_map cuda --dtype bf16 --max_new_tokens 128

run_model_sharded 'llama32v' \
  "$BENCH_DIR/benchmark_llama32v.py" \
  '/data/jianzhiy/models/llama32v-11b-vision-instruct' \
  --local_only --device_map cuda:0 --dtype bfloat16

run_model_sharded 'llava_onevision' \
  "$BENCH_DIR/benchmark_llava_onevision.py" \
  '/data/jianzhiy/hf_cache/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068' \
  --local_only --device_map cuda --dtype bf16 --max_new_tokens 128

run_model_sharded 'deepseek_vl' \
  "$BENCH_DIR/benchmark_deepseek_vl_v2.py" \
  '/data/jianzhiy/hf_cache/hub/models--deepseek-ai--deepseek-vl-7b-chat/snapshots/6f16f00805f45b5249f709ce21820122eeb43556' \
  --device cuda --dtype bf16 --max_new_tokens 128

run_model_sharded 'internvl25' \
  "$BENCH_DIR/benchmark_internvl25_freeform_with_judge.py" \
  '/data/jianzhiy/hf_cache/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463' \
  --local_only --trust_remote_code --device cuda --dtype bf16 --max_new_tokens 64 --max_words 20

run_gemma_chunked

log 'ALL MODELS FINISHED'

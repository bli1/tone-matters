#!/usr/bin/env bash
set -u
set -o pipefail

BASE='/data/jianzhiy'
BENCH_DIR="$BASE/Final/Final Benchmark"
ANN_JSON="$BASE/Final/final_annotations_800.json"
IMG_DIR="$BASE/Final/Final Dataset"
FINAL_RESULTS_DIR="$BASE/Final/results/Final_results"
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$BASE/Final/results/rerun_internvl3_internvl25_800_stable_${RUN_TS}"
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

pick_gpu() {
  python - <<'PY'
import subprocess
out = subprocess.check_output([
    'nvidia-smi',
    '--query-gpu=index,memory.used,memory.total',
    '--format=csv,noheader,nounits'
], text=True)
best = None
for line in out.strip().splitlines():
    i_s, used_s, total_s = [x.strip() for x in line.split(',')]
    idx = int(i_s); used = int(used_s); total = int(total_s)
    if idx == 0:
        continue
    if used >= total - 1500:
        continue
    if best is None or used < best[1]:
        best = (idx, used)
print(best[0] if best else 1)
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

  local shard out lg gpu got
  for shard in 0 1 2 3 4 5 6 7; do
    out="$model_dir/results_shard_${shard}.csv"
    lg="$model_dir/shard_${shard}.log"
    gpu=$(pick_gpu)
    log "$name shard=$shard run on GPU$gpu"
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
      --num_shards 8 > "$lg" 2>&1 || true

    got=$(count_rows_no_header "$out")
    if (( got < 500 )); then
      gpu=$(pick_gpu)
      log "$name shard=$shard incomplete ($got/500), retry on GPU$gpu"
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
        --num_shards 8 > "$model_dir/shard_${shard}_retry.log" 2>&1 || true
      got=$(count_rows_no_header "$out")
      log "$name shard=$shard rows after retry=$got/500"
    else
      log "$name shard=$shard rows=$got/500"
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

  local shard out lg gpu got
  for shard in 0 1 2 3 4 5 6 7; do
    out="$model_dir/results_shard_${shard}.csv"
    lg="$model_dir/shard_${shard}.log"
    gpu=$(pick_gpu)
    log "$name shard=$shard run on GPU$gpu"
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
      --num_shards 8 > "$lg" 2>&1 || true

    got=$(count_rows_no_header "$out")
    if (( got < 500 )); then
      gpu=$(pick_gpu)
      log "$name shard=$shard incomplete ($got/500), retry on GPU$gpu"
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
        --max_num_patches 4 \
        --shard "$shard" \
        --num_shards 8 > "$model_dir/shard_${shard}_retry.log" 2>&1 || true
      got=$(count_rows_no_header "$out")
      log "$name shard=$shard rows after retry=$got/500"
    else
      log "$name shard=$shard rows=$got/500"
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

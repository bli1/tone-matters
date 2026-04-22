#!/usr/bin/env bash
set -u

ROOT='/data/jianzhiy/Final/results/Final_results'
SCORER='/data/jianzhiy/Final/Final Benchmark/score_hybrid_600_200.py'
MASTER_LOG="$ROOT/pure_gpt_batch_800_master.log"
STATUS_CSV="$ROOT/pure_gpt_batch_800_status.csv"

echo "model,input,output,status,start_time,end_time" > "$STATUS_CSV"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

log "Batch start: pure-gpt scoring for 800 merged files"

shopt -s nullglob
files=("$ROOT"/*__results_merged_800.csv)
IFS=$'\n' files=($(printf '%s\n' "${files[@]}" | sort))
unset IFS

for in_csv in "${files[@]}"; do
  base=$(basename "$in_csv" .csv)
  model="${base%__results_merged_800}"
  out_csv="$ROOT/${model}__results_scored_pure_gpt.csv"
  sum_csv="$ROOT/${model}__tone_summary_pure_gpt.csv"
  split_csv="$ROOT/${model}__split_summary_pure_gpt.csv"

  start_time=$(ts)

  if [[ -f "$out_csv" && -f "$sum_csv" && -f "$split_csv" ]]; then
    log "[SKIP] $model already done"
    echo "$model,$in_csv,$out_csv,skipped,$start_time,$(ts)" >> "$STATUS_CSV"
    continue
  fi

  model_log="$ROOT/${model}__pure_gpt_run.log"
  log "[START] $model"

  python -u "$SCORER" \
    --input "$in_csv" \
    --output "$out_csv" \
    --summary "$sum_csv" \
    --summary-split "$split_csv" \
    --judge-model 'gpt-4o-mini' \
    --retries 3 \
    --pure-gpt > "$model_log" 2>&1
  rc=$?

  if [[ $rc -eq 0 ]]; then
    log "[DONE] $model"
    echo "$model,$in_csv,$out_csv,done,$start_time,$(ts)" >> "$STATUS_CSV"
  else
    log "[FAIL] $model rc=$rc (see $model_log)"
    echo "$model,$in_csv,$out_csv,failed_rc_$rc,$start_time,$(ts)" >> "$STATUS_CSV"
  fi
done

log "Batch finished"

#!/usr/bin/env bash
# run_gemma4_e4b.sh
# Full pipeline for Gemma4-E4B on 800-image benchmark:
#   Step 1 — shard annotations
#   Step 2 — run benchmark (8 shards x 1 GPU in parallel)
#   Step 3 — validate & merge
#   Step 4 — ASR eval (categories 01-06)
#   Step 5 — rule-based scoring (score_split_hallucination)
#   Step 6 — GPT severity scoring (score_hybrid_600_200, needs OPENAI_API_KEY)
set -uo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
PYTHON="/data/jianzhiy/envs/gemma4/bin/python3"
BENCH_DIR="/data/jianzhiy/Final/Final Benchmark"
ANN_800="/data/jianzhiy/Final/final_annotations_800.json"
IMG_DIR="/data/jianzhiy/Final/Final Dataset"
MODEL_PATH="/data/jianzhiy/data/gemma4_e4b/model"
RESULTS_BASE="/data/jianzhiy/Final/results"
FINAL_RESULTS_DIR="$RESULTS_BASE/Final_results"

MODEL_NAME="gemma4_e4b"
RUN_ID=$(date +%Y%m%d_%H%M%S)
# Set USE_4BIT=1 if each GPU has <12GB free (4B model needs ~14GB in bf16)
USE_4BIT="${USE_4BIT:-0}"
RUN_DIR="$RESULTS_BASE/${MODEL_NAME}_800_8gpu_${RUN_ID}"
SHARD_DIR="$RUN_DIR/shards"
MASTER_LOG="$RUN_DIR/master.log"

# ─── Output files ─────────────────────────────────────────────────────────────
FINAL_CSV="$RUN_DIR/results_merged_800.csv"
ASR_CSV="$RUN_DIR/results_merged_800__asr_eval_0106.csv"
ASR_SUMMARY="$RUN_DIR/results_merged_800__asr_summary_0106.csv"
SCORED_CSV="$RUN_DIR/results_scored_pure_gpt.csv"
TONE_SUMMARY="$RUN_DIR/tone_summary_pure_gpt.csv"
SPLIT_SUMMARY="$RUN_DIR/split_summary_pure_gpt.csv"

mkdir -p "$RUN_DIR" "$SHARD_DIR" "$FINAL_RESULTS_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }

# ─── Step 1: split annotation into 8 shards ──────────────────────────────────
log "=== Step 1: Preparing annotation shards ==="
$PYTHON - <<'PY' "$ANN_800" "$SHARD_DIR"
import json, os, sys
ann_path, shard_dir = sys.argv[1], sys.argv[2]
os.makedirs(shard_dir, exist_ok=True)
with open(ann_path, 'r', encoding='utf-8') as f:
    ann = json.load(f)
items = ann['items']
for i in range(8):
    d = dict(ann)
    d['items'] = [it for idx, it in enumerate(items) if idx % 8 == i]
    with open(os.path.join(shard_dir, f'ann_shard_{i}.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
print(f"Created 8 shards from {len(items)} items ({len(items)*5} expected rows total)")
PY

# ─── Step 2: run 8 shards in parallel (1 GPU each) ───────────────────────────
log "=== Step 2: Launching 8 shards in parallel ==="
PIDS=()
for SHARD in 0 1 2 3 4 5 6 7; do
    ANN_SHARD="$SHARD_DIR/ann_shard_${SHARD}.json"
    OUT_TMP="$RUN_DIR/results_shard_${SHARD}.tmp.csv"
    OUT_CSV="$RUN_DIR/results_shard_${SHARD}.csv"
    SHARD_LOG="$RUN_DIR/shard_${SHARD}.log"
    rm -f "$OUT_TMP"

    EXTRA_ARGS=""
    [[ "$USE_4BIT" == "1" ]] && EXTRA_ARGS="--use_4bit"

    CUDA_VISIBLE_DEVICES="$SHARD" $PYTHON \
        "$BENCH_DIR/benchmark_gemma4_e4b.py" \
        --image_dir  "$IMG_DIR" \
        --ann_json   "$ANN_SHARD" \
        --model_path "$MODEL_PATH" \
        --out_csv    "$OUT_TMP" \
        --dtype      bf16 \
        --device_map cuda \
        --local_only \
        --max_new_tokens 128 \
        $EXTRA_ARGS \
        > "$SHARD_LOG" 2>&1 &
    PIDS+=($!)
    log "  Shard $SHARD started (PID ${PIDS[-1]}, GPU $SHARD)"
done

for PID in "${PIDS[@]}"; do
    wait "$PID" && log "  PID $PID finished OK" \
                || log "  PID $PID exited with error"
done

# ─── Step 3: validate, commit, merge ─────────────────────────────────────────
log "=== Step 3: Validating and merging shards ==="
ALL_OK=true
for SHARD in 0 1 2 3 4 5 6 7; do
    ANN_SHARD="$SHARD_DIR/ann_shard_${SHARD}.json"
    OUT_TMP="$RUN_DIR/results_shard_${SHARD}.tmp.csv"
    OUT_CSV="$RUN_DIR/results_shard_${SHARD}.csv"
    EXP=$($PYTHON -c "
import json
with open('$ANN_SHARD') as f: d = json.load(f)
print(len(d['items']) * 5)
")
    GOT=0
    if [[ -f "$OUT_TMP" ]]; then
        GOT=$($PYTHON -c "
with open('$OUT_TMP', encoding='utf-8', errors='ignore') as f:
    print(max(sum(1 for _ in f) - 1, 0))
")
    fi
    if (( GOT >= EXP )); then
        mv -f "$OUT_TMP" "$OUT_CSV"
        log "  Shard $SHARD OK ($GOT/$EXP rows)"
    else
        log "  Shard $SHARD INCOMPLETE ($GOT/$EXP) — see shard_${SHARD}.log"
        ALL_OK=false
    fi
done

if [[ "$ALL_OK" != true ]]; then
    log "WARNING: some shards incomplete. Merging available data anyway."
fi

$PYTHON - <<'PY' "$FINAL_CSV" "$RUN_DIR"
import csv, os, sys
out_path, run_dir = sys.argv[1], sys.argv[2]
shards = sorted([os.path.join(run_dir, f)
                 for f in os.listdir(run_dir)
                 if f.startswith('results_shard_') and f.endswith('.csv')])
header, rows = None, []
for p in shards:
    with open(p, newline='', encoding='utf-8', errors='ignore') as f:
        r = csv.reader(f)
        h = next(r, None)
        if h is None: continue
        if header is None: header = h
        for row in r:
            if row: rows.append(row)

idx_img  = header.index('Image_Num') if 'Image_Num' in header else None
idx_tone = header.index('Tone')      if 'Tone'      in header else None
if idx_img is not None and idx_tone is not None:
    rows.sort(key=lambda r: (int(r[idx_img]), int(r[idx_tone])))

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"Merged {len(rows)} rows -> {out_path}")
PY

cp -f "$FINAL_CSV" \
    "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__results_merged_800.csv"
log "  Merged CSV: $FINAL_CSV"

# ─── Step 4: ASR eval (categories 01-06) ─────────────────────────────────────
log "=== Step 4: ASR eval (01-06) ==="
$PYTHON "$BENCH_DIR/eval_asr_policy_0106.py" \
    --in_csv     "$FINAL_CSV" \
    --out_csv    "$ASR_CSV" \
    --summary_csv "$ASR_SUMMARY" \
    2>&1 | tee -a "$MASTER_LOG"

cp -f "$ASR_CSV"     "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__results_merged_800__asr_eval_0106.csv"     2>/dev/null || true
cp -f "$ASR_SUMMARY" "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__results_merged_800__asr_summary_0106.csv" 2>/dev/null || true

# ─── Step 5: rule-based scoring (score_split_hallucination) ──────────────────
log "=== Step 5: Rule-based scoring ==="
RULE_SCORED="$RUN_DIR/results_scored_rule.csv"
RULE_SUMMARY="$RUN_DIR/tone_summary_rule.csv"

$PYTHON "$BENCH_DIR/score_split_hallucination.py" \
    --in_csv     "$FINAL_CSV" \
    --out_csv    "$RULE_SCORED" \
    --summary_csv "$RULE_SUMMARY" \
    --mode       balanced \
    2>&1 | tee -a "$MASTER_LOG"

# ─── Step 6: GPT severity scoring (score_hybrid_600_200) ─────────────────────
log "=== Step 6: GPT severity scoring (score_hybrid_600_200) ==="
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log "  WARNING: OPENAI_API_KEY not set — skipping GPT scoring."
    log "  To run manually:"
    log "    export OPENAI_API_KEY=<your_key>"
    log "    $PYTHON \"$BENCH_DIR/score_hybrid_600_200.py\" \\"
    log "      --input  \"$FINAL_CSV\" \\"
    log "      --output \"$SCORED_CSV\" \\"
    log "      --summary \"$TONE_SUMMARY\" \\"
    log "      --summary-split \"$SPLIT_SUMMARY\" \\"
    log "      --judge-model gpt-4o-mini --retries 3 --pure-gpt"
else
    $PYTHON "$BENCH_DIR/score_hybrid_600_200.py" \
        --input         "$FINAL_CSV" \
        --output        "$SCORED_CSV" \
        --summary       "$TONE_SUMMARY" \
        --summary-split "$SPLIT_SUMMARY" \
        --judge-model   gpt-4o-mini \
        --retries       3 \
        --pure-gpt \
        2>&1 | tee -a "$MASTER_LOG"

    cp -f "$SCORED_CSV"    "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__results_scored_pure_gpt.csv"  2>/dev/null || true
    cp -f "$TONE_SUMMARY"  "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__tone_summary_pure_gpt.csv"   2>/dev/null || true
    cp -f "$SPLIT_SUMMARY" "$FINAL_RESULTS_DIR/${MODEL_NAME}_800_8gpu_${RUN_ID}__split_summary_pure_gpt.csv"  2>/dev/null || true
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
log "=============================="
log "ALL STEPS DONE"
log "Run dir:        $RUN_DIR"
log "Merged CSV:     $FINAL_CSV"
log "ASR eval:       $ASR_CSV"
log "Rule scored:    $RULE_SCORED"
log "GPT scored:     $SCORED_CSV"
log "Tone summary:   $TONE_SUMMARY"
log "Split summary:  $SPLIT_SUMMARY"
log "=============================="

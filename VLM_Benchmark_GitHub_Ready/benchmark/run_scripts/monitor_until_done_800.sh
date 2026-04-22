#!/usr/bin/env bash
set -euo pipefail

ROOT_LINK='/data/jianzhiy/Final/results/current_800_run'
LOG='/data/jianzhiy/Final/results/monitor_800_until_done.log'
LATEST='/data/jianzhiy/Final/results/monitor_800_latest.txt'
INTERVAL="${1:-60}"

mkdir -p /data/jianzhiy/Final/results

while true; do
  ts="$(date '+%F %T')"

  if [[ ! -e "$ROOT_LINK" ]]; then
    {
      echo "[$ts] current_800_run not found"
      echo
    } >> "$LOG"
    sleep "$INTERVAL"
    continue
  fi

  run_root="$(readlink -f "$ROOT_LINK" || echo "$ROOT_LINK")"

  report="$(python - <<'PY' "$run_root"
import csv, glob, os, re, sys
root = sys.argv[1]
models = ['qwen2_5_vl','qwen3_vl','internvl3','llama32v','llava_onevision','deepseek_vl','internvl25','gemma_vl']
TARGET=4000
master=os.path.join(root,'master.log')
active='-'
if os.path.exists(master):
    with open(master,encoding='utf-8',errors='ignore') as f:
        for line in f:
            m=re.search(r'START\s+([a-zA-Z0-9_]+)',line)
            if m:
                active=m.group(1)

def uniq_pairs(files):
    s=set(); rows=0
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp,newline='',encoding='utf-8',errors='ignore') as fh:
            r=csv.DictReader(fh)
            for row in r:
                rows += 1
                img=row.get('Image_File') or row.get('image_file')
                tone=row.get('Tone') or row.get('tone')
                if img and tone:
                    s.add((img.strip(), str(tone).strip()))
    return rows, len(s)

all_done=True
lines=[]
lines.append(f'RUN_ROOT {root}')
lines.append(f'ACTIVE {active}')
for m in models:
    dlist=sorted([d for d in glob.glob(f'{root}/{m}_800_8gpu_*') if os.path.isdir(d)])
    if not dlist:
        all_done=False
        lines.append(f'{m}\t0/{TARGET}\tinflight=0\tstatus=pending')
        continue
    d=dlist[-1]
    merged=glob.glob(f'{d}/results_merged_800.csv')
    committed_files = merged if merged else glob.glob(f'{d}/results_shard_*.csv')+glob.glob(f'{d}/results_chunk_*.csv')
    inflight_files = glob.glob(f'{d}/results_shard_*.tmp.csv')+glob.glob(f'{d}/results_chunk_*.tmp.csv')
    _, cu = uniq_pairs(committed_files)
    _, iu = uniq_pairs(inflight_files)
    status='done' if merged else ('running' if m==active else 'waiting')
    if cu < TARGET:
        all_done=False
    lines.append(f'{m}\t{cu}/{TARGET}\tinflight={iu}\tstatus={status}')

lines.append(f'ALL_DONE {1 if all_done else 0}')
print('\n'.join(lines))
PY
)"

  {
    echo "[$ts]"
    echo "$report"
    echo
  } >> "$LOG"

  {
    echo "[$ts]"
    echo "$report"
  } > "$LATEST"

  if grep -q '^ALL_DONE 1$' <<< "$report"; then
    echo "[$ts] monitor exit: all models done" >> "$LOG"
    exit 0
  fi

  sleep "$INTERVAL"
done

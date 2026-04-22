#!/usr/bin/env bash
set -u
BASE='/data/jianzhiy/Final/results'
ROOT_LINK="$BASE/current_800_run"
OUT="$BASE/progress_800.log"

while true; do
  ts=$(date '+%F %T')
  {
    echo "[$ts]"
    python - <<'PY'
import os,glob,re
root='/data/jianzhiy/Final/results/current_800_run'
order=['qwen2_5_vl','qwen3_vl','internvl3','llama32v','llava_onevision','deepseek_vl','internvl25','gemma_vl']
master=f'{root}/master.log'
active='-'
if os.path.exists(master):
    for line in open(master,encoding='utf-8',errors='ignore'):
        m=re.search(r'START\s+([a-zA-Z0-9_]+)',line)
        if m: active=m.group(1)

def rows(path):
    if not os.path.exists(path): return 0
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        return max(sum(1 for _ in f)-1,0)
print('active=',active)
for m in order:
    dlist=sorted([d for d in glob.glob(f'{root}/{m}_800_8gpu_*') if os.path.isdir(d)])
    if not dlist:
        print(f'{m}: committed=0/800 inflight=0 status=pending')
        continue
    d=dlist[-1]
    merged=f'{d}/results_merged_800.csv'
    if os.path.exists(merged):
        c=rows(merged)//5
        print(f'{m}: committed={c}/800 inflight=0 status=done')
        continue
    c=sum(rows(x) for x in glob.glob(f'{d}/results_shard_*.csv') if '.tmp.' not in x)
    c+=sum(rows(x) for x in glob.glob(f'{d}/results_chunk_*.csv') if '.tmp.' not in x)
    i=sum(rows(x) for x in glob.glob(f'{d}/results_shard_*.tmp.csv'))
    i+=sum(rows(x) for x in glob.glob(f'{d}/results_chunk_*.tmp.csv'))
    st='running' if m==active else 'waiting'
    print(f'{m}: committed={c//5}/800 inflight={i//5} status={st}')
PY
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/gpu_proc: /' || true
    echo
  } >> "$OUT"
  sleep 60
done

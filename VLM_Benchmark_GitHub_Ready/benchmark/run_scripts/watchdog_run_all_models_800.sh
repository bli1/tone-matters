#!/usr/bin/env bash
set -u
WORKER='/data/jianzhiy/Final/Final Benchmark/run_all_models_800_resilient.sh'
BASE='/data/jianzhiy/Final/results'
mkdir -p "$BASE"
WD_LOG="$BASE/watchdog_800.log"

echo "[$(date '+%F %T')] watchdog start" >> "$WD_LOG"
while true; do
  bash "$WORKER" >> "$WD_LOG" 2>&1
  rc=$?
  if grep -q 'ALL MODELS FINISHED' "$BASE/current_800_run/master.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] watchdog exit done" >> "$WD_LOG"
    exit 0
  fi
  echo "[$(date '+%F %T')] worker exited rc=$rc, restarting in 20s" >> "$WD_LOG"
  sleep 20
done

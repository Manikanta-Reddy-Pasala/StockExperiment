#!/bin/bash
# Multi-year backtest with false-alarm filters enabled.
# Filters: min_crossover_gap_pct=0.003, volume_confirm_mult=1.5, htf_filter
#
# Runs on prod inside trading_system_app container.

set -e

OUT_ROOT="${OUT_ROOT:-/app/exports/backtests/multiyear_filtered}"
mkdir -p "$OUT_ROOT"
PY="/usr/local/bin/python"

run_year() {
  local STRATEGY=$1
  local FROM=$2
  local TO=$3
  local LABEL=$4
  local EXTRA_FLAGS=$5
  local OUT_DIR="$OUT_ROOT/nifty50_${STRATEGY}_${LABEL}"
  mkdir -p "$OUT_DIR"
  echo "=== $STRATEGY $LABEL ($FROM .. $TO) ==="
  if [ "$STRATEGY" = "ema_9_21" ]; then
    EMA_ARGS="--ema-fast 9 --ema-slow 21 --warmup-days 60"
  else
    EMA_ARGS=""
  fi
  $PY /app/tools/backtests/run_ema_200_400_backtest.py \
    --universe nifty50 --from "$FROM" --to "$TO" --out "$OUT_DIR" \
    --min-crossover-gap-pct 0.003 \
    --volume-confirm-mult 1.5 \
    --htf-filter \
    $EMA_ARGS $EXTRA_FLAGS
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT_DIR" \
    --capital 1000000 1 2 3 5 8 10 > "$OUT_DIR/_capital_sim.txt" 2>&1
  $PY /app/tools/backtests/monthly_profile.py "$OUT_DIR" \
    --capital 1000000 --max-concurrent 2 > "$OUT_DIR/_monthly_profile.md" 2>&1 || true
}

for strategy in ema_200_400 ema_9_21; do
  for year in 2023_2024:2023-05-13:2024-05-12 \
              2024_2025:2024-05-13:2025-05-12 \
              2025_2026:2025-05-13:2026-05-12; do
    LABEL=$(echo "$year" | cut -d: -f1)
    FROM=$(echo "$year" | cut -d: -f2)
    TO=$(echo "$year" | cut -d: -f3)
    run_year "$strategy" "$FROM" "$TO" "$LABEL"
  done
done

echo "DONE. Results in $OUT_ROOT"

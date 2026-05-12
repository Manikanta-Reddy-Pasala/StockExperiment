#!/bin/bash
# Phase 13: extend optimization to N100, N150, selector top-10 universes.
# Best Phase 12 configs: EMA 9/21 v3_htfonly, ORB-60 v1_relaxed.

set -e
OUT_ROOT="${OUT_ROOT:-/app/exports/backtests/optimize_p13}"
mkdir -p "$OUT_ROOT"
PY="/usr/local/bin/python"
UNIVERSES_DIR="/app/exports/backtests/universes"

run_ema921() {
  local LABEL=$1
  local UFILE=$2
  local FROM=$3
  local TO=$4
  local FILTERS=$5
  local OUT="$OUT_ROOT/ema921_${LABEL}"
  mkdir -p "$OUT"
  echo "=== EMA9/21 $LABEL [$FROM..$TO universe=$UFILE filters=$FILTERS] ==="
  $PY /app/tools/backtests/run_ema_200_400_backtest.py \
    --universe-file "$UFILE" --from "$FROM" --to "$TO" --out "$OUT" \
    --ema-fast 9 --ema-slow 21 --warmup-days 60 \
    $FILTERS 2>&1 | tail -3
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
  $PY /app/tools/backtests/monthly_profile.py "$OUT" --capital 1000000 --max-concurrent 2 > "$OUT/_monthly_profile.md" 2>&1 || true
}

run_orb() {
  local LABEL=$1
  local UFILE=$2
  local FROM=$3
  local TO=$4
  local VOL=$5
  local TGT=$6
  local OUT="$OUT_ROOT/orb_${LABEL}"
  mkdir -p "$OUT"
  echo "=== ORB60 $LABEL [$FROM..$TO universe=$UFILE vol=$VOL tgt=$TGT] ==="
  $PY /app/tools/backtests/run_orb60_backtest.py \
    --universe-file "$UFILE" --from "$FROM" --to "$TO" --out "$OUT" \
    --vol-mult "$VOL" --target-atr "$TGT" 2>&1 | tail -3
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
}

# 3 years x 2 universes (N100, N150)
for YEAR in 2023_2024:2023-05-13:2024-05-12 2024_2025:2024-05-13:2025-05-12 2025_2026:2025-05-13:2026-05-12; do
  LBL=$(echo "$YEAR" | cut -d: -f1)
  FRM=$(echo "$YEAR" | cut -d: -f2)
  TO=$(echo "$YEAR" | cut -d: -f3)
  YR_START=$(echo "$FRM" | sed 's/2023-05-13/2023-05-12/;s/2024-05-13/2024-05-12/;s/2025-05-13/2025-05-12/')

  # EMA 9/21 with htf-only filter on N100 + N150
  run_ema921 "n100_htf_${LBL}" "${UNIVERSES_DIR}/nifty100_eq_${YR_START}.json" "$FRM" "$TO" "--htf-filter"
  run_ema921 "n150_htf_${LBL}" "${UNIVERSES_DIR}/nifty150_eq_${YR_START}.json" "$FRM" "$TO" "--htf-filter"

  # ORB-60 N100 + N150
  run_orb "n100_${LBL}" "${UNIVERSES_DIR}/nifty100_eq_${YR_START}.json" "$FRM" "$TO" "1.5" "1.5"
  run_orb "n150_${LBL}" "${UNIVERSES_DIR}/nifty150_eq_${YR_START}.json" "$FRM" "$TO" "1.5" "1.5"

  # EMA 200/400 swing on N100 + N150 (extend Phase 9 winner)
  OUT="$OUT_ROOT/ema200400_n100_${LBL}"
  mkdir -p "$OUT"
  echo "=== EMA 200/400 N100 ${LBL} ==="
  $PY /app/tools/backtests/run_ema_200_400_backtest.py \
    --universe-file "${UNIVERSES_DIR}/nifty100_eq_${YR_START}.json" --from "$FRM" --to "$TO" --out "$OUT" 2>&1 | tail -3
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
  $PY /app/tools/backtests/monthly_profile.py "$OUT" --capital 1000000 --max-concurrent 2 > "$OUT/_monthly_profile.md" 2>&1 || true

  OUT="$OUT_ROOT/ema200400_n150_${LBL}"
  mkdir -p "$OUT"
  echo "=== EMA 200/400 N150 ${LBL} ==="
  $PY /app/tools/backtests/run_ema_200_400_backtest.py \
    --universe-file "${UNIVERSES_DIR}/nifty150_eq_${YR_START}.json" --from "$FRM" --to "$TO" --out "$OUT" 2>&1 | tail -3
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
  $PY /app/tools/backtests/monthly_profile.py "$OUT" --capital 1000000 --max-concurrent 2 > "$OUT/_monthly_profile.md" 2>&1 || true
done

echo "DONE. Results in $OUT_ROOT"

#!/bin/bash
# Phase 14b: EMA 9/21 + ORB-60 on smart universe x 3 years
set -e
mkdir -p /app/exports/backtests/optimize_p14
PY=/usr/local/bin/python
UFILE=/app/exports/backtests/universes/n50_smart.json

for YEAR in 2023_2024:2023-05-13:2024-05-12 2024_2025:2024-05-13:2025-05-12 2025_2026:2025-05-13:2026-05-12; do
  LBL=$(echo "$YEAR" | cut -d: -f1)
  FRM=$(echo "$YEAR" | cut -d: -f2)
  TO=$(echo "$YEAR" | cut -d: -f3)

  OUT=/app/exports/backtests/optimize_p14/ema921_smart_htf_${LBL}
  mkdir -p "$OUT"
  echo "=== EMA 9/21 smart htf $LBL ==="
  $PY /app/tools/backtests/run_ema_200_400_backtest.py --universe-file "$UFILE" --from "$FRM" --to "$TO" --out "$OUT" --ema-fast 9 --ema-slow 21 --warmup-days 60 --htf-filter
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1

  OUT=/app/exports/backtests/optimize_p14/orb_smart_${LBL}
  mkdir -p "$OUT"
  echo "=== ORB-60 smart $LBL ==="
  $PY /app/tools/backtests/run_orb60_backtest.py --universe-file "$UFILE" --from "$FRM" --to "$TO" --out "$OUT"
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
done
echo DONE

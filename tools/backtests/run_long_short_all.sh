#!/bin/bash
# Run all 4 long-short modes for all 3 years, then cap-sim each.
# Sequential to respect container CPU limits.
set -e

UNIV=/app/logs/momrot/universes/n100_current.json
BASE=/app/exports/backtests/long_short
mkdir -p $BASE

declare -a YEARS=(
  "2023-05-13 2024-05-12 y1"
  "2024-05-13 2025-05-12 y2"
  "2025-05-13 2026-05-12 y3"
)

declare -a MODES=(long_only short_only long_short turncoat)

for y in "${YEARS[@]}"; do
  read -r FROM TO TAG <<< "$y"
  for M in "${MODES[@]}"; do
    OUT="$BASE/${M}_${TAG}"
    if [ -f "$OUT/_meta.json" ]; then
      echo "skip already done: $OUT"
      continue
    fi
    echo "=== $M $TAG ($FROM -> $TO) ==="
    cd /app && python tools/backtests/momrot_long_short_backtest.py backtest \
      --universe-file $UNIV \
      --from $FROM --to $TO \
      --mode $M --top-n 5 \
      --out $OUT 2>&1 | tail -10
  done
done

echo
echo "=== CAPITAL SIMULATIONS ==="
RESULTS=$BASE/_results.json
echo "{}" > $RESULTS

for y in "${YEARS[@]}"; do
  read -r FROM TO TAG <<< "$y"
  for M in "${MODES[@]}"; do
    DIR="$BASE/${M}_${TAG}"
    [ -f "$DIR/_meta.json" ] || continue
    echo "--- capsim $M $TAG ---"
    cd /app && python tools/backtests/momrot_long_short_backtest.py capsim \
      --case-dir $DIR --capital 1000000 \
      --max-concurrent 1 --mode $M \
      --out-json $DIR/_capsim.json
  done
done

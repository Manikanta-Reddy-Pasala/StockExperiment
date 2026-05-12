#!/bin/bash
# Optimize EMA 9/21 + ORB-60 + EMA 200/400 — multi-year parameter sweep.
# Goal: all 3 models ≥ 30%/yr.

set -e
OUT_ROOT="${OUT_ROOT:-/app/exports/backtests/optimize}"
mkdir -p "$OUT_ROOT"
PY="/usr/local/bin/python"

# EMA 9/21 variants — try different filter combos + universes
# Variant 1: ema_9_21 selector top-10 (multi-year) raw
# Variant 2: ema_9_21 selector top-10 + volume only
# Variant 3: ema_9_21 N50 + relaxed filters (vol=1.2, min_gap=0.001)
# Variant 4: ema_9_21 selector top-10 + vol=1.2

# ORB-60 variants — different vol/atr/universe
# Variant 1: ORB-60 N500 (wider universe)
# Variant 2: ORB-60 N50 with vol_mult=1.2 (relaxed)
# Variant 3: ORB-60 N50 with target_atr=2.5 (wider target)
# Variant 4: ORB-60 N50 with time filter (10:30-13:00 only)

# Window function
run_ema921_variant() {
  local LABEL=$1
  local UNIVERSE=$2
  local FROM=$3
  local TO=$4
  local EXTRA=$5
  local OUT="$OUT_ROOT/ema921_${LABEL}"
  mkdir -p "$OUT"
  echo "=== EMA9/21 $LABEL [$UNIVERSE $FROM..$TO] $EXTRA ==="
  $PY /app/tools/backtests/run_ema_200_400_backtest.py \
    --universe "$UNIVERSE" --from "$FROM" --to "$TO" --out "$OUT" \
    --ema-fast 9 --ema-slow 21 --warmup-days 60 \
    $EXTRA 2>&1 | tail -5
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
  $PY /app/tools/backtests/monthly_profile.py "$OUT" --capital 1000000 --max-concurrent 2 > "$OUT/_monthly_profile.md" 2>&1 || true
  echo --- result max=2:
  grep -E "^\s+2\s+" "$OUT/_capital_sim.txt"
  echo --- result max=3:
  grep -E "^\s+3\s+" "$OUT/_capital_sim.txt"
}

run_orb_variant() {
  local LABEL=$1
  local UNIVERSE=$2
  local FROM=$3
  local TO=$4
  local VOL=$5
  local TARGET=$6
  local OUT="$OUT_ROOT/orb_${LABEL}"
  mkdir -p "$OUT"
  echo "=== ORB60 $LABEL [$UNIVERSE $FROM..$TO vol=$VOL atr=$TARGET] ==="
  $PY /app/tools/backtests/run_orb60_backtest.py \
    --universe "$UNIVERSE" --from "$FROM" --to "$TO" --out "$OUT" \
    --vol-mult "$VOL" --target-atr "$TARGET" 2>&1 | tail -3
  $PY /app/tools/backtests/realistic_capital_sim.py "$OUT" --capital 1000000 1 2 3 5 8 10 > "$OUT/_capital_sim.txt" 2>&1
  echo --- result max=2:
  grep -E "^\s+2\s+" "$OUT/_capital_sim.txt"
  echo --- result max=5:
  grep -E "^\s+5\s+" "$OUT/_capital_sim.txt"
}

# ============================================================
# EMA 9/21 OPTIMIZATION — 3 years
# ============================================================
for YEAR in 2023_2024:2023-05-13:2024-05-12 \
            2024_2025:2024-05-13:2025-05-12 \
            2025_2026:2025-05-13:2026-05-12; do
  LBL=$(echo "$YEAR" | cut -d: -f1)
  FRM=$(echo "$YEAR" | cut -d: -f2)
  TO=$(echo "$YEAR" | cut -d: -f3)
  # V1: relaxed filters
  run_ema921_variant "v1_relaxed_${LBL}" "nifty50" "$FRM" "$TO" "--min-crossover-gap-pct 0.001 --volume-confirm-mult 1.2"
  # V2: volume only
  run_ema921_variant "v2_volonly_${LBL}" "nifty50" "$FRM" "$TO" "--volume-confirm-mult 1.5"
  # V3: HTF only
  run_ema921_variant "v3_htfonly_${LBL}" "nifty50" "$FRM" "$TO" "--htf-filter"
done

# ============================================================
# ORB60 OPTIMIZATION — N500 + different params, 3 years
# ============================================================
for YEAR in 2023_2024:2023-05-13:2024-05-12 \
            2024_2025:2024-05-13:2025-05-12 \
            2025_2026:2025-05-13:2026-05-12; do
  LBL=$(echo "$YEAR" | cut -d: -f1)
  FRM=$(echo "$YEAR" | cut -d: -f2)
  TO=$(echo "$YEAR" | cut -d: -f3)
  # V1: N50 vol=1.2 target_atr=1.5
  run_orb_variant "v1_relaxed_${LBL}" "nifty50" "$FRM" "$TO" "1.2" "1.5"
  # V2: N50 vol=1.0 (any volume) target_atr=2.0
  run_orb_variant "v2_widetgt_${LBL}" "nifty50" "$FRM" "$TO" "1.0" "2.0"
done

echo "DONE. Results in $OUT_ROOT"

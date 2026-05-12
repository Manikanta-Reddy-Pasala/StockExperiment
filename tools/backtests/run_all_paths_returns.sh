#!/bin/bash
# Run path_returns_analysis.py for all top paths.
# Designed to run inside trading_system_app container.
#
# Usage: bash /app/tools/backtests/run_all_paths_returns.sh

set -e

OUT_DIR="/app/exports/backtests/path_returns"
mkdir -p "$OUT_DIR"

# Path A: EMA 9/21 raw, max=3
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/ema921_top10 \
  --capital 1000000 --max-concurrent 3 \
  --path-name "Path A — EMA 9/21 raw + max=3" \
  --out "$OUT_DIR/PATH_A.md"

# Path B: EMA 9/21 + sector + cal + vol-sizing 2%, max=2
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/ema921_sec_cal \
  --capital 1000000 --max-concurrent 2 \
  --vol-sizing --risk-per-trade-pct 2.0 \
  --path-name "Path B ⭐ — EMA 9/21 + filters + vol-sizing 2%" \
  --out "$OUT_DIR/PATH_B.md"

# Path C: EMA 9/21 + all overlays, max=2
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/ema921_sec_cal \
  --capital 1000000 --max-concurrent 2 \
  --dd-throttle --vol-sizing --risk-per-trade-pct 1.5 \
  --consecutive-loss-pause 3 --consecutive-loss-window 5 \
  --path-name "Path C — EMA 9/21 ultra-defensive" \
  --out "$OUT_DIR/PATH_C.md"

# Path D: BB Squeeze N50 top-19, max=2
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/y1_n50_bb_squeeze \
  --capital 1000000 --max-concurrent 2 \
  --path-name "Path D — BB Squeeze N50 (low DD)" \
  --out "$OUT_DIR/PATH_D.md"

# Path E: EMA 200/400 selector + sector + cal, max=2 (old Phase 5 winner)
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/selector_top10_sec_cal \
  --capital 1000000 --max-concurrent 2 \
  --path-name "Path E — EMA 200/400 + filters (Phase 5 winner)" \
  --out "$OUT_DIR/PATH_E.md"

# Path F: EMA 200/400 selector + sector + cal + vol-2%, max=5
python /app/tools/backtests/path_returns_analysis.py \
  --case /tmp/selector_top10_sec_cal \
  --capital 1000000 --max-concurrent 5 \
  --vol-sizing --risk-per-trade-pct 2.0 \
  --path-name "Path F — EMA 200/400 + vol-sizing 2% + max=5" \
  --out "$OUT_DIR/PATH_F.md"

echo
echo "==> All path reports in $OUT_DIR"
ls -la "$OUT_DIR"

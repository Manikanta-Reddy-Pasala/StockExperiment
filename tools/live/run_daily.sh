#!/bin/bash
# Daily live-trading cron wrapper for Momentum N100 model.
#
# Only deployed model: tools/models/momentum_n100_top5_max1/.
# Always live — orders go straight to Fyers.
#
# Cron suggestions (Indian market hours 09:15-15:30 IST):
#   1st of month 09:00 IST  -- rebalance: emit ENTRY/STOP_HIT signals
#   daily 09:30 IST         -- monitor rotation (force-emit if needed)
#   15:35 IST               -- post-close daily summary
#
# Usage:
#   ./run_daily.sh prefetch       # update OHLCV cache (N100)
#   ./run_daily.sh signals        # emit momentum rotation signals (rebalance-gated)
#   ./run_daily.sh signals-force  # force-emit (ignore rebalance gate)
#   ./run_daily.sh live           # real Fyers orders
#   ./run_daily.sh summary        # post-close NAV/P&L summary
#
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
UNIVERSE_FILE="${UNIVERSE_FILE:-/app/logs/momrot/universes/n100_current.json}"
DATE=$(date +%Y-%m-%d)
SIGNAL_FILE="/app/logs/momrot/signals/${DATE}_momrot_n100.json"

cmd="${1:-help}"

case "$cmd" in
  prefetch)
    echo "[$DATE] Prefetching today's bars to cache..."
    $PYTHON tools/shared/prefetch_ohlcv.py --universe all --days 2 \
      --intervals 1h,D --sleep 0.2
    ;;

  signals)
    echo "[$DATE] Generating Model 3 momentum rotation signals..."
    mkdir -p /app/logs/momrot/signals
    $PYTHON tools/models/momentum_n100_top5_max1/live_signal.py \
      --universe-file "$UNIVERSE_FILE" --top-n 5 --rebalance-only \
      --signals-out "$SIGNAL_FILE"
    ;;

  signals-force)
    echo "[$DATE] Force-emitting Model 3 signals..."
    mkdir -p /app/logs/momrot/signals
    $PYTHON tools/models/momentum_n100_top5_max1/live_signal.py \
      --universe-file "$UNIVERSE_FILE" --top-n 5 --force \
      --signals-out "$SIGNAL_FILE"
    ;;

  live)
    echo "[$DATE] LIVE Fyers execution — placing real orders"
    [ -f "$SIGNAL_FILE" ] && $PYTHON tools/live/fyers_executor.py \
      --signals "$SIGNAL_FILE" --user-id "${USER_ID:-1}"
    ;;

  summary)
    echo "[$DATE] Daily summary:"
    $PYTHON tools/live/daily_summary.py
    ;;

  *)
    echo "Usage: $0 {prefetch|signals|signals-force|live|summary}"
    echo "Env: UNIVERSE_FILE (default /app/logs/momrot/universes/n100_current.json),"
    echo "     USER_ID (1), PYTHON (python)"
    exit 1
    ;;
esac

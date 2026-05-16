#!/bin/bash
# Daily live-trading cron wrapper. Idempotent.
#
# Only deployed model: Model 3 — N100 monthly momentum rotation (top-5, max=1).
# Signal generator: tools/live/momentum_rotation_signal.py
# Universe file: paper_portfolio/universes/n100.json (refresh weekly)
#
# Cron suggestions (Indian market hours 09:15-15:30 IST):
#   1st of month 09:00 IST  -- rebalance: emit ENTRY/STOP_HIT signals
#   daily 09:30 IST         -- monitor rotation (TARGET / STOP)
#   every 30min during market -- mark-to-market
#   15:35 IST               -- post-close daily report
#
# Usage:
#   ./run_daily.sh prefetch    # update OHLCV cache (N100)
#   ./run_daily.sh signals     # emit momentum rotation signals
#   ./run_daily.sh paper       # paper-trade today's signals
#   ./run_daily.sh monitor     # mark-to-market open positions
#   ./run_daily.sh report      # write daily P&L report
#   ./run_daily.sh live        # real Fyers orders (requires LIVE_TRADING=true)
#
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
UNIVERSE_FILE="${UNIVERSE_FILE:-paper_portfolio/universes/n100.json}"
DATE=$(date +%Y-%m-%d)
SIGNAL_FILE="signals/${DATE}_momrot_n100.json"
LEDGER="paper_portfolio/${DATE}.json"

cmd="${1:-help}"

case "$cmd" in
  prefetch)
    echo "[$DATE] Prefetching today's bars to cache..."
    $PYTHON tools/backtests/prefetch_ohlcv.py --universe all --days 2 \
      --intervals 1h,D --sleep 0.2
    ;;

  signals)
    echo "[$DATE] Generating Model 3 momentum rotation signals..."
    mkdir -p signals
    $PYTHON tools/live/momentum_rotation_signal.py \
      --universe-file "$UNIVERSE_FILE" --top-n 5 --rebalance-only \
      --signals-out "$SIGNAL_FILE"
    ;;

  signals-force)
    echo "[$DATE] Force-emitting Model 3 signals (ignores rebalance gate)..."
    mkdir -p signals
    $PYTHON tools/live/momentum_rotation_signal.py \
      --universe-file "$UNIVERSE_FILE" --top-n 5 --force \
      --signals-out "$SIGNAL_FILE"
    ;;

  paper)
    echo "[$DATE] Paper-trading today's signals..."
    mkdir -p paper_portfolio
    [ -f "$SIGNAL_FILE" ] && $PYTHON tools/live/paper_executor.py \
      --signals "$SIGNAL_FILE" --ledger "$LEDGER"
    ;;

  monitor)
    echo "[$DATE] Marking open positions to market..."
    $PYTHON tools/live/position_monitor.py --ledger "$LEDGER"
    ;;

  live)
    if [ "${LIVE_TRADING:-false}" != "true" ]; then
      echo "REFUSING: LIVE_TRADING env var not 'true'. Set explicitly to enable real orders."
      exit 1
    fi
    echo "[$DATE] LIVE Fyers execution (LIVE_TRADING=true) — placing real orders"
    [ -f "$SIGNAL_FILE" ] && $PYTHON tools/live/fyers_executor.py \
      --signals "$SIGNAL_FILE" --user-id "${USER_ID:-1}"
    ;;

  report)
    echo "[$DATE] Daily report:"
    $PYTHON tools/live/daily_report.py --date "$DATE" --ledger "$LEDGER" 2>/dev/null || \
      cat "$LEDGER" 2>/dev/null | python -m json.tool | head -30
    ;;

  full)
    "$0" prefetch
    "$0" signals
    "$0" paper
    "$0" report
    ;;

  *)
    echo "Usage: $0 {prefetch|signals|signals-force|paper|monitor|live|report|full}"
    echo "Env: UNIVERSE_FILE (default paper_portfolio/universes/n100.json),"
    echo "     LIVE_TRADING (false), USER_ID (1), PYTHON (python)"
    exit 1
    ;;
esac

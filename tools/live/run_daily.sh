#!/bin/bash
# Daily live-trading cron wrapper. Idempotent.
#
# Cron suggestions (Indian market hours 09:15-15:30 IST):
#   09:00 IST  -- pre-market: refresh data + generate swing signals
#   09:30 IST  -- 1H bar starts: generate EMA signals + execute
#   every 5min 09:30-11:15 IST -- ORB signals
#   every 30min during market -- monitor + exit triggers
#   15:35 IST  -- post-close: end-of-day report
#
# Usage (manual):
#   ./run_daily.sh prefetch    # update OHLCV cache
#   ./run_daily.sh signals     # generate today's signals (all 4 models)
#   ./run_daily.sh paper       # paper-trade today's signals
#   ./run_daily.sh monitor     # mark-to-market open positions
#   ./run_daily.sh report      # write daily P&L report
#
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
UNIVERSE="${UNIVERSE:-nifty50}"
DATE=$(date +%Y-%m-%d)
MODELS=(ema_200_400 ema_9_21 swing_pullback orb_15min)

cmd="${1:-help}"

case "$cmd" in
  prefetch)
    echo "[$DATE] Prefetching today's bars to cache..."
    $PYTHON tools/backtests/prefetch_ohlcv.py --universe all --days 2 --intervals 1h,15m,D --sleep 0.2
    ;;

  signals)
    echo "[$DATE] Generating signals for $UNIVERSE..."
    mkdir -p signals
    for m in "${MODELS[@]}"; do
      $PYTHON tools/live/signal_generator.py --model "$m" --universe "$UNIVERSE" \
        --signals-out "signals/${DATE}_${m}_${UNIVERSE}.json"
    done
    ;;

  paper)
    echo "[$DATE] Paper-trading today's signals..."
    mkdir -p paper_portfolio
    for m in "${MODELS[@]}"; do
      f="signals/${DATE}_${m}_${UNIVERSE}.json"
      [ -f "$f" ] && $PYTHON tools/live/paper_executor.py --signals "$f" \
        --ledger "paper_portfolio/${DATE}.json"
    done
    ;;

  monitor)
    echo "[$DATE] Marking open positions to market..."
    $PYTHON tools/live/position_monitor.py --ledger "paper_portfolio/${DATE}.json"
    ;;

  live)
    if [ "${LIVE_TRADING:-false}" != "true" ]; then
      echo "REFUSING: LIVE_TRADING env var not 'true'. Set explicitly to enable real orders."
      exit 1
    fi
    echo "[$DATE] LIVE Fyers execution (LIVE_TRADING=true) — placing real orders"
    for m in "${MODELS[@]}"; do
      f="signals/${DATE}_${m}_${UNIVERSE}.json"
      [ -f "$f" ] && $PYTHON tools/live/fyers_executor.py --signals "$f" --user-id "${USER_ID:-1}"
    done
    ;;

  report)
    echo "[$DATE] Daily report:"
    $PYTHON tools/live/daily_report.py --date "$DATE" --ledger "paper_portfolio/${DATE}.json" 2>/dev/null || \
      cat "paper_portfolio/${DATE}.json" | python -m json.tool | head -30
    ;;

  full)
    "$0" prefetch
    "$0" signals
    "$0" paper
    "$0" report
    ;;

  *)
    echo "Usage: $0 {prefetch|signals|paper|live|monitor|report|full}"
    echo "Env: CAPITAL_INR (default 200000), MAX_CONCURRENT (2), MIN_PRICE (50),"
    echo "     MAX_DAILY_LOSS_PCT (-5.0), LIVE_TRADING (false), USER_ID (1),"
    echo "     UNIVERSE (nifty50), PYTHON (python)"
    exit 1
    ;;
esac

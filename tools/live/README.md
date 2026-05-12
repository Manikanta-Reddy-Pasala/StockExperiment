# Live Trading Stack — Python Scripts (no LLM, no cloud)

Pure Python scripts for daily signal generation + paper trading + live
Fyers execution. Built on top of the same strategy modules used in the
backtest harness.

## Scripts

| Script | Purpose |
|---|---|
| `signal_generator.py` | Read cached OHLCV → run strategy → emit signals JSON |
| `risk_manager.py` | Capital lock, max concurrent, position sizing, kill-switch |
| `paper_executor.py` | Consume signals → simulate portfolio → write daily ledger |
| `fyers_executor.py` | Place real Fyers orders (requires `LIVE_TRADING=true`) |
| `position_monitor.py` | Mark-to-market open positions, check SL/T1 triggers |
| `daily_report.py` | End-of-day P&L summary from ledger |
| `run_daily.sh` | Cron wrapper combining all of above |

## Quickstart (paper mode)

```bash
# One-time: ensure OHLCV cache is up to date
python tools/backtests/prefetch_ohlcv.py --universe nifty50 --days 30 --intervals 1h,15m,D

# Generate today's signals for one model
python tools/live/signal_generator.py --model ema_200_400 --universe nifty50

# Paper-trade the signals
CAPITAL_INR=200000 MAX_CONCURRENT=2 \
  python tools/live/paper_executor.py \
  --signals signals/$(date +%F)_ema_200_400_nifty50.json

# Mark-to-market during the day
python tools/live/position_monitor.py

# End-of-day report
python tools/live/daily_report.py --date $(date +%F)
```

## Cron setup (Linux)

```
# Indian market hours 09:15-15:30 IST = 03:45-10:00 UTC
# Pre-market: refresh cache + generate swing signals
0  3 * * 1-5 cd /opt/StockExperiment && tools/live/run_daily.sh prefetch
30 3 * * 1-5 cd /opt/StockExperiment && UNIVERSE=nifty50 tools/live/run_daily.sh signals
40 3 * * 1-5 cd /opt/StockExperiment && tools/live/run_daily.sh paper

# Every 5 min during market hours: monitor + exit triggers
*/5 4-9 * * 1-5 cd /opt/StockExperiment && tools/live/run_daily.sh monitor

# Post-close: end-of-day report
0 10 * * 1-5 cd /opt/StockExperiment && tools/live/run_daily.sh report
```

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `CAPITAL_INR` | 200000 | Locked-in capital pool |
| `MAX_CONCURRENT` | 2 | Max simultaneous open positions |
| `MAX_PER_TRADE_INR` | capital / max_concurrent | Per-trade ₹ cap |
| `MAX_DAILY_LOSS_PCT` | -5.0 | Kill-switch (negative %) |
| `MIN_PRICE` | 50.0 | Penny filter |
| `ENABLE_SHORT` | false | Allow SELL entries |
| `LIVE_TRADING` | false | Set 'true' to enable real Fyers orders |
| `USER_ID` | 1 | Fyers user_id in broker_configurations |
| `UNIVERSE` | nifty50 | nifty50 or nifty500 |

## Going Live (Fyers real orders)

⚠️ Real money. Verify:
1. Backtest results acceptable (see `exports/backtests/WINNERS.md`)
2. Paper mode for 1-2 weeks with no SEV
3. Fyers token in `broker_configurations` for `USER_ID`
4. CAPITAL_INR matches actual ₹ available
5. `MAX_DAILY_LOSS_PCT` kill-switch set conservatively

Then:
```bash
LIVE_TRADING=true CAPITAL_INR=200000 MAX_CONCURRENT=2 \
  python tools/live/fyers_executor.py --signals signals/...json --user-id 1
```

Without `LIVE_TRADING=true`, `fyers_executor.py` forces dry-run mode and
won't place orders.

## Idempotency

- `paper_portfolio/{date}.json` is the single source of truth for the day.
- Re-running `paper_executor.py` with the same signals appends/skips
  correctly (each ENTRY checks `already in symbol`).
- `position_monitor.py` only updates marks; doesn't re-place orders.
- `signal_generator.py` is read-only (cache + strategy → JSON).

## Architecture

```
                       cron
                         |
                         v
   tools/backtests/prefetch_ohlcv.py   <-- Fyers historical API
                         |
                         v
                  Postgres cache
              (historical_data_*)
                         |
                         v
   tools/live/signal_generator.py
                         |
                         v
                  signals/<date>_<model>_<universe>.json
                         |
              ┌──────────┴──────────┐
              v                     v
   tools/live/paper_executor.py    tools/live/fyers_executor.py
              |                     |
              v                     v
   paper_portfolio/<date>.json    Fyers API (real orders)
              |                     |
              v                     v
   tools/live/position_monitor.py (every 5 min) + daily_report.py
```

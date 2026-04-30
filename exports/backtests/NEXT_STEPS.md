# EMA 200/400 Migration — Next Steps Runbook

Local Docker stack is not running, so the three migration / data steps could not
be executed in this session. Below are the exact paste-ready commands to run
once the stack is up. They preserve the existing Fyers login / token flow.

---

## 1. Apply migrations

```bash
# Boot stack (if not already running)
./run.sh dev

# Wait for Postgres to accept connections
docker exec trading_system_db pg_isready -U trader -d trading_system

# Schema migration: new tables + drop legacy 8-21 columns
docker cp migrations/2026_04_30_ema_200_400_strategy.sql \
    trading_system_db:/tmp/2026_04_30_ema_200_400_strategy.sql
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_04_30_ema_200_400_strategy.sql

# Trade-deal wipe (auth tables untouched)
docker cp migrations/2026_04_30_clear_trade_deals.sql \
    trading_system_db:/tmp/2026_04_30_clear_trade_deals.sql
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_04_30_clear_trade_deals.sql

# Verify
docker exec trading_system_db psql -U trader -d trading_system -c "
  SELECT table_name FROM information_schema.tables
   WHERE table_name IN ('historical_data_1h','ema_crossover_state','ema_crossover_signals')
   ORDER BY table_name;"
```

Expected: 3 rows. Both migrations are idempotent — re-running is safe.

---

## 2. Backfill 1H history (Nifty 500 universe)

Requires a valid Fyers token for `user_id=1` (login via
`http://localhost:5001/brokers/fyers` if needed first).

```bash
# Refresh the Nifty 500 cache (in case the index re-balanced)
docker exec trading_system venv/bin/python tools/refresh_nifty500.py

# Pull 120 days of 1H data for the full Nifty 500
docker exec trading_system venv/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
runner = get_ema_crossover_runner()
result = runner.backfill_universe(user_id=1, days=120, max_symbols=500)
print('inserted:', sum(d.get('inserted', 0) for d in result['details']))
print('updated:',  sum(d.get('updated', 0)  for d in result['details']))
print('failed:',   result['failed'])
"
```

Verify rows landed:

```bash
docker exec trading_system_db psql -U trader -d trading_system -c "
  SELECT symbol, COUNT(*) bars,
         MIN(candle_time) first_bar, MAX(candle_time) last_bar
    FROM historical_data_1h
   GROUP BY symbol ORDER BY bars DESC LIMIT 10;"
```

EMA400 needs ≥405 bars per symbol (~100 trading days of 1H).

---

## 3. Trigger first strategy run + verify signals

```bash
docker exec trading_system venv/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().run_for_user(user_id=1, max_symbols=200))
"
```

Verify signals populated:

```bash
docker exec trading_system_db psql -U trader -d trading_system -c "
  SELECT signal_type, COUNT(*) n
    FROM ema_crossover_signals
   WHERE created_at > NOW() - INTERVAL '1 day'
   GROUP BY signal_type ORDER BY n DESC;"

docker exec trading_system_db psql -U trader -d trading_system -c "
  SELECT date, symbol, recommendation, target_price, stop_loss, selection_score
    FROM daily_suggested_stocks
   WHERE strategy = 'ema_200_400'
     AND date = CURRENT_DATE
   ORDER BY selection_score DESC LIMIT 20;"
```

---

## Offline backtest

Local backtest harness uses Yahoo's chart API (no Fyers token, no DB needed).

```bash
# 5-stock smoke set (HDFCBANK, RELIANCE, INFY, TCS, ICICIBANK)
venv/bin/python tools/backtests/run_ema_200_400_backtest.py --days 720

# Full Nifty 500 (~10-15 minutes)
venv/bin/python tools/refresh_nifty500.py
venv/bin/python tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --universe nifty500 --out exports/backtests/nifty500_full

# Sample first N from Nifty 500 (quick check)
venv/bin/python tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --universe nifty500 --limit 25 --out exports/backtests/nifty500_sample
```

Outputs to `exports/backtests/<dir>/<symbol>.md` plus `_summary.md`.

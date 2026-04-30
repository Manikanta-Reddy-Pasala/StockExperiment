# NSE Stock Trading System

**Automated NSE trading on the EMA 200/400 1H crossover strategy.**

---

## Strategy in 60 seconds

Timeframe: **1 hour (NSE)**. Universe: **Nifty 500** (`src/data/symbols/nifty500.csv`,
refresh with `tools/refresh_nifty500.py`).

```
Trend
  BUY trend  : EMA200 crosses above EMA400 on 1H close.
  SELL trend : EMA200 crosses below EMA400 on 1H close.

Buy setup (long side)
  Alert 1  Price breaks the high of the crossover candle and closes above it.
  Alert 2  Price retests EMA200 (closes below it). The retest candle is locked.
  Entry 1  Price breaks the retest candle high and sustains -> BUY.
  Alert 3  Price touches / crosses below EMA400. New retest candle locked.
  Entry 2  Price breaks the new retest candle high and sustains -> BUY (pyramid).

Sell setup is the mirror image.

Risk management
  Stop loss : 1H close on the wrong side of EMA400 -> exit all entries.
  Target    : +/- 5000 points (index symbols) or 1:3 RR (equities).
  Multiple entries allowed under the same trend.
```

The visual reference is the standard "EMA 200 & EMA 400 Crossover Strategy"
chart on NSE 1H bars; signals are emitted on **closed 1H candles only**.

---

## Architecture

```
Docker Services:
  trading_system        Flask web app (Port 5001)
  scheduler             Strategy + cron jobs
  database              PostgreSQL 15
  dragonfly             Redis-compatible cache
```

**Login / token flow** is unchanged from the legacy system: Fyers OAuth,
auto-refresh every 5 hours, manual re-auth via `/brokers/fyers`.

### Data pipeline (1H)

```
Fyers history (resolution=60)
   -> historical_data_1h         (per-symbol OHLCV, EMA200/400 cache)
   -> ema_crossover_strategy     (state machine evaluates each candle)
   -> ema_crossover_state        (per-user/per-symbol stage + entries)
   -> ema_crossover_signals      (audit log of every alert/entry/exit)
   -> daily_suggested_stocks     (today's actionable BUY/SELL picks)
   -> auto_trading_service       (places orders for picks)
```

The strategy runs a minute after every NSE 1H candle close
(10:16, 11:16, 12:16, 13:16, 14:16, 15:31 IST) plus a 22:00 catch-up.

---

## Quick Start

```bash
# 1. Configure
cp .env.example .env
nano .env                           # FYERS_CLIENT_ID, FYERS_SECRET_KEY

# 2. Boot
./run.sh dev

# 3. Apply migrations (first run only)
docker exec trading_system_db psql -U trader -d trading_system \
  -f /migrations/2026_04_30_ema_200_400_strategy.sql
docker exec trading_system_db psql -U trader -d trading_system \
  -f /migrations/2026_04_30_clear_trade_deals.sql

# 4. Authorize Fyers in the web UI
open http://localhost:5001/brokers/fyers

# 5. Backfill 1H history for Nifty 500 (one-shot)
docker exec trading_system venv/bin/python tools/refresh_nifty500.py
docker exec trading_system venv/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
runner = get_ema_crossover_runner()
print(runner.backfill_universe(user_id=1, days=120, max_symbols=500))
"

# 6. Trigger a strategy run
docker exec trading_system venv/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().run_for_user(user_id=1))
"
```

---

## Schedule

| Time (IST)            | Job                                |
|-----------------------|------------------------------------|
| 09:20                 | Auto-trading execution             |
| 10:16, 11:16, 12:16, 13:16, 14:16, 15:31 | EMA 200/400 strategy run (1H close +1m) |
| 10:00 / 11:00 / ... / 15:15 | Position monitoring          |
| 15:20                 | Day-trading position close         |
| 18:00                 | Performance reconciliation         |
| 22:00                 | Strategy catch-up + housekeeping   |
| Sun 03:00             | Cleanup snapshots (>90 days)       |
| Every 5h              | Fyers token API refresh            |
| Every 6h              | Token status check                 |

---

## Database (current)

**Strategy tables (new):**
- `historical_data_1h` — 1H OHLCV + cached EMA200/400
- `ema_crossover_state` — per-user/per-symbol state machine
- `ema_crossover_signals` — every alert/entry/exit emitted

**Strategy tables (existing, repurposed):**
- `daily_suggested_stocks` — today's actionable picks (`strategy='ema_200_400'`)
- `historical_data` — daily OHLCV (kept for screening / sma_50 / sma_200)
- `technical_indicators` — daily SMA50/200 cache (legacy 8/21/DeMarker columns
  retained but no longer populated)

**Trading:**
- `users`, `broker_configurations`, `webauthn_credentials` (auth — never wiped)
- `orders`, `trades`, `positions`, `holdings`
- `auto_trading_settings`, `auto_trading_executions`, `order_performances`
- `dry_run_portfolios`, `dry_run_positions`

---

## Migrations

| File | Purpose |
|------|---------|
| `migrations/2026_04_30_ema_200_400_strategy.sql` | Create new tables, drop legacy 8-21 columns from `stocks`, reset `daily_suggested_stocks` |
| `migrations/2026_04_30_clear_trade_deals.sql`   | TRUNCATE all order / trade / position / dry-run / suggested-stocks rows. Auth tables untouched. |

Run them in order. Both are idempotent.

---

## Key APIs

```bash
curl "http://localhost:5001/api/suggested-stocks?limit=10"
curl "http://localhost:5001/api/charts/stock/NSE:HDFCBANK-EQ"
curl "http://localhost:5001/api/dashboard/metrics"
curl "http://localhost:5001/api/broker/status"
```

Web pages:
- `/` — Dashboard
- `/api/suggested-stocks` — EMA 200/400 picks
- `/settings` — Trading + broker settings
- `/brokers/fyers` — Fyers OAuth (login/token flow unchanged)

---

## Troubleshooting

**No 1H data yet** — run the backfill snippet from Quick Start (step 5).

**Token expired** — go to `/brokers/fyers`, click *Refresh Token*. The auto-refresh
service re-runs every 5 hours.

**No picks today** — check the runner ran:
```sql
SELECT * FROM ema_crossover_signals
ORDER BY created_at DESC LIMIT 20;
```

**Reset everything** — re-run `2026_04_30_clear_trade_deals.sql` (clears trades
only, keeps users / brokers / candle history).

---

## License

Private use only.

## Disclaimer

Stock trading involves substantial risk of loss. This software is provided
"as is", with no warranty. Past performance does not predict future results.

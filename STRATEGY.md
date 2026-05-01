# EMA 200 / 400 1H Crossover Strategy

Production: `77.42.45.12` · Repo HEAD `92b66400` · Universe: 504 NSE Nifty 500.

---

# A. FUNCTIONAL

## What it does

Trend-following on 1H bars. Detects trend via EMA 200 vs EMA 400 crossover,
waits for retest confirmation, takes up to 2 entries per cycle.

## Stage chain (BUY; SELL is mirror)

| # | Stage | Trigger | Action |
|---|-------|---------|--------|
| 1 | Trend ID | EMA 200 crosses above EMA 400 | Lock crossover candle |
| 2 | First Alert | 1H close > crossover candle high | Watch for retest |
| 3 | Second Alert | 1H close < EMA 200 (retest) | Lock retest1 candle |
| 4 | **Entry 1** | 1H close > retest1 high | **BUY** (score 100) |
| 5 | Third Alert | 1H low ≤ EMA 400 | Lock retest2 candle |
| 6 | **Entry 2** | 1H close > retest2 high | **BUY** pyramid (score 90) |
| — | Exit | 1H close < EMA 400 | Close ALL entries |

## Risk model

| Item | Rule |
|------|------|
| Stop loss | 1H close past EMA 400 |
| Target (equity) | 1 : 3 RR (entry + 3 × distance to EMA 400) |
| Target (index) | 5000 absolute pts |
| Pyramid | Entry 2 may repeat on each EMA 400 retest |
| Position size | Max 2 stocks per sector |
| Re-entry | Allowed after next opposite crossover |

## Backtest (Fyers Nifty 500, 720d, 1H)

| Metric | Value |
|--------|-------|
| Trades | 2,695 |
| Win rate | 31.6% |
| **Reward : Risk** | **2.77 : 1** |
| Avg win / loss | +159.50 / −57.55 |
| **Net P&L per unit** | **+9,429** |
| Profitable stocks | 247 / 483 |
| Target hits (always wins) | 758 |
| EMA-exit closes | 1,937 (1,843 losers, 94 winners) |

**Why 68% loss rate works:** R:R 2.77 turns low win rate into +11.05 expectancy.
100% of losses come from EMA 400 close-exits (capped). All target hits win.

## Stage funnel

| Transition | Rate |
|------------|------|
| Crossover → Alert 1 | 85.9% |
| Alert 1 → Alert 2 | 95.5% |
| Alert 2 → **Entry 1** | 77.5% |
| Entry 1 → Alert 3 | 74.1% |
| Alert 3 → **Entry 2** | 46.9% |
| Entry → Exit | 97.1% |

## What user sees in UI

- `/strategies` — strategy description, entry/exit rules, settings
- `/suggested_stocks` — daily picks table (Trend, Stage, Score, Target, EMA 400 Stop)
- Modal — recommendation, selection score, target price, stop loss
- CSV export — same fields

---

# B. TECHNICAL

## Code map

| Layer | File |
|-------|------|
| State machine | `src/services/technical/ema_crossover_strategy.py` |
| Hourly orchestrator | `src/services/technical/ema_crossover_runner.py` |
| 1H Fyers fetcher | `src/services/data/historical_1h_service.py` |
| Universe loader | `src/services/data/nifty500_universe.py` |
| Models | `src/models/historical_models.py` |
| Auto-trade consumer | `src/services/trading/auto_trading_service.py` |
| API route | `src/web/routes/suggested_stocks_routes.py` |
| UI | `src/web/templates/{strategies,suggested_stocks}.html`, `v2/{picks,settings}.html` |
| Backtest harness | `tools/backtests/run_ema_200_400_backtest.py` |
| Migrations | `migrations/2026_04_30_{ema_200_400_strategy,clear_trade_deals}.sql` |

## Schema

**New tables**

| Table | Purpose |
|-------|---------|
| `historical_data_1h` | 1H OHLCV per symbol/ts + cached EMA 200/400 |
| `ema_crossover_state` | Per-user/symbol state (stage, retest1/2, entries) |
| `ema_crossover_signals` | Append-only audit: CROSSOVER, ALERT1-3, ENTRY1-2, EXIT |

**Dropped:** `ema_8`, `ema_21`, `demarker`, `buy_signal`, `sell_signal`, `signal_quality`, `fib_target_1/2/3`, `ema_8_21_score`.

**Wiped (clear_trade_deals.sql):** `trades`, `orders`, `positions`, `holdings`, `auto_trading_executions`, `order_performance`, `dry_run_*`, `daily_suggested_stocks`. Auth tables untouched.

## Data flow

```
Fyers API (1h interval, 95d chunks)
    ↓
Historical1HService.backfill_universe(user_id=1, days=120)
    ↓
historical_data_1h (Postgres)
    ↓
EMACrossoverRunner.run_for_user(user_id)  ← hourly during 09:15-15:30 IST
    ↓
EMACrossoverStrategy.evaluate()           ← state machine per symbol
    ↓
ema_crossover_signals (audit) + ema_crossover_state (incremental)
    ↓
_promote_to_daily_picks() — only ENTRY1/ENTRY2 → daily_suggested_stocks
    ↓
auto_trading_service._select_top_strategies() — query ema_200_400 picks
    ↓
Fyers order placement (live or paper)
```

## API contract — `daily_suggested_stocks`

```sql
strategy        = 'ema_200_400'
model_type      = 'crossover'
recommendation  IN ('BUY', 'SELL')
selection_score = 100  -- Entry 1 (high conviction)
                | 90   -- Entry 2 (pyramid)
target_price    = entry + 3 * |entry - ema_400|   -- equity
                | entry ± 5000                     -- index
stop_loss       = current EMA 400
```

Upsert key: `(date, symbol, strategy, model_type)` — idempotent re-runs.

## State machine — `EMACrossoverState`

| Field | Notes |
|-------|-------|
| `trend` | NONE / BUY / SELL |
| `stage` | 0–5 |
| `crossover_ts/high/low` | Locked at trend flip |
| `retest1_ts/high/low` | Set at ALERT 2 |
| `retest2_ts/high/low` | Set at ALERT 3 (loops) |
| `entries_count` | Total entries this cycle |
| `entry1_price/time`, `entry2_price/time` | First two entries |
| `stop_loss` | EMA 400 at Entry 1 time |
| `target_price` | RR or 5000 pts |
| `position_active` | True between Entry 1 and EXIT |
| `last_evaluated_ts` | Incremental replay marker |

## Config — `StrategyConfig`

```python
target_points     = 5000.0   # index
rr_multiple       = 3.0      # equity
ema_fast_period   = 200
ema_slow_period   = 400
sustain_minutes   = 15       # informational on 1H
```

## Container layout (production VM)

| Container | Role | New code |
|-----------|------|----------|
| `trading_system_app` | Flask UI + API (`:5001`) | ✓ |
| `trading_system_technical_scheduler` | Hourly strategy run | ✓ |
| `trading_system_data_scheduler` | 1H + daily data pipeline | ✓ |
| `trading_system_db` | Postgres 15 | schema migrated |
| `trading_system_dragonfly` | Cache | unchanged |

Volumes: only `./logs`, `./exports`, `./init-scripts` mounted. Code is **baked
into images** — production code changes require `docker compose build`.

## Ops runbook

```bash
# Apply migrations
docker cp migrations/2026_04_30_ema_200_400_strategy.sql trading_system_db:/tmp/
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_04_30_ema_200_400_strategy.sql

# First-time backfill (120d × 504 symbols)
docker exec -w /app trading_system_app /usr/local/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().backfill_universe(user_id=1, days=120, max_symbols=500))"

# Hourly run (manual trigger)
docker exec -w /app trading_system_app /usr/local/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().run_for_user(user_id=1, max_symbols=500))"

# Today's picks
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT symbol, recommendation, target_price, stop_loss, selection_score
  FROM daily_suggested_stocks
 WHERE strategy='ema_200_400' AND date=CURRENT_DATE
 ORDER BY selection_score DESC LIMIT 20;"

# Signal audit
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT signal_type, COUNT(*) FROM ema_crossover_signals
 WHERE created_at > NOW() - INTERVAL '1 day'
 GROUP BY signal_type ORDER BY 2 DESC;"
```

## Backtest harness

```bash
# Yahoo (offline, no DB, no token)
venv/bin/python tools/backtests/run_ema_200_400_backtest.py --days 720 --source yahoo
venv/bin/python tools/backtests/run_ema_200_400_backtest.py --days 720 --source yahoo \
    --universe nifty500 --out exports/backtests/nifty500_full

# Fyers (production data; container)
docker exec -w /app trading_system_app /usr/local/bin/python \
    /app/tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --source fyers --user-id 1 \
    --universe nifty500 --out /app/exports/backtests_fyers/nifty500_full
```

Per-stock report contains: signal counts, P&L summary, **Strategy Cycles**
section (per-cycle stage table with time/price/EMA/note), closed trades with
exit reasons (TARGET / EXIT_EMA400).

## Performance numbers (per session)

- Fyers Nifty 500 backfill: ~30-40 min (504 × 8 chunks × 0.3s rate)
- Hourly strategy run: ~2-3 min for 504 symbols
- Backtest (Yahoo, 720d, 504 symbols): ~12-15 min
- Backtest (Fyers, 720d, 504 symbols): ~30-45 min

## Known limitations

1. Indices target 5000 pts unreachable on 1H (NIFTY moves 50-300/session)
2. No HTF filter (daily trend ignored)
3. Pyramid Entry 2 can loop unbounded
4. No volume/ATR confirmation
5. Fyers history capped at ~2 years for 1H (Yahoo gives ~3 years)

## Future improvements

| Tweak | Expected impact |
|-------|-----------------|
| Daily HTF filter | WR 31.6% → ~40-45% |
| Disable Entry 2 | Halve loss count, halve compound exposure |
| ATR stop instead of EMA 400 close | Tighter losses |
| Volume filter on retest break | Drop low-conviction entries |
| Tighter index target (200-500 pts) | Indices become tradeable |

---

# C. RESULT FILES

| File | Content |
|------|---------|
| `STRATEGY.md` | This doc |
| `exports/backtests/NIFTY500_RESULTS.md` | Yahoo Nifty 500 aggregate |
| `exports/backtests/FYERS_NIFTY500_RESULTS.md` | Fyers vs Yahoo compare |
| `exports/backtests/INDICES_RESULTS.md` | NIFTY/BANKNIFTY/sectoral |
| `exports/backtests/STAGE_HIT_COUNTS.md` | Funnel + BUY/SELL split |
| `exports/backtests/LOSS_ANALYSIS.md` | Loss source breakdown |
| `exports/backtests/<dir>/<symbol>.md` | Per-stock cycle reports |

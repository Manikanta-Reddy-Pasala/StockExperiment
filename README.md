# StockExperiment — EMA 200/400 1H Crossover

NSE swing-trading bot. Universe: Nifty 50/500. Strategy-1 spec (`BTC Trade Rules_V1.1.pdf`)
on 1H bars with audit fixes. Full state-machine doc: `STRATEGY.md`.

Production: `77.42.45.12` · App: <https://stock.oneshell.in>

---

## Strategy — full spec (BUY and SELL)

EMA200 / EMA400 crossover sets trend. Then a 5-stage chain emits up to 2 entries
(retest1 + retest2), each with its own re-entry cap, SL rule, and partial-book
logic. **SL triggers and ALERT2/ALERT3 retest detection are all close-based**
and require the price to come from the correct side ("from upside" for BUY,
"from downside" for SELL).

### BUY setup

| # | Stage | Trigger | Side guard | Action |
|---|---|---|---|---|
| 0 | Wait | — | — | Wait for crossover |
| 1 | Trend ID (BUY) | `EMA200 cross above EMA400` from below + gap ≥ `min_crossover_gap_pct` | — | Lock crossover candle |
| 2 | ALERT1 | `close > crossover_high && high > crossover_high` | — | Watch for retest |
| 3 | ALERT2 (retest1) | `close < EMA200 && low < EMA200` | **prev close > prev EMA200** (true transition from upside) | Lock retest1 candle (`high`, `low`); attempts=0; invalidated=False |
| 3 | retest1 invalidate | `low ≤ EMA400` AND attempts==0 | — | Skip ENTRY1, advance to Stage 4 |
| 3 | **ENTRY1 BUY** | edge cross of `retest1.high` → PENDING; ENTRY fires after `sustain_minutes` AND close still > level | — | up to 4 attempts (1 + 3 re-entries); `SL = current EMA400` (dynamic, close-based exit) |
| 4 | ALERT3 (retest2) | `low ≤ EMA400` | **prev close > prev EMA400** (from upside) | Lock retest2 candle; attempts=0; max_alert3_locks counter ++ |
| 5 | **ENTRY2 BUY** | edge cross of `retest2.high` + sustain | — | up to 4 attempts; `SL = retest2.low` (static, close-based exit) |
| Per-bar | Position management | TP / partial / SL on each open position | — | TARGET / PARTIAL / STOP_HIT |

**BUY risk model (per position)**:

| Item | Rule | Spec / Default |
|---|---|---|
| Target | `entry × (1 + target_pct)` | 10% (configurable to 10/15/20%) |
| Partial @ ENTRY1 | `bar.high ≥ entry × (1 + partial_pct_entry1)` → book 50% qty, trail SL → EMA200 | 5% |
| Partial @ ENTRY2 | `bar.high ≥ entry × (1 + partial_pct_entry2_buy)` → book 50% qty, trail SL → EMA200 | 15% |
| ENTRY1 SL | `bar.close < EMA400` | dynamic |
| ENTRY2 SL | `bar.close < retest2.low` | static |
| Trend reset | opposite crossover OR EMA inversion grace period | sanity_flip_trend |

### SELL setup (mirror)

| # | Stage | Trigger | Side guard | Action |
|---|---|---|---|---|
| 0 | Wait | — | — | Wait for crossover |
| 1 | Trend ID (SELL) | `EMA200 cross below EMA400` from above + gap ≥ `min_crossover_gap_pct` | — | Lock crossover candle |
| 2 | ALERT1 | `close < crossover_low && low < crossover_low` | — | Watch for retest |
| 3 | ALERT2 (retest1) | `close > EMA200 && high > EMA200` | **prev close < prev EMA200** (true transition from downside) | Lock retest1 candle (`high`, `low`); attempts=0; invalidated=False |
| 3 | retest1 invalidate | `high ≥ EMA400` AND attempts==0 | — | Skip ENTRY1, advance to Stage 4 |
| 3 | **ENTRY1 SELL** | edge cross of `retest1.low` (price drops below) + sustain | — | up to 4 attempts; `SL = current EMA400` (dynamic, close-based) |
| 4 | ALERT3 (retest2) | `high ≥ EMA400` | **prev close < prev EMA400** (from downside) | Lock retest2 candle; attempts=0 |
| 5 | **ENTRY2 SELL** | edge cross of `retest2.low` + sustain | — | up to 4 attempts; `SL = retest2.high` (static, close-based) |

**SELL risk model (per position)**:

| Item | Rule | Spec / Default |
|---|---|---|
| Target | `entry × (1 − target_pct)` | 10% |
| Partial @ ENTRY1 | `bar.low ≤ entry × (1 − partial_pct_entry1)` → book 50% qty, trail SL → EMA200 | 5% |
| **Partial @ ENTRY2** | `bar.low ≤ entry × (1 − partial_pct_entry2_sell)` → book 50% qty, trail SL → EMA200 | **5%** ← asymmetric vs BUY (15%) |
| ENTRY1 SL | `bar.close > EMA400` | dynamic |
| ENTRY2 SL | `bar.close > retest2.high` | static |

### Cycle reset

- **Opposite crossover edge** (`cross_dn` cancels BUY trend, vice-versa) — closes all open positions, starts new cycle from Stage 1.
- **EMA inversion sanity flip** — if `state.trend=BUY` but `EMA200 < EMA400` and no edge fired this bar, force end-cycle. Catches missed crossovers on gaps. (`sanity_flip_trend=True` default.)

---

## Configuration parameters (all per-user via `auto_trading_settings.ema_strategy_config` JSONB)

| Group | Parameter | Default | Notes |
|---|---|---|---|
| Profit | `target_pct` | **0.10** | Take-profit. Customizable to 0.10 / 0.15 / 0.20 per spec. |
| Profit | `partial_pct_entry1` | 0.05 | Partial-book trigger at retest1 entry (BUY+SELL). |
| Profit | `partial_pct_entry2_buy` | 0.15 | Partial-book trigger at retest2 BUY. |
| Profit | `partial_pct_entry2_sell` | 0.05 | Partial-book trigger at retest2 SELL (asymmetric). |
| Profit | `partial_qty_frac` | 0.5 | 50% of qty booked at partial. |
| Profit | `re_entry_cap` | 4 | 1 initial + 3 re-entries per alert. |
| Profit | `sustain_minutes` | 15 | Wait after retest break before ENTRY fires. |
| Spec guards | `require_retest_from_upside` | True | Only lock retest candle on real transition. |
| Spec guards | `sanity_flip_trend` | True | Force end-cycle on EMA inversion without edge. |
| Spec guards | `sma_seed_ema` | True | Pine-style EMA (matches Fyers chart). |
| Quality | `min_crossover_gap_pct` | **0.0003** | Filter touching crossings. Elbow on Nifty50 1y. |
| Quality | `volume_confirm_mult` | 0.0 | Skip ENTRY if break-bar vol < mult × avg. 0=off. |
| HTF | `htf_filter_enabled` | True | BUY: close > 200d SMA; SELL: close < 200d SMA. |
| HTF | `htf_buy_period_bars` | 1400 | ~200d SMA on 1H. |
| HTF | `htf_sell_period_bars` | 1400 | Same for SELL. |
| Tuning | `sell_slope_bars` / `_min_pct` | 350 / 0.005 | EMA200 50d slope ≥ 0.5% drop required for SELL. |
| Tuning | `max_alert3_locks_per_cycle` | 0 | Cap retest2 re-locks. 0 = unlimited. |
| Tuning | `retest2_sl_cap_pct` | 0.0 | Tighten ENTRY2 SL. 0 = use spec retest2.low/high. |
| Tuning | `skip_buy` / `skip_sell` / `skip_retest2` | False | Direction toggles. |

Edit at **/settings → EMA 200/400 Strategy Parameters**, or via `PUT /api/auto-trading/ema-strategy/config`.
Loader merges DB overrides onto code defaults at every strategy run.

---

## Backtest results — **ADANIPORTS, 365d Fyers**

Reference symbol while iterating on the strategy. Run other symbols via
`--symbol NSE:<NAME>-EQ` or remove `--symbol` for the full universe.

| Profile | Legs | Win% | Tgt | SL | Sum% |
|---|---|---|---|---|---|
| Spec strict (gap=0, static 10%) | 7 | 57.1% | 0 | 5 | +6.1% |
| **gap=0.0003 + static 10%** ← default | 0 | — | — | — | — |

Single-symbol 1y is a small sample — default filter discards every ADANIPORTS
crossover in this window. Across the full Nifty 50 universe (where it has
proper sample size), gap=0.0003 lifts win rate 38%→61%, cuts SL hits 34×.

```bash
# Reproduce
docker exec -w /app trading_system_app python /app/tools/backtests/run_ema_200_400_backtest.py \
    --symbol 'NSE:ADANIPORTS-EQ' --source fyers --user-id 1 --days 365 \
    --out /app/exports/backtests/adaniports_y1_default
```

### Threshold sweep on `min_crossover_gap_pct` (1y Fyers)

| gap | Legs | Win% | Sum% | Tgt | SL |
|---|---|---|---|---|---|
| 0 (off) | 446 | 38.3% | +331 | 61 | 308 |
| 0.0001 | 209 | 37.8% | +122 | 32 | 140 |
| 0.0002 | 69 | 34.8% | +40 | 12 | 46 |
| **0.0003** ← elbow | 23 | 60.9% | +76 | 7 | 9 |
| 0.0004 | 12 | 100% | +85 | 5 | 0 (small sample) |
| 0.001 | 0 | — | 0 | — | — |

---

## Stack

Python 3.11 / Flask · PostgreSQL 15 · Dragonfly · Fyers API · Docker Compose.

Containers (`docker-compose.yml`):

| Container | Role |
|---|---|
| `trading_system_app` | Flask UI + API (`:5001`) |
| `trading_system_technical_scheduler` | Hourly strategy run + 15m sustain |
| `trading_system_data_scheduler` | 1H + daily data pipeline |
| `trading_system_db` | Postgres 15 |
| `trading_system_dragonfly` | Cache |

Code is **baked into images** — code changes require `docker compose build`.

---

## Run locally

```bash
cp .env.example .env          # add Fyers creds, DATABASE_URL
./run.sh dev
# → http://localhost:5001
```

Tests:
```bash
pip install -r requirements.txt
python -m pytest
```

Backtest (offline, no DB / no token):
```bash
venv/bin/python tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --universe nifty50 --source yahoo \
    --out exports/backtests/run1
```

CLI flags: `--universe smoke|nifty50|nifty500|indices`, `--source auto|fyers|yahoo`,
`--htf-filter`, `--sell-slope-bars N --sell-slope-min-pct X`,
`--max-alert3-locks N`, `--retest2-sl-cap-pct X`,
`--skip-buy / --skip-sell / --skip-retest2`.

Outputs: `_summary.md`, `_summary_buy.md`, `_summary_sell.md`, per-symbol `<symbol>.md`.

---

## Code map

| Layer | File |
|---|---|
| State machine | `src/services/technical/ema_crossover_strategy.py` |
| Hourly orchestrator | `src/services/technical/ema_crossover_runner.py` |
| 1H Fyers fetcher | `src/services/data/historical_1h_service.py` |
| Universe loader | `src/services/data/nifty500_universe.py` |
| Models / state / signals | `src/models/historical_models.py` |
| Auto-trade consumer | `src/services/trading/auto_trading_service.py` |
| API route | `src/web/routes/suggested_stocks_routes.py` |
| Backtest harness | `tools/backtests/run_ema_200_400_backtest.py` |

Schema: `historical_data_1h` (OHLCV + cached EMA 200/400), `historical_data_15m`
(sustain check), `ema_crossover_state` (per user/symbol stage),
`ema_crossover_signals` (audit log), `daily_suggested_stocks`
(`strategy='ema_200_400'`, `model_type='crossover'`).

---

## Ops

```bash
# Apply migrations (idempotent)
for m in 2026_04_30_ema_200_400_strategy.sql \
         2026_05_06_ema_strategy_v2.sql \
         2026_05_07_ema_strategy_v2_tuning.sql \
         2026_05_07_historical_data_15m.sql; do
  docker cp migrations/$m trading_system_db:/tmp/
  docker exec trading_system_db psql -U trader -d trading_system -f /tmp/$m
done

# Backfill universe (120d × 504 symbols, ~30-40 min)
docker exec -w /app trading_system_app python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().backfill_universe(user_id=1, days=120, max_symbols=500))"

# Today's picks
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT symbol, recommendation, target_price, stop_loss, selection_score
  FROM daily_suggested_stocks
 WHERE strategy='ema_200_400' AND date=CURRENT_DATE
 ORDER BY selection_score DESC LIMIT 20;"

# Signal audit (last 24h)
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT signal_type, COUNT(*) FROM ema_crossover_signals
 WHERE created_at > NOW() - INTERVAL '1 day'
 GROUP BY signal_type ORDER BY 2 DESC;"
```

---

## Reference

- `STRATEGY.md` — full spec, state machine, signal taxonomy
- `BTC Trade Rules_V1.1.pdf` — source rules
- `exports/backtests/720d/` — 720-day reference runs
- `exports/backtests/{buy_htf,buy_plain,sell_all,sell_plain}/` — variant reports
- `exports/backtests/capital_report.md` — 200K INR capital simulation

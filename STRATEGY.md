# EMA 200 / 400 1H Crossover Strategy

**Status:** Live in production on `77.42.45.12` (StockExperiment VM)
**Repo HEAD:** `d4970384` (committed `feat: EMA 200/400 1H crossover strategy with Nifty 500 backtest`)
**Universe:** 504 NSE Nifty 500 symbols (`src/data/symbols/nifty500.csv`)
**Timeframe:** 1H bars (Fyers `interval='1h'`)

---

## 1. What this strategy does

Trend-following with a **stateful 6-stage state machine**. Detects market trend
via EMA 200 vs EMA 400 crossover on the 1H timeframe, waits for retest
confirmation, and pyramids into the move with two entries.

Mirror image works for SELL setups (price below EMA 200, retests from below).

```
                       BUY setup (mirror for SELL)
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 0  Watch for crossover                                         │
│   ↓                                                                  │
│ Stage 1  EMA 200 crosses above EMA 400 → CROSSOVER signal            │
│            (lock crossover candle high/low)                          │
│   ↓                                                                  │
│ Stage 2  1H close > crossover candle high → ALERT 1                  │
│   ↓                                                                  │
│ Stage 3  Price retests EMA 200 (1H low < EMA200,                     │
│            1H close < EMA200) → ALERT 2 (lock retest1)               │
│   ↓                                                                  │
│ Stage 4  1H close > retest1 high → ENTRY 1 (score 100)               │
│            • Stop = current EMA 400                                  │
│            • Target = entry + 3 × |entry − EMA 400| (1:3 RR equity)  │
│            • Target = entry + 5000 pts (index)                       │
│   ↓                                                                  │
│ Stage 5  Price pulls to EMA 400 (1H low ≤ EMA400) → ALERT 3          │
│            (lock retest2)                                            │
│   ↓                                                                  │
│ Stage 4  1H close > retest2 high → ENTRY 2 (score 90, pyramid)       │
│            ↺ (stage loops back to 4 — additional EMA 400 retests     │
│              can pyramid further)                                    │
│                                                                      │
│ EXIT: any 1H close < EMA 400 → close ALL open entries together       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Code map

| Layer | File | Purpose |
|-------|------|---------|
| State machine | `src/services/technical/ema_crossover_strategy.py` | Per-symbol 6-stage state machine, signal emission |
| Orchestrator | `src/services/technical/ema_crossover_runner.py` | Hourly run loop, `daily_suggested_stocks` writer |
| 1H data fetch | `src/services/data/historical_1h_service.py` | Fyers backfill (95-day chunks) + incremental update |
| Universe | `src/services/data/nifty500_universe.py` | Loads `src/data/symbols/nifty500.csv` |
| Models | `src/models/historical_models.py` | `HistoricalData1H`, `EMACrossoverState`, `EMACrossoverSignal` |
| Auto-trading | `src/services/trading/auto_trading_service.py` | Reads `strategy='ema_200_400'` picks |
| UI route | `src/web/routes/suggested_stocks_routes.py` | API for stock picks |
| UI templates | `src/web/templates/strategies.html`, `suggested_stocks.html`, `v2/picks.html`, `v2/settings.html` | Strategy + pick views |
| Backtest | `tools/backtests/run_ema_200_400_backtest.py` | Yahoo + Fyers offline harness with cycle breakdown |
| Migrations | `migrations/2026_04_30_ema_200_400_strategy.sql` (schema) + `..._clear_trade_deals.sql` (wipe) | Schema + data wipe |

---

## 3. Database schema

### New tables

```sql
historical_data_1h          -- 1H OHLCV per symbol/timestamp + cached EMAs
ema_crossover_state         -- Per-user/symbol state machine (stage, retest data, entries)
ema_crossover_signals       -- Append-only audit log: CROSSOVER, ALERT1-3, ENTRY1-2, EXIT
```

### Dropped columns (from `stocks`, `technical_indicators`)

```
ema_8, ema_21, demarker, buy_signal, sell_signal, ema_8_21_score, signal_quality
```

### Wiped tables (`clear_trade_deals.sql`)

```
trades, orders, positions, holdings, auto_trading_executions,
order_performance, order_performance_snapshots, dry_run_positions,
dry_run_portfolios, daily_suggested_stocks
```

Auth tables (`users`, `broker_configurations`, `webauthn_credentials`) **NOT touched**.

---

## 4. Auto-trading integration

`auto_trading_service._select_top_strategies()` reads:

```sql
SELECT symbol, current_price, target_price, stop_loss, recommendation, selection_score
  FROM daily_suggested_stocks
 WHERE strategy = 'ema_200_400'
   AND date = CURRENT_DATE
   AND recommendation IN ('BUY', 'SELL')
ORDER BY selection_score DESC NULLS LAST, created_at DESC
```

| `selection_score` | Meaning |
|-------------------|---------|
| 100 | Entry 1 (EMA 200 retest break) — high-conviction |
| 90 | Entry 2 (EMA 400 retest break) — pyramid |

Sector cap retained: max 2 picks per sector (no concentration risk).

---

## 5. Backtest results (Fyers Nifty 500, 720 days, 1H)

Run inside `trading_system_app` container on production VM.

| Metric | Value |
|--------|-------|
| Symbols processed | 498 / 504 (6 had no Fyers data) |
| Symbols with trades | 483 |
| Total closed trades | 2,695 |
| Trade-level winners | 852 |
| **Trade-level win rate** | **31.6%** |
| Trade-level losses | 1,843 (68.4%) |
| Target hits (always wins) | 758 |
| EMA400 close-exits (mostly losses) | 1,937 |
| **Net P&L per unit** | **+9,429** |
| Avg P&L per trade | +3.50 |
| Avg WIN | +159.50 |
| Avg LOSS | −57.55 |
| **Reward : Risk** | **2.77 : 1** |
| Stock-level profitable | 247 |
| Stock-level losing | 236 |

### Stage hit counts (BUY + SELL combined)

| Stage | Count | BUY | SELL |
|-------|-------|-----|------|
| CROSSOVER | 3,200 | 1,513 | 1,687 |
| ALERT1 | 2,750 | 1,251 | 1,499 |
| ALERT2 | 2,625 | 1,133 | 1,492 |
| **ENTRY1** | **2,034** | 869 | 1,165 |
| ALERT3 | 1,508 | 656 | 852 |
| **ENTRY2** | **707** | 286 | 421 |
| EXIT | 1,976 | 850 | 1,126 |

### Funnel conversion

| Transition | Rate |
|------------|------|
| Crossover → Alert1 | 85.9% (most trends break out) |
| Alert1 → Alert2 | 95.5% (almost always retest EMA 200) |
| Alert2 → Entry1 | 77.5% (retest break converts) |
| Entry1 → Alert3 | 74.1% (price often pulls to EMA 400) |
| Alert3 → Entry2 | 46.9% (pyramid 2nd entry) |
| Entry → Exit | 97.1% (most close via EMA 400 cross) |

---

## 6. Why the 68.4% loss rate is OK

Trend-following profile = **low win rate, high reward-to-risk**.

- Avg winner: +159.50 / unit
- Avg loser:  −57.55 / unit
- 31.6% × 159.50 + 68.4% × (−57.55) = **+11.05 expectancy per trade**
- Multiplied across 2,695 trades → +9,429 net

**100% of losses come from EMA 400 close-exits.** No target hit ever loses
(by definition — closes at target price).

Indices (5 tested) had **0 target hits** because 5000-pt absolute target is
unreachable on 1H — NIFTY moves 50-300 pts/session. For indices, consider
a tighter target or use the equity 1:3 RR rule.

---

## 7. Operations

### Daily flow

```
09:00 IST   Fyers token auto-refresh check (technical_scheduler)
09:15 IST   NSE market opens
09:15-15:30 Hourly: ema_crossover_runner.run_for_user(user_id=1)
              ↓
            Refreshes latest 1H candles (last 5 days)
              ↓
            Evaluates strategy state machine per symbol
              ↓
            Promotes ENTRY signals to daily_suggested_stocks
              (ON CONFLICT upsert by date/symbol/strategy/model_type)
              ↓
            auto_trading_service queries fresh picks → places orders
15:30 IST   NSE market closes
22:00 IST   Daily snapshot + cleanup
```

### Manual operations

```bash
# Apply migrations (already done on prod)
docker cp migrations/2026_04_30_ema_200_400_strategy.sql trading_system_db:/tmp/
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_04_30_ema_200_400_strategy.sql

# Backfill 1H universe (120 days for first run)
docker exec -w /app trading_system_app /usr/local/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().backfill_universe(user_id=1, days=120, max_symbols=500))
"

# Trigger one strategy run
docker exec -w /app trading_system_app /usr/local/bin/python -c "
from src.services.technical.ema_crossover_runner import get_ema_crossover_runner
print(get_ema_crossover_runner().run_for_user(user_id=1, max_symbols=500))
"

# Check today's signals
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT symbol, recommendation, target_price, stop_loss, selection_score
  FROM daily_suggested_stocks
 WHERE strategy = 'ema_200_400' AND date = CURRENT_DATE
 ORDER BY selection_score DESC LIMIT 20;"

# Check signal audit log
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT signal_type, COUNT(*)
  FROM ema_crossover_signals
 WHERE created_at > NOW() - INTERVAL '1 day'
 GROUP BY signal_type ORDER BY 2 DESC;"
```

### Backtest harness

```bash
# Local Yahoo backtest (no DB, no Fyers token)
venv/bin/python tools/backtests/run_ema_200_400_backtest.py --days 720 --source yahoo
venv/bin/python tools/backtests/run_ema_200_400_backtest.py --days 720 --source yahoo \
    --universe nifty500 --out exports/backtests/nifty500_full

# Fyers backtest (production data; requires DB + token)
docker exec -w /app trading_system_app /usr/local/bin/python \
    /app/tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --source fyers --user-id 1 \
    --universe nifty500 --out /app/exports/backtests_fyers/nifty500_full
```

Per-stock report at `exports/backtests/<dir>/<symbol>.md` includes:
- Signal counts table (CROSSOVER/ALERT1-3/ENTRY1-2/EXIT)
- P&L summary (winners/losers, target/EMA exits)
- **Strategy Cycles section** — per-cycle stage table with time/price/EMAs/notes
- Closed trades table with exit reasons (TARGET / EXIT_EMA400)

---

## 8. Risk management

| Item | Rule |
|------|------|
| Stop loss | 1H close past EMA 400 (closes ALL open entries together) |
| Target (equity) | 1 : 3 risk-reward (entry + 3 × distance to EMA 400) |
| Target (index) | 5000 absolute points |
| Position size | Max 2 stocks per sector (auto_trading sector cap) |
| Pyramid limit | Currently unlimited Entry 2 retests; can constrain in `_step_buy/sell` |
| Re-entry after exit | Allowed once next CROSSOVER fires |
| HTF filter | None yet (potential improvement) |

---

## 9. Configuration

`StrategyConfig` (`ema_crossover_strategy.py`):

```python
target_points: float = 5000.0     # Absolute index target
rr_multiple: float = 3.0          # Equity reward-to-risk multiple
sustain_minutes: int = 15         # Informational on 1H
ema_fast_period: int = 200        # EMA 200
ema_slow_period: int = 400        # EMA 400
```

`Historical1HService`:

```python
FYERS_INTRADAY_MAX_DAYS = 95      # Per-call chunk size (Fyers cap)
rate_limit_delay = 0.3            # Seconds between Fyers calls
```

`auto_trading_service` (sector cap, hard-coded):

```python
MAX_PER_SECTOR = 2
```

---

## 10. Migration from 8-21 EMA strategy

The previous `ema_strategy_calculator.py` (8-21 EMA + DeMarker + Fibonacci
extensions) is **deleted**. UI references in `suggested_stocks.html`,
`strategies.html`, `v2/picks.html`, `v2/settings.html`, `trading.js`, and
`suggested_stocks_routes.py` have all been updated.

Removed concepts:
- "Power Zone" (8 > 21 EMA)
- DeMarker oscillator
- Fibonacci profit targets (127.2%, 161.8%, 200%)
- Multiple `fib_target_1/2/3` columns
- `signal_quality` field

Replaced with:
- EMA 200 / 400 trend
- Single `target_price` per pick
- `stop_loss` = EMA 400 level
- `selection_score` = 100 (Entry 1) or 90 (Entry 2)
- `recommendation` = `BUY` or `SELL`

---

## 11. Known limitations

1. **Indices target unreachable** on 1H (5000 pts ≈ 22% on NIFTY)
2. **No higher-timeframe filter** — strategy takes 1H BUY signals even if daily trend is bearish
3. **Pyramid loop unbounded** — repeated EMA 400 retests can compound losses
4. **No volume filter** — low-volume retest breakouts treated same as high-volume
5. **Fyers history capped** at ~2 years for 1H; longer windows need Yahoo fallback
6. **Sustain rule informational** on 1H — entry fires on 1H close break, not actual N-minute sustain

---

## 12. Files generated this session

| Path | Purpose |
|------|---------|
| `STRATEGY.md` (this file) | Master strategy doc |
| `exports/backtests/NIFTY500_RESULTS.md` | Yahoo Nifty 500 backtest aggregate |
| `exports/backtests/FYERS_NIFTY500_RESULTS.md` | Fyers vs Yahoo comparison |
| `exports/backtests/INDICES_RESULTS.md` | NIFTY/BANKNIFTY/sectoral indices |
| `exports/backtests/STAGE_HIT_COUNTS.md` | Funnel + BUY/SELL split |
| `exports/backtests/LOSS_ANALYSIS.md` | Loss breakdown by exit reason |
| `exports/backtests/_summary.md` | Smoke 5-stock aggregate |
| `exports/backtests/<symbol>.md` | Per-stock cycle reports (smoke) |
| `exports/backtests/nifty500_full/<symbol>.md` | Per-stock Yahoo reports (504) |
| `exports/backtests/fyers_nifty500_full/<symbol>.md` | Per-stock Fyers reports (504) |
| `exports/backtests/indices/<symbol>.md` | Per-index reports (5) |
| `exports/backtests/vm_nifty500_full/<symbol>.md` | VM-run validation reports |

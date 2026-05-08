# StockExperiment — EMA 200/400 1H Crossover

NSE swing-trading bot. Universe: Nifty 50/500. Strategy: EMA 200 vs EMA 400 on 1H bars
with BTC trade rules. Spec: `BTC Trade Rules_V1.1.pdf`. Full doc: `STRATEGY.md`.

Production: `77.42.45.12` · App: <https://stock.oneshell.in>

---

## Strategy in 30 seconds

```
Trend       EMA200 cross EMA400        BUY (above) / SELL (below)
ALERT1      close beyond crossover candle
ALERT2      retest EMA200 (lock retest1 candle)
ENTRY1      break retest1 high + 15m sustain   SL = current EMA400 (dynamic)
ALERT3      retest EMA400 (lock retest2 candle)
ENTRY2      break retest2 high + 15m sustain   SL = retest2.low (static)

Per-position:  TP = entry × 1.30
               Partial = bar.high ≥ entry × 1.15 → book 50%, trail SL → entry
               Re-entry cap: 4 attempts each at retest1/retest2
               Trend reset: opposite crossover only
```

Live filters layered on the spec:

| Filter | BUY | SELL |
|---|---|---|
| HTF SMA(200d) | close > 200d SMA | close < 200d SMA |
| EMA200 slope (50d) | — | EMA200 dropped ≥ 0.5% over 50d |

Pure-spec mode: pass `StrategyConfig()`.

---

## Backtest — Nifty 50, 720d

| Variant | Win % | Avg %/leg | Sum % | BUY | SELL |
|---|---|---|---|---|---|
| Pure spec | 22.4 | 0.99 | +1,309 | +1,430 | −121 |
| HTF only | 27.5 | 1.69 | +1,495 | +1,629 | −134 |
| **HTF + slope50 (live)** | **30.0** | **1.96** | **+1,697** | +1,552 | +145 |
| HTF + alert3-cap2 + slope50 | 36.3 | 3.10 | +2,249 | +2,104 | +145 |

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
- `exports/backtests/nifty50_htf_slope50/` — live-config reference run
- `exports/backtests/nifty50_slope50/` — best variant (alert3-cap2)

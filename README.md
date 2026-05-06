# NSE Stock Trading System

EMA 200/400 1H crossover strategy with BTC trade rules. Universe: Nifty 50/500.

---

## Strategy

**Timeframe**: 1H bars. **Trend**: EMA200 crosses EMA400.

```
BUY trend   EMA200 above EMA400 (crossover candle locked)
SELL trend  EMA200 below EMA400

Stage chain (BUY; SELL is mirror)
  ALERT1  close > crossover candle high
  ALERT2  close < EMA200 (retest1 candle locked)
  ENTRY1  break retest1 high + sustain (cap 4 attempts)
          - omitted if EMA400 touched before retest1 break
          - SL = current EMA400 (dynamic, per-position)
  ALERT3  low <= EMA400 (retest2 candle locked)
  ENTRY2  break retest2 high + sustain (cap 4 attempts)
          - SL = retest2 candle low
```

**Risk per position**:
- Target = entry × (1 + 30%)
- Partial = bar.high >= entry × 1.15 → book 50% qty, trail SL to entry
- Trend reset: opposite crossover only
- Multiple positions allowed (pyramid).

Spec source: `BTC Trade Rules_V1.1.pdf`.

---

## Production filters (live config)

Default `EMACrossoverRunner` ships with empirically tuned filters. Spec
state machine UNCHANGED — filters only gate the CROSSOVER signal.

| Filter | BUY | SELL |
|---|---|---|
| **HTF SMA (200d)** | close > 200d SMA | close < 200d SMA |
| **EMA200 slope (50d)** | — | EMA200 dropped ≥0.5% over last 50d |

```python
StrategyConfig(
    htf_filter_enabled=True,
    htf_buy_period_bars=1400,    # 200d on 1H
    htf_sell_period_bars=1400,
    sell_slope_bars=350,         # 50d
    sell_slope_min_pct=0.005,    # 0.5%
)
```

To disable filters (pure-spec mode): pass `StrategyConfig()`.

---

## Backtest results — Nifty 50, 720d (Yahoo)

| Variant | Win % | Avg %/leg | Sum % | BUY sum % | SELL sum % |
|---|---|---|---|---|---|
| Pure spec (no filters) | 22.4 | 0.99 | +1,309 | +1,430 | −121 |
| HTF only | 27.5 | 1.69 | +1,495 | +1,629 | −134 |
| **HTF + slope50 (live)** | **30.0** | **1.96** | **+1,697** | **+1,552** | **+145** |
| HTF + alert3-cap2 + slope50 | 36.3 | 3.10 | +2,249 | +2,104 | +145 |

**Live config** (HTF+slope50) flips SELL from −134% → +145%. Optional
`max_alert3_locks_per_cycle=2` adds another +30% if enabled.

---

## Build commands

### Java (Spring Boot)
```bash
./mvnw clean package -DskipTests
./mvnw test -Dtest=ClassName#methodName
```

### Python (Flask backend)
```bash
pip install -r requirements.txt && python main.py     # Port 5100
python -m pytest
```

### Frontend
```bash
cd PosFrontend && npm install && npm run start         # React 16
cd PosAdmin && yarn install && yarn dev                # React 18, port 5174
```

---

## Backtest harness

```bash
# Default (Fyers preferred, Yahoo fallback per-symbol)
venv/bin/python tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --universe nifty50 --out exports/backtests/run1
```

CLI flags:
- `--universe` smoke / nifty50 / nifty500 / indices
- `--source` auto / fyers / yahoo
- `--htf-filter` enable HTF SMA gate
- `--sell-slope-bars N --sell-slope-min-pct X` slope confirm for SELL
- `--max-alert3-locks N` cap retest2 re-locks per cycle (0=off)
- `--retest2-sl-cap-pct X` cap ENTRY2 SL distance (0=off)
- `--skip-retest2` / `--skip-sell` / `--skip-buy` direction toggles

Outputs:
- `_summary.md` (combined)
- `_summary_buy.md` (BUY only)
- `_summary_sell.md` (SELL only)
- per-stock `<symbol>.md` with cycles + closed legs in %

---

## Migrations

```bash
docker cp migrations/2026_05_06_ema_strategy_v2.sql        trading_system_db:/tmp/
docker cp migrations/2026_05_07_ema_strategy_v2_tuning.sql trading_system_db:/tmp/
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_05_06_ema_strategy_v2.sql
docker exec trading_system_db psql -U trader -d trading_system \
    -f /tmp/2026_05_07_ema_strategy_v2_tuning.sql
```

---

## Code map

| Layer | File |
|---|---|
| Strategy state machine | `src/services/technical/ema_crossover_strategy.py` |
| Hourly runner | `src/services/technical/ema_crossover_runner.py` |
| 1H Fyers fetcher | `src/services/data/historical_1h_service.py` |
| Universe loader | `src/services/data/nifty500_universe.py` |
| State + signals models | `src/models/historical_models.py` |
| Auto-trade consumer | `src/services/trading/auto_trading_service.py` |
| API route | `src/web/routes/suggested_stocks_routes.py` |
| Backtest harness | `tools/backtests/run_ema_200_400_backtest.py` |

---

## Containers (production VM 77.42.45.12)

```
trading_system_app                 Flask UI + API (:5001)
trading_system_technical_scheduler hourly strategy run
trading_system_data_scheduler      1H + daily data pipeline
trading_system_db                  Postgres 15
trading_system_dragonfly           cache
```

Code is **baked into images** — code changes require `docker compose build`.

---

## Ops cheatsheet

```bash
# Apply migrations (idempotent)
ssh root@77.42.45.12 "cd /opt/trading_system && \
    docker exec trading_system_db psql -U trader -d trading_system \
        -f /tmp/<migration>.sql"

# Backfill universe (120d × 504 symbols)
docker exec -w /app trading_system_app /usr/local/bin/python -c "
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

## Reference: deeper docs

- `STRATEGY.md` — full strategy spec, state machine, signal taxonomy
- `BTC Trade Rules_V1.1.pdf` — original rules document
- `exports/backtests/nifty50_htf_slope50/` — live-config backtest reference
- `exports/backtests/nifty50_slope50/` — best variant (with alert3-cap2)

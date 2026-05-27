# Trading Models

**4 equity models.** All but the real-N100 baseline deliver ≥ 130% CAGR over 3-year backtest (2023-05-15 → 2026-05-12; midcap to 2026-05-15). Each subfolder is self-contained: data ingest, backtest, live signal, scheduler, and docs. (The options model `finnifty_ic_otm4_w300_lots5` was removed 2026-05-25 — NIFTY/FinNifty IC abandoned; use equity momentum instead.)

## Models

| # | Folder | Category | Universe | CAGR (3yr) | Max DD | LIVE |
|--:|---|---|---|---:|---:|:-:|
| 1 | `momentum_n100_top5_max1` | Large-cap equity | Real NSE Nifty 100 | **+125.13%** | 28.21% | ✅ |
| 2 | `momentum_pseudo_n100_adv` | Large/mid blend | Top-100 ADV from N500 MINUS Small | **+149.15%** | 16.17% | ✅ |
| 3 | `midcap_narrow_60d_breakout` | Mid+small equity | Top-100 ADV from N500 MINUS Large | **+141.73%** | 8.12% | ✅ |
| 4 | `n20_daily_large_only` | Top-20 ADV + Nifty 100 | Top-20 ADV + uptrend + NSE Nifty 100 | **+139.55%** | 25.66% | ✅ |

## How universes are constructed (per model)

### 1. momentum_n100_top5_max1 — REAL Nifty 100 (LIVE)
- Source: `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv`
- Refresh: `python tools/refresh_nifty100.py` after each NSE Mar/Sep rebalance
- Cached: `src/data/symbols/nifty100.csv` (104 stocks)
- Selection: all 104 → rank by 30-day return → pick top-1 monthly
- **No filter**: NSE already curates the constituents (free-float market cap leaders)

### 2. momentum_pseudo_n100_adv — Pseudo-N100 by ADV (LIVE)
- Source: `src/data/symbols/nifty500.csv` (NSE 500)
- Compute 20-day ADV = avg(close × volume) per stock
- Sort descending → take top 100 = pseudo-N100
- Rebuilt at each year-start using current data at that time (PIT-safe)
- Differs from real N100 by 47 stocks: BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, IRFC, IDEA, ITI, NBCC, PAYTM, COFORGE, DIXON, COHANCE, HFCL, GROWW etc. (retail-volume mid-caps captured by ADV but excluded from NSE free-float ranking)

### 3. midcap_narrow_60d_breakout — Pseudo-midcap by ADV
- Source: `src/data/symbols/nifty500.csv`
- Same ADV calc as pseudo-N100
- **Take top-100** by ADV
- **Exclude Large-caps**: drop stocks in NSE Nifty 100 → **~42 genuine midcaps** (live `build_universe.py` mirrors this exactly since 2026-05-26)
- Real NSE Nifty Midcap 150 on same strategy = -18% CAGR (without lookahead)

### 4. n20_daily_large_only — Top-20 ADV ∩ Nifty 100 (LIVE)
- Source: `src/data/symbols/nifty500.csv` + `src/data/symbols/nifty100.csv`
- Compute 20-day ADV per stock, take top-20, rebuilt daily
- Gate: close > 200d SMA (uptrend) AND stock in NSE Nifty 100
- Rank survivors by 30d return → pick top-1, rotate daily

## Strategy templates

### Equity rotation (n100_top5_max1 + pseudo_n100_adv)
- Monthly rebalance (1st of month)
- Rank universe by 30-day return
- Hold top-1 (max_concurrent = 1)
- Exit on rotation (when not rank-1)
- No SL, no target

### Equity swing breakout (midcap_narrow_60d_breakout)
- Daily scan
- Entry: 40-day high + vol > 2× 20d avg + close > 200d SMA
- max_concurrent = 1
- Exits: +100% TARGET, -20% STOP from entry, -20% TRAIL from peak after +10%, 120-day MAX_HOLD
- **SMA20 exit DISABLED** (V2 winner config)

## Per-model folder layout

```
tools/models/<name>/
├── README.md            strategy spec, results, reproduction
├── __init__.py
├── backtest.py          backtest engine
├── build_universe.py    universe builder
├── live_signal.py       live signal emitter → JSON
├── data_pull.py         daily/weekly data ingest jobs
├── cron.py              schedule registration (data + trading)
└── trade_ledger.json    backtest output (trades + open + summary)
```

Results land in `exports/models/<name>/`:

```
exports/models/<name>/
├── SUMMARY.md           one-page summary: yearly ROI, top winners, caveats
└── TRADE_LEDGER.md      full trade-by-trade ledger
```

## Data requirements per model

| Model | N100 OHLCV | N500 OHLCV | Universe CSV |
|---|---|---|---|
| `momentum_n100_top5_max1` | ✅ | — | NSE Nifty 100 (quarterly refresh) |
| `momentum_pseudo_n100_adv` | — | ✅ | derived (ADV rank from N500) |
| `midcap_narrow_60d_breakout` | — | ✅ | derived (top-100 ADV from N500 minus Nifty-100) |
| `n20_daily_large_only` | — | ✅ | NSE Nifty 100 (quarterly refresh) + ADV rank from N500 |

## Saga pipeline (data_scheduler 21:00 IST daily, admin trigger button)

`src/services/data/pipeline_saga.py` runs 4 steps:

| Step | Purpose |
|---|---|
| 1. SYMBOL_MASTER | Refresh Fyers symbol master |
| 2. STOCKS | Populate `stocks` table (price, market_cap, sector) |
| 3. HISTORICAL_DATA | Pull daily OHLCV into `historical_data` |
| 6. PIPELINE_VALIDATION | Row-count quality check |

Invoked by:
- `data_scheduler.py` at 21:00 IST daily (cron)
- Admin Triggers page POST `/admin/trigger/pipeline` (manual)
- Admin Triggers page POST `/admin/trigger/all` (sequential wrapper)

## Adding a new model

1. `mkdir tools/models/<new_name>/` + `touch __init__.py`
2. Write `data_pull.py` + `cron.py` exposing `register_data_jobs` and `register_trading_jobs`
3. Add import + register call into `data_scheduler.py` and `scheduler.py`
4. Write `README.md` covering: stock category, universe construction, strategy rules, backtest result, caveats
5. Write `SUMMARY.md` + `TRADE_LEDGER.md` in `exports/models/<new_name>/`

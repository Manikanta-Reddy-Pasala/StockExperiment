# Trading Models

3 equity + 1 options. Each subfolder is a self-contained strategy: data ingest, backtest, live signal, scheduler, and docs. Models slot into the main schedulers via uniform `cron.py` register pattern.

## Models by stock category

| Category | Model | Stock universe | Selection logic | CAGR (3yr) | Max DD | LIVE? |
|---|---|---|---|---:|---:|:-:|
| **Large-cap (production, REAL universe)** | `momentum_n100_top5_max1` v2 | Real NSE Nifty 100 (104 stocks, `src/data/symbols/nifty100.csv`) | Top-1 by 30-day return, monthly rotation | +80.38% | -29.71% | ✅ |
| **Large-cap (lookahead V1, comparison)** | `momentum_n100_top5_max1` v1 | Pseudo-N100 (top-100 N500 by 20-day ADV) | Top-1 by 30-day return, monthly rotation | +136.39% | -16.15% | — |
| **Mid-cap** | `midcap_narrow_60d_breakout` | Pseudo-midcap (N500, skip top-30 ADV, take next 100) | 40d breakout high + vol>2× + close>200SMA → top-1 by volume ratio | +337.62% (lookahead) | -6.76% | ❌ |
| **Options** | `finnifty_ic_otm4_w300_lots5` | FINNIFTY weekly/monthly options | Sell OTM 4% CE+PE Iron Condor, buy ±300pt wings, 5 lots | +337.6%/yr | -13.88% | ❌ |

3rd equity (small-cap) **not implemented** — tested variants (mid+small-only universe +20% CAGR, full N500 +2% CAGR) failed; small-cap rotation = death spiral in Indian regime. Use Nifty Smallcap 100 ETF for passive small-cap exposure.

## How stock lists are created

### Large-cap: REAL Nifty 100 (LIVE)
- Source: `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv`
- Refresh: `python tools/refresh_nifty100.py` (run manually after NSE Mar/Sep rebalance)
- Cached: `src/data/symbols/nifty100.csv` (104 stocks)
- Builder: `tools/models/momentum_n100_top5_max1/build_universe.py --out <path>`
  - Reads CSV → emits JSON with all 104 stocks
- Strategy then ranks by 30d momentum and picks top-1

### Large-cap (V1 lookahead): Pseudo-N100 by ADV
- Source: Nifty 500 (`src/data/symbols/nifty500.csv`)
- Compute 20-day ADV (close × volume) per stock
- Sort descending → top 100 = "pseudo-N100"
- ⚠️ Includes mid/small-caps with retail-heavy volume (HFCL, BSE, GROWW, COHANCE, DIXON etc.) — 47/100 NOT in real NSE Nifty 100
- Result inflated by lookahead bias (today's ADV applied retroactively to 2023)

### Mid-cap: Pseudo-midcap by ADV
- Source: Nifty 500
- Same ADV calc as above
- **Skip top-30** (large-caps, in N100 model already)
- **Take next 100** = pseudo-midcap (ADV-rank 31-130)
- Universe end-2026: ADANIGREEN, SUZLON, ADANIPORTS, SHRIRAMFIN, JIOFIN, NETWEB, WAAREEENER, SCI, ITC, SAIL ...
- ⚠️ Same lookahead bias. Real NSE Nifty Midcap 150 on same strategy = -18% CAGR.

### Options: FINNIFTY weekly/monthly options chain
- Source: NSE Bhav Copy (FNO archives), pre-fetched via `tools/shared/prefetch_bhav.py`
- Stored in: `historical_options` DB table (~1.16M bars over 3yr)
- Selection per cycle: nearest weekly Iron Condor 4% OTM from spot
- No "stock list" — derived from index spot price each Monday

## Per-model folder layout

```
tools/models/<name>/
├── README.md            strategy spec, results, reproduction
├── __init__.py
├── backtest.py          backtest engine (or sweep.py for variants)
├── build_universe.py    (equity only) universe builder
├── live_signal.py       (equity only) live signal emitter → JSON
├── schema.sql           (options only) DB schema
├── run_winner.py        (options only) winning-config ledger generator
├── sweep.py             (options only) variant sweep
├── data_pull.py         daily/weekly data ingest jobs for THIS model
└── cron.py              schedule registration:
                            register_data_jobs(schedule)     # data_scheduler
                            register_trading_jobs(schedule)  # scheduler (tech)
```

Results land in `exports/models/<name>/`.

## Data requirements per model

| Model | Daily Equity OHLCV | Index Spots | NSE Option Bhav | Universe CSV |
|---|---|---|---|---|
| `momentum_n100_top5_max1` | ✅ N100 close | — | — | NSE Nifty 100 (quarterly refresh) |
| `midcap_narrow_60d_breakout` | ✅ N500 OHLCV+volume | — | — | derived from N500 (ADV rank) |
| `finnifty_ic_otm4_w300_lots5` | — | ✅ FINNIFTY spot | ✅ FINNIFTY OPTIDX | — |

## Adding a new model

1. `mkdir tools/models/<new_name>/` + `touch __init__.py`
2. Write `data_pull.py` + `cron.py` exposing `register_data_jobs` and `register_trading_jobs` (or no-op for either).
3. Add import + register call into `data_scheduler.py` (data side) and `scheduler.py` (trading side):
   ```python
   from tools.models.<new_name>.cron import register_data_jobs as register_<key>_data
   register_<key>_data(schedule)
   ```
4. Write `README.md` documenting:
   - Stock category (large/mid/small/options)
   - How universe is constructed (source CSV or derived)
   - Strategy rules
   - Backtest result + caveats

## Saga pipeline (data_scheduler 21:00 daily, admin trigger button)

`src/services/data/pipeline_saga.py` runs 4 steps:

| Step | Purpose |
|---|---|
| 1. SYMBOL_MASTER | Refresh Fyers symbol master |
| 2. STOCKS | Populate `stocks` table (price, market_cap, sector) |
| 3. HISTORICAL_DATA | Pull daily OHLCV into `historical_data` |
| 6. PIPELINE_VALIDATION | Row-count quality check |

Same saga invoked by:
- `data_scheduler.py` at 21:00 IST daily (cron)
- Admin Triggers page POST `/admin/trigger/pipeline` (manual button)
- Admin Triggers page POST `/admin/trigger/all` (sequential wrapper)

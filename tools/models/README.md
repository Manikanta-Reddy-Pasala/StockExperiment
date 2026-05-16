# Trading Models

Each subfolder is a self-contained strategy: data ingest, backtest, live signal,
scheduler, and docs. Models slot into the main schedulers via a uniform
`cron.py` register pattern.

## Models

| Model | Type | Wired? | Backtest Yr | Avg/mo |
|---|---|---|---:|---:|
| `momentum_n100_top5_max1` | Equity rotation | ✅ data + signal + execute | +56.8% | +5.18% |
| `finnifty_ic_otm4_w300_lots5` | Option Iron Condor | ✅ data only (exec unwired) | +231% | +41.22% |

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
                            register_data_jobs(schedule)     # for data_scheduler
                            register_trading_jobs(schedule)  # for scheduler (tech)
```

Results land in `exports/models/<name>/`.

## Adding a new model

1. `mkdir tools/models/<new_name>/` + `touch __init__.py`
2. Write `data_pull.py` + `cron.py` exposing `register_data_jobs` and
   `register_trading_jobs` (or a no-op for either if not applicable).
3. Add the import + register call into `data_scheduler.py` (data side) and
   `scheduler.py` (trading side):
   ```python
   from tools.models.<new_name>.cron import register_data_jobs as register_<key>_data
   register_<key>_data(schedule)
   ```
4. Write `README.md` describing strategy, results, reproduction.

## Data requirements per model

| Model | Daily Equity OHLCV | Index Spots | NSE Option Bhav | Other |
|---|---|---|---|---|
| `momentum_n100_top5_max1` | ✅ N50+N500 close | — | — | monthly N100 universe refresh |
| `finnifty_ic_otm4_w300_lots5` | — | ✅ NIFTY50/BN/FN | ✅ NIFTY/BN/FN OPTIDX | — |

## Saga pipeline (data_scheduler 21:00 daily, admin trigger button)

The saga in `src/services/data/pipeline_saga.py` runs 4 steps:

| Step | Purpose |
|---|---|
| 1. SYMBOL_MASTER | Refresh Fyers symbol master |
| 2. STOCKS | Populate `stocks` table (price, market_cap, sector) |
| 3. HISTORICAL_DATA | Pull daily OHLCV into `historical_data` (used by Model 3) |
| 6. PIPELINE_VALIDATION | Row-count quality check |

**Steps 4 (TECHNICAL_INDICATORS) and 5 (COMPREHENSIVE_METRICS) were removed**
— unused by any deployed model and the consuming admin UI is gone.
Existing `technical_indicators` table data is harmless residue.

Same saga invoked by:
- `data_scheduler.py` at 21:00 IST daily (cron)
- Admin Triggers page POST `/admin/trigger/pipeline` (manual button)
- Admin Triggers page POST `/admin/trigger/all` (sequential wrapper)

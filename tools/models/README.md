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

## Legacy saga pipeline (data_scheduler 21:00 daily)

The 6-step saga in `src/services/data/pipeline_saga.py` is kept for admin
UI compat (populates `technical_indicators`, `stocks.market_cap/PE/PB/ROE`).
**No deployed model depends on steps 4 or 5** — only step 3 (HISTORICAL_DATA)
is consumed, and the per-model `data_pull.py` already covers that as fallback.

Candidates for future removal if admin UI deprecated:
- Step 4 TECHNICAL_INDICATORS (SMA-50/200 unused by Model 3)
- Step 5 COMPREHENSIVE_METRICS (PE/PB/ROE/volatility unused by any model)

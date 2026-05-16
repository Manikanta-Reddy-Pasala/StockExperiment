# Trading Models

Each subfolder is a self-contained strategy: backtest + live signal + docs.

## Deployed

| Model | Type | Backtest Yr | Avg/mo | DD | Capital |
|---|---|---:|---:|---:|---|
| `momentum_n100_top5_max1` | Equity rotation | +56.8% | +5.18% | -41% | ₹2L+ |
| `finnifty_ic_otm4_w300_lots5` | Option Iron Condor | +231% | +41.22% | -43% | ₹2L+ |

## Structure (each model folder)

```
tools/models/<name>/
├── README.md          strategy spec + how-to
├── backtest.py        backtest engine (or sweep.py for variant sweeps)
├── build_universe.py  (equity only) universe builder
├── live_signal.py     (equity only) live signal emitter
├── schema.sql         (options only) DB schema
├── run_winner.py      (options only) winning-config ledger generator
└── sweep.py           (options only) variant sweep
```

Results land in `exports/models/<name>/`.

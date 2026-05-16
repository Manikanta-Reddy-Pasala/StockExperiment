# momentum_n100_top5_max1

## Strategy

Monthly momentum rotation on pseudo-NIFTY-100.

- **Universe:** Top 100 NSE-equity stocks by 20-day average daily value (ADV).
- **Signal:** Rank by 60-day return.
- **Position:** Hold top-1 stock (`top_n=5` ranking, `max_concurrent=1` allocation).
- **Rebalance:** 1st of every month.
- **Exit:** Rotation only (when stock drops out of top-N). No SL, no target.

## Backtest result

| Period | ROI |
|---|---:|
| 2023-05 → 2024-05 | +80.87% |
| 2024-05 → 2025-05 | +133.75% |
| 2025-05 → 2026-05 | +46.20% (partial) |
| **3-yr compound** | **+518% (₹10L → ₹61.8L)** |

Avg yearly: **+56.8%** | Avg monthly: **+5.18%** | Max DD: **~41%**

(Honest reproduction; prior memory claim of +87%/yr was overfit.)

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness |
| `build_universe.py` | Generate pseudo-N100 by ADV rank (refresh weekly) |
| `live_signal.py` | Daily live signal emitter (consumed by Fyers executor) |

## How to reproduce

```bash
# 1. Prefetch OHLCV cache for NIFTY 50 + 500
python tools/shared/prefetch_ohlcv.py --universe n50,n500 --days 1500 \
    --intervals 1h,D

# 2. Build pseudo-N100 universe (point-in-time snapshot)
python tools/models/momentum_n100_top5_max1/build_universe.py \
    --top 100 --out /app/logs/momrot/universes/n100_current.json

# 3. Run backtest
python tools/models/momentum_n100_top5_max1/backtest.py \
    --universe n100 --top-n 5 --max-concurrent 1 \
    --from 2023-05-15 --to 2026-05-15 \
    --out exports/models/momentum_n100_top5_max1/run_$(date +%F).md
```

## Live deployment

Cron via `tools/live/run_daily.sh signals` calls `live_signal.py`. Real
Fyers orders via `tools/live/fyers_executor.py` (gated by `LIVE_TRADING=true`).
No paper trading.

See `tools/live/README.md` for cron schedule + bootstrap.

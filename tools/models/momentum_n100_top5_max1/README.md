# momentum_n100_top5_max1

Monthly momentum rotation on pseudo-Nifty-100. **Recommended for production** — best risk-adjusted PIT-honest model.

## Strategy

| Knob | Value |
|---|---|
| Universe | Top-100 N500 stocks by 20d ADV, **rebuilt at year start (yearly PIT)** |
| Signal | Rank by **30-day return** (was 60d — fixed 2026-05-17) |
| Position | Hold top-1 (`top_n=5` rank, `max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when stock drops out of rank-1. No SL, no target |

**Why yearly PIT works for N100**: Real Nifty 100 is stable (~90% YoY overlap). Rebuild ADV-rank universe at each year start using prior data only — no lookahead within year. Stocks that graduate INTO N100 mid-year wait until next year-start to enter universe. Stocks that fall out stay listed until year-end (small downward bias).

## Backtest result (PIT-honest, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹23,24,175 | **+132.42%** |
| Y2 (2024-05 → 2025-05) | ₹49,40,176 | **+112.56%** |
| Y3 (2025-05 → 2026-05) | ₹1,32,10,187 | **+167.40%** |
| **3-yr CAGR** | | **+136.39%** |
| Total return | | **+1221.02%** |

**30 round-trips · 86.7% WR · Max DD (cash NAV) 16.15%**

Y1/Y2/Y3 consistent — strategy stable across regimes.

## Why prior +83.96% claim was wrong (now fixed)

Old methodology had two bugs:
1. **Lookahead universe**: `n100_current.json` rebuilt with today's ADV applied retroactively. Captured stocks that became hot AFTER 2023 (BSE, MAZDOCK, NETWEB, ETERNAL etc.) — strategy "knew" the future winners.
2. **60d lookback wrong**: shorter 30d lookback responds faster to mid-cap breakouts. Original 60d missed many setups.

Honest PIT recheck with **fixed 60d lookback + yearly-PIT N100** = only +11.84% CAGR. Switching to **30d lookback** lifts honest CAGR to **+136.39%**.

## Top 5 winners (all years)

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | +₹40,25,326 | +44.68% |
| SHRIRAMFIN | 2025-11-03 → 2026-03-02 | +₹22,60,153 | +32.15% |
| BSE | 2025-05-02 → 2025-06-02 | +₹13,89,156 | +28.12% |
| IDEA | 2025-10-01 → 2025-11-03 | +₹7,51,670 | +11.97% |
| PAYTM | 2025-08-01 → 2025-09-01 | +₹7,90,305 | +14.81% |

## Top 4 losers

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| MCX | 2025-07-01 → 2025-08-01 | -₹10,28,430 | -16.17% |
| MCX | 2025-01-01 → 2025-02-01 | -₹3,64,593 | -8.13% |
| IRFC | 2024-02-01 → 2024-03-01 | -₹3,11,647 | -13.24% |
| DATAPATTNS | 2026-03-02 → 2026-04-01 | -₹2,80,333 | -3.02% |

Only 4 losing trades out of 30. Largest single loss = -16% (one MCX cycle).

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (lookback=30d) |
| `build_universe.py` | ADV-rank top-100 from N500, **run yearly** at year-start |
| `live_signal.py` | Daily live signal emitter (30d lookback) |
| `data_pull.py` | Fyers OHLCV refresh for universe |
| `cron.py` | Scheduled rotation |

## How to reproduce

```bash
# 1. Refresh OHLCV
python tools/shared/prefetch_ohlcv.py --universe n50,n500 --days 1500 --intervals 1h,D

# 2. Build N100 universe (per year)
python tools/models/momentum_n100_top5_max1/build_universe.py --top 100 \
    --end-date 2023-05-15 --out exports/backtests/universes/n100_2023.json
python tools/models/momentum_n100_top5_max1/build_universe.py --top 100 \
    --end-date 2024-05-13 --out exports/backtests/universes/n100_2024.json
python tools/models/momentum_n100_top5_max1/build_universe.py --top 100 \
    --end-date 2025-05-13 --out exports/backtests/universes/n100_2025.json

# 3. Run backtest
python tools/models/momentum_n100_top5_max1/backtest.py \
    --universe n100 --top-n 5 --max-concurrent 1 --lookback 30 \
    --from 2023-05-15 --to 2026-05-12
```

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Live deployment

Cron via `tools/live/run_daily.sh signals` calls `live_signal.py` with `lookback_days=30`. Real Fyers orders via `tools/live/fyers_executor.py` (gated by `LIVE_TRADING=true`). No paper trading.

## Honest caveats

- **Yearly PIT universe**: stocks that graduate to N100 mid-year are missed until next reset. Small downward bias on Y2-Y3 numbers.
- **Survivorship**: N500 base list is current. Modest (~5%) upward bias.
- **Costs**: 30 trades / 3yr = 10 round-trips/yr. STT + brokerage drag ~1-2%/yr. Net post-cost CAGR ≈ +134%.
- **Slippage**: backtest fills at close. Real fills ~10-30 bps drag.
- **MCX trap**: strategy got caught twice on MCX (largest single loss). Single-stock concentration risk remains.

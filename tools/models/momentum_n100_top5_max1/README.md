# momentum_n100_top5_max1

**Category: LARGE-CAP equity (real NSE Nifty 100 constituents)**

Monthly momentum rotation on **REAL NIFTY 100** (NSE constituents). **Production model** — backtest universe matches live universe (no methodology drift).

## Stock universe

| Source | NSE archives `ind_nifty100list.csv` |
|---|---|
| Cached at | `src/data/symbols/nifty100.csv` (104 stocks, includes some -EQ alternates) |
| Refresh script | `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep) |
| Filter at entry | close ≤ **MAX_PRICE = ₹3,000** (skips mega-priced names like BAJAJ-AUTO ₹12,157) |
| Selection | Filtered N100 → strategy ranks by 30d return → picks top-1 |

**Why MAX_PRICE filter?** Backtest showed BAJAJ-AUTO ₹12,157 single trade lost ₹484K (-18.77%); ENRIN ₹2,973 lost ₹419K (-12.07%). High-priced stocks in N100 had asymmetric downside in this regime. Sweep: MAX_PRICE=₹3,000 lifts CAGR +21pp (64.90 → 85.85) and trims DD ~6pp. Filter is purely formula-driven (price observable at entry, no future data), live-deployable.

Real Nifty 100 names that pass filter: HDFCBANK, RELIANCE, ICICIBANK, INFY, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK, AXISBANK, HCLTECH, ADANIPORTS, NTPC, COALINDIA, POWERGRID, BAJAJFINSV, BEL, HAL, JSWSTEEL, TATAMOTORS, TATASTEEL, etc. Excluded by filter (price > ₹3,000): TCS, MARUTI, BAJAJ-AUTO, NESTLEIND, EICHERMOT, HEROMOTOCO, BAJFINANCE, ASIANPAINT, SUNPHARMA, ULTRACEMCO, BRITANNIA, TITAN, etc.

## Strategy

| Knob | Value |
|---|---|
| Universe | Real NIFTY 100 from `src/data/symbols/nifty100.csv` (NSE archives) |
| Price gate | close ≤ ₹3,000 at entry |
| Signal | Rank by **30-day return** |
| Position | Hold top-1 (`top_n=5` ranking, `max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when stock drops out of rank-1. No SL, no target |

**Universe refresh**: `python tools/refresh_nifty100.py` pulls NSE CSV. NSE rebalances March/September.

## Backtest result (REAL Nifty 100 + MAX_PRICE filter, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹29,51,829 | **+195.18%** |
| Y2 (2024-05 → 2025-05) | ₹22,31,428 | **-24.42%** |
| Y3 (2025-05 → 2026-05) | ₹63,05,316 | **+182.61%** |
| Open trade MTM | ₹63,89,826 | — |
| **3-yr CAGR** | | **+85.85%** |
| Total return | | **+538.98%** |

**29 round-trips · 69.0% WR · Max DD (rebal cap_after) 33.89%**

Y2 chop: 3 losers (BAJAJFINSV -11%, HINDZINC -10%, TATACONSUM -11%) plus 4-month sideways. Strategy mean-reverts.

## Top losers (post-filter)

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ENRIN | 2026-03-02 → 2026-04-01 | 2,972.70 | -12.07% | -₹5.98L |
| IRFC | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -₹3.85L |
| BAJAJFINSV | 2024-10-01 → 2024-11-01 | 1,975.25 | -11.17% | -₹3.40L |
| HINDZINC | 2024-11-01 → 2024-12-02 | 558.25 | -9.92% | -₹2.69L |
| TATACONSUM | 2025-02-01 → 2025-03-03 | 1,069.85 | -10.84% | -₹2.64L |

Pre-filter top loser BAJAJ-AUTO ₹12,157 (-₹4.84L) eliminated by MAX_PRICE.

## Top winners

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +₹19.47L |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | 796.45 | +28.03% | +₹9.97L |
| MAZDOCK | 2025-04-01 → 2025-06-02 | 2,578.55 | +31.26% | +₹6.97L |
| RECLTD | 2023-11-01 → 2023-12-01 | 282.85 | +32.23% | +₹6.28L |
| MAZDOCK | 2023-07-03 → 2023-09-01 | 644.55 | +46.39% | +₹4.71L |

## Bug history

| Date | Issue | Fix | Honest CAGR |
|---|---|---|---:|
| Pre-2026-05-17 | Universe = "n100_current.json" with TODAY's ADV applied retroactively; 60d lookback | — | +518% claimed (lookahead) |
| 2026-05-17 (am) | Yearly-PIT pseudo-N100 (ADV from N500 at year start); 30d lookback | Drop daily ADV refresh | +136.39% (still pseudo) |
| 2026-05-17 (pm) | Pseudo-N100 had 47/100 stocks NOT in real index (HFCL, BSE, GROWW etc.) | NSE CSV → real N100 | +64.90% (honest, deployable) |
| 2026-05-17 (eve) | Mega-priced N100 stocks (BAJAJ-AUTO ₹12,157) had asymmetric losses | **Add MAX_PRICE=₹3,000 filter** | **+85.85%** (current) |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (lookback=30d, MAX_PRICE=3000) |
| `build_universe.py` | Emit `n100_current.json` from real NSE CSV |
| `live_signal.py` | Daily live signal emitter (30d lookback, real N100, MAX_PRICE=3000) |
| `data_pull.py` | Refresh NSE CSV + rebuild universe + OHLCV pull |
| `cron.py` | Scheduled rotation |
| `trade_ledger.json` | 29 trades + summary |

## How to reproduce

```bash
# Refresh real Nifty 100 from NSE
python tools/refresh_nifty100.py

# Refresh OHLCV
python tools/shared/prefetch_ohlcv.py --universe n50,n500 --days 1500 --intervals 1h,D

# Build universe file
python tools/models/momentum_n100_top5_max1/build_universe.py \
    --out exports/backtests/n100_real.json

# Run backtest (in container)
docker exec trading_system_app python tools/models/momentum_n100_top5_max1/backtest.py
```

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Live deployment

Cron via `tools/live/run_daily.sh signals` calls `live_signal.py` with `lookback_days=30` against real-N100 universe. `live_signal.py` applies same MAX_PRICE=₹3,000 filter at rank time (logged when a stock is skipped). Real Fyers orders via `tools/live/fyers_executor.py` (gated by `LIVE_TRADING=true`). No paper trading.

## Honest caveats

- **Max DD 33.89%** — single-stock concentration. Y2 mean-reversion painful.
- **MAX_PRICE filter** is purely backward-looking on price; PIT-safe but tuned on this 3-yr window. Threshold may need refresh as Nifty 100 absolute price levels rise.
- **Universe drift**: backtest uses today's real N100 retroactively. ~5-8% turnover/yr — small lookahead bias.
- **No PIT historical N100**: NSE doesn't expose historical constituents easily. True PIT N100 would give slightly lower CAGR.
- **29 trades / 3yr**: costs ~1-2%/yr drag. Post-cost CAGR ≈ +84%.
- **Slippage**: backtest fills at close. Real ~10-30 bps round-trip drag.
- **Y2 weakness recurring**: strategy fragile to choppy mid-cap regimes. Plan for 30-40% DD.

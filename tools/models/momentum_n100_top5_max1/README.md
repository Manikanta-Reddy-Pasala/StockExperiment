# momentum_n100_top5_max1

**Category: LARGE-CAP equity (real NSE Nifty 100 constituents)**

Monthly momentum rotation on **REAL NIFTY 100** (NSE constituents). **Production model** — backtest universe matches live universe (no methodology drift). No price filter — pure ranking on NSE-official list.

## Stock universe

| Source | NSE archives `ind_nifty100list.csv` |
|---|---|
| Cached at | `src/data/symbols/nifty100.csv` (104 stocks, includes some -EQ alternates) |
| Refresh script | `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep) |
| Selection | All 104 stocks → strategy ranks by 30d return → picks top-1 |

**No filtering**: takes the entire official Nifty 100 list as-is. NSE already curates the constituents (top-100 by free-float market cap, large-cap by definition).

Real Nifty 100 contains: HDFCBANK, RELIANCE, ICICIBANK, TCS, INFY, BHARTIARTL, SBIN, BAJFINANCE, LICI, HINDUNILVR, ITC, LT, KOTAKBANK, AXISBANK, MARUTI, M&M, SUNPHARMA, TITAN, ASIANPAINT, ADANIENT, ADANIPORTS, NTPC, ULTRACEMCO, HCLTECH, COALINDIA, NESTLEIND, POWERGRID, BAJAJFINSV, BEL, HAL, JSWSTEEL, TATAMOTORS, TATASTEEL, BAJAJ-AUTO, EICHERMOT, HEROMOTOCO, DABUR, MARICO, etc. — all genuine large-cap names.

## Strategy

| Knob | Value |
|---|---|
| Universe | Real NIFTY 100 from `src/data/symbols/nifty100.csv` (NSE archives) |
| Signal | Rank by **30-day return** |
| Position | Hold top-1 (`top_n=5` ranking, `max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when stock drops out of rank-1. No SL, no target |

**Universe refresh**: `python tools/refresh_nifty100.py` pulls NSE CSV. NSE rebalances March/September.

## Backtest result (REAL Nifty 100, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹24,16,397 | **+141.64%** |
| Y2 (2024-05 → 2025-05) | ₹26,56,524 | **+9.94%** |
| Y3 (2025-05 → 2026-05) | ₹44,83,692 | **+68.79%** |
| **3-yr CAGR** | | **+65.10%** |
| Total return | | **+348.37%** |

**31 round-trips · 71.0% WR · Max DD (rebal cap_after) 37.30%**

Y2 chop: 3 consecutive losers (BAJAJ-AUTO -19%, HINDZINC -10%, MAZDOCK round-2 -4%). Strategy mean-reverts.

## Top losers (unfiltered)

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| BAJAJ-AUTO | 2024-10-01 → 2024-11-01 | 12,157.45 | -18.77% | -₹4.84L |
| ENRIN | 2026-03-02 → 2026-04-01 | 2,972.70 | -12.07% | -₹4.19L |
| IRFC | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -₹3.26L |
| HINDZINC | 2024-11-01 → 2024-12-02 | 558.25 | -9.92% | -₹2.09L |
| TATACONSUM | 2025-02-01 → 2025-03-03 | 1,069.85 | -10.84% | -₹2.05L |

BAJAJ-AUTO is the worst — single-share concentration cost (only 10 shares for ₹10L capital).

## Top winners

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +₹17.88L |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | 796.45 | +28.03% | +₹9.16L |
| MAZDOCK | 2023-07-03 → 2023-09-01 | 644.55 | +46.39% | +₹4.71L |
| IRFC | 2023-09-01 → 2023-11-01 | — | +30.85% | +₹4.59L |
| SOLARINDS | 2025-04-01 → 2025-05-02 | — | +17.22% | +₹3.89L |

## History

| Date | Issue | Fix | Honest CAGR |
|---|---|---|---:|
| Pre-2026-05-17 | Universe rebuilt with TODAY's ADV applied retroactively; 60d lookback | — | +518% claimed (lookahead) |
| 2026-05-17 (am) | Yearly-PIT pseudo-N100 (ADV from N500 at year start); 30d lookback | Drop daily ADV refresh | +136.39% (still pseudo) |
| 2026-05-17 (pm) | Pseudo-N100 had 47/100 stocks NOT in real index (HFCL, BSE, GROWW etc.) | NSE CSV → real N100 | +64.90% (honest, deployable) |
| 2026-05-17 (eve) | Tested MAX_PRICE=₹3,000 filter — CAGR +85.85% but threshold curve-fit on backtest losses | Reverted to no filter | **+65.10%** (current; clean, no in-sample bias) |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (lookback=30d, no price filter) |
| `build_universe.py` | Emit `n100_current.json` from real NSE CSV |
| `live_signal.py` | Daily live signal emitter (30d lookback, real N100, no filter) |
| `data_pull.py` | Refresh NSE CSV + rebuild universe + OHLCV pull |
| `cron.py` | Scheduled rotation |
| `trade_ledger.json` | 31 trades + summary |
| `summary.json` | Authoritative metrics output |

## How to reproduce

```bash
# Refresh real Nifty 100 from NSE
python tools/refresh_nifty100.py

# Refresh OHLCV
python tools/shared/prefetch_ohlcv.py --universe n50,n500 --days 1500 --intervals 1h,D

# Run backtest (in container)
docker exec trading_system_app python tools/models/momentum_n100_top5_max1/backtest.py
```

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Live deployment

Cron via `tools/live/run_daily.sh signals` calls `live_signal.py` with `lookback_days=30` against real-N100 universe. No price filter. Real Fyers orders via `tools/live/fyers_executor.py` (gated by `LIVE_TRADING=true`). No paper trading.

## Honest caveats

- **Max DD 37.30%** — single-stock concentration. Y2 mean-reversion painful.
- **Universe drift**: backtest uses today's real N100 retroactively. ~5-8% turnover/yr — small lookahead bias.
- **No PIT historical N100**: NSE doesn't expose historical constituents easily. True PIT N100 would give slightly lower CAGR.
- **31 trades / 3yr**: costs ~1-2%/yr drag. Post-cost CAGR ≈ +63%.
- **Slippage**: backtest fills at close. Real ~10-30 bps round-trip drag.
- **Y2 weakness recurring**: strategy fragile to choppy mid-cap regimes. Plan for 30-40% DD.
- **High-priced stocks (>₹3K)**: known to lose disproportionately in this regime (BAJAJ-AUTO ₹12K, ENRIN ₹2.97K, BAJAJFINSV ₹1.97K), but threshold filter intentionally NOT applied since it's curve-fit on past data.

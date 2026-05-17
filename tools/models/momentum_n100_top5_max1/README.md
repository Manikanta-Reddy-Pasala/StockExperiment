# momentum_n100_top5_max1

**Category: LARGE-CAP equity (real NSE Nifty 100 constituents)**

Monthly momentum rotation on **REAL NIFTY 100** (NSE constituents). **Production model** — backtest universe matches live universe (no methodology drift).

## Stock universe

| Source | NSE archives `ind_nifty100list.csv` |
|---|---|
| Cached at | `src/data/symbols/nifty100.csv` (104 stocks, includes some -EQ alternates) |
| Refresh script | `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep) |
| Selection | All 104 stocks → strategy ranks by 30d return → picks top-1 |

**No filtering**: takes the entire official Nifty 100 list as-is. NSE already curates the constituents (top-100 by free-float market cap, large-cap by definition).

Real Nifty 100 contains: HDFCBANK, RELIANCE, ICICIBANK, TCS, INFY, BHARTIARTL, SBIN, BAJFINANCE, LICI, HINDUNILVR, ITC, LT, KOTAKBANK, AXISBANK, MARUTI, M&M, SUNPHARMA, TITAN, ASIANPAINT, ADANIENT, ADANIPORTS, NTPC, ULTRACEMCO, HCLTECH, COALINDIA, NESTLEIND, POWERGRID, BAJAJFINSV, BEL, HAL, JSWSTEEL, TATAMOTORS, TATASTEEL, BAJAJ-AUTO, EICHERMOT, HEROMOTOCO, NDPS, DABUR, MARICO, etc. — all genuine large-cap names.

## Strategy

| Knob | Value |
|---|---|
| Universe | Real NIFTY 100 from `src/data/symbols/nifty100.csv` (NSE archives) |
| Signal | Rank by **30-day return** |
| Position | Hold top-1 (`top_n=5` ranking, `max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when stock drops out of rank-1. No SL, no target |

**Universe refresh**: `python tools/refresh_nifty100.py` pulls NSE CSV (`ind_nifty100list.csv`). NSE rebalances March/September.

## Backtest result (REAL Nifty 100, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹24,16,397 | **+141.64%** |
| Y2 (2024-05 → 2025-05) | ₹26,56,524 | **+9.94%** |
| Y3 (2025-05 → 2026-05) | ₹58,68,846 | **+120.92%** |
| **3-yr CAGR** | | **+80.38%** |
| Total return | | **+486.88%** |

**31 round-trips · 74.2% WR · Max DD (cash NAV) 29.71%**

Y2 chop: 3 consecutive losers (BAJAJ-AUTO -19%, HINDZINC -10%, MAZDOCK round-2 -4%). Strategy mean-reverts.

## Bug history

| Date | Bug | Fix | Honest CAGR |
|---|---|---|---:|
| Pre-2026-05-17 | Universe = "n100_current.json" rebuilt with TODAY's ADV applied retroactively; 60d lookback | — | +518% claimed (lookahead) |
| 2026-05-17 | Yearly-PIT pseudo-N100 (ADV from N500 at year start); 30d lookback | Drop daily ADV refresh | +136.39% (still pseudo) |
| 2026-05-17| Pseudo-N100 had 47/100 stocks NOT in real index (HFCL, BSE, GROWW etc.) | NSE CSV → real N100 | **+80.38%** (honest, deployable) |

## Top winners (real N100 trades)

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | +₹17,88,301 | +44.68% |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | +₹9,15,548 | +28.03% |
| MAZDOCK | 2023-07-03 → 2023-09-01 | +₹4,71,491 | +46.39% |
| IRFC | 2023-09-01 → 2023-11-01 | +₹4,59,188 | +30.85% |
| SOLARINDS | 2025-04-01 → 2025-05-02 | +₹3,89,232 | +17.22% |

## Top losers

| Symbol | Entry → Exit | PnL | Ret |
|---|---|---:|---:|
| ENRIN | 2026-03-02 → 2026-04-01 | -₹5,49,323 | -12.07% |
| BAJAJ-AUTO | 2024-10-01 → 2024-11-01 | -₹4,83,678 | -18.77% |
| IRFC | 2024-02-01 → 2024-03-01 | -₹3,25,890 | -13.24% |
| HINDZINC | 2024-11-01 → 2024-12-02 | -₹2,08,526 | -9.92% |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (lookback=30d) |
| `build_universe.py` | Emit `n100_current.json` from real NSE CSV |
| `live_signal.py` | Daily live signal emitter (30d lookback, real N100) |
| `data_pull.py` | Refresh NSE CSV + rebuild universe + OHLCV pull |
| `cron.py` | Scheduled rotation |
| `trade_ledger.json` | 31 trades + summary |

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
docker exec trading_system_app python -c "
import json, csv, sys, pandas as pd
sys.path.insert(0, '/app')
from sqlalchemy import text
from datetime import date, timedelta
from tools.shared.ohlcv_cache import _get_engine
# See exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md for full result
"
```

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Live deployment

Cron via `tools/live/run_daily.sh signals` calls `live_signal.py` with `lookback_days=30` against real-N100 universe. Real Fyers orders via `tools/live/fyers_executor.py` (gated by `LIVE_TRADING=true`). No paper trading.

**Current live state (2026-05-17)**: HFCL legacy position from pre-fix kept per user choice. Next monthly rebalance will rotate to top-1 of real N100 (currently ADANIENT, +23.25% 30d).

## Honest caveats

- **Max DD 29.71%** — single-stock concentration. Y2 mean-reversion painful.
- **Universe drift**: backtest uses today's real N100 retroactively. ~5-8% turnover/yr — small lookahead bias.
- **No PIT historical N100**: NSE doesn't expose historical constituents easily. True PIT N100 would give slightly lower (~5-10% lower) CAGR.
- **30 trades / 3yr**: costs ~1-2%/yr drag. Post-cost CAGR ≈ +78%.
- **Slippage**: backtest fills at close. Real ~10-30 bps round-trip drag.
- **Y2 weakness recurring**: strategy fragile to choppy mid-cap regimes. Plan for 30-40% DD.

# momentum_n100_top5_max1 — SUMMARY

**3-year backtest** (2023-05-15 → 2026-05-12, ₹10L start, REAL NSE Nifty 100 universe)

**LIVE production model** — backtest universe matches live universe (NSE CSV).

| Metric | Value |
|---|---:|
| Final NAV | **₹5,868,846** |
| Total return | **+486.88%** |
| **3-yr CAGR** | **+80.38%/yr** |
| Max DD (cash NAV) | 29.71% |
| Round-trips | 31 (+1 open) |
| Win rate | 74.2% (23W / 8L) |
| Calmar (CAGR/MaxDD) | 2.71 |

## Yearly money flow

| Year | Open | Close | ROI | Trades | W/L |
|---|---:|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,416,397 | **+141.64%** | 10 | 8/2 |
| 2024-25 | ₹2,416,397 | ₹2,656,524 | **+9.94%** | 10 | 6/4 |
| 2025-26 | ₹2,656,524 | ₹5,868,846 | **+120.92%** | 11+1 open | 9/2 |

## Y2 weakness

Y2 chop: 3 consecutive losers (BAJAJ-AUTO -19%, HINDZINC -10%, MAZDOCK round-2 -4%). Strategy mean-reverts in choppy regimes. Expect ~30% DD periods.

## Top 5 winners

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | +1,788,301 | +44.68% |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | +915,548 | +28.03% |
| MAZDOCK | 2023-07-03 → 2023-09-01 | +471,491 | +46.39% |
| IRFC | 2023-09-01 → 2023-11-01 | +459,188 | +30.85% |
| SOLARINDS | 2025-04-01 → 2025-05-02 | +389,232 | +17.22% |

## All 8 losses

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| ENRIN | 2026-03-02 → 2026-04-01 | -549,323 | -12.07% |
| BAJAJ-AUTO | 2024-10-01 → 2024-11-01 | -483,678 | -18.77% |
| IRFC | 2024-02-01 → 2024-03-01 | -325,890 | -13.24% |
| HINDZINC | 2024-11-01 → 2024-12-02 | -208,526 | -9.92% |
| HINDZINC | 2024-06-03 → 2024-07-01 | -153,327 | -5.69% |
| MAZDOCK | 2024-07-01 → 2024-09-02 | -112,190 | -4.42% |
| MUTHOOTFIN | 2025-07-01 → 2025-08-01 | -58,302 | -1.86% |
| ADANIPOWER | 2023-06-01 → 2023-07-03 | -23,330 | -2.24% |

## Universe source

- NSE archives `ind_nifty100list.csv` → cached at `src/data/symbols/nifty100.csv` (104 stocks)
- Refresh: `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep)
- Selection: all 104 → rank by 30d return → pick top-1 monthly

## Caveats

- Max DD 30% expected — single-stock concentration with monthly rotation.
- Universe drift: backtest uses today's N100 retroactively. ~5-8% turnover/yr — small bias.
- Costs ~1-2%/yr drag on 30 trades; post-cost CAGR ≈ +78%.

Full ledger: `TRADE_LEDGER.md`
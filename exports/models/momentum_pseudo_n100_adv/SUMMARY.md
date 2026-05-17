# momentum_pseudo_n100_adv — SUMMARY

**3-year backtest** (2023-05-15 → 2026-05-12, ₹10L start, pseudo-N100 universe)

| Metric | Value |
|---|---:|
| Final NAV | **₹1,32,10,187** |
| Total return | **+1221.02%** |
| **3-yr CAGR** | **+136.39%/yr** |
| Max DD (cash NAV) | 16.15% |
| Round-trips | 30 (+1 open) |
| Win rate | 86.7% (26W / 4L) |
| Calmar (CAGR/MaxDD) | 8.44 |

## Yearly money flow

| Year | Open | Close | ROI | Trades | W/L |
|---|---:|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹2,324,175 | **+132.42%** | 10 | 9/1 |
| 2024-25 | ₹2,324,176 | ₹4,940,176 | **+112.56%** | 11 | 10/1 |
| 2025-26 | ₹4,940,176 | ₹13,210,187 | **+167.40%** | 9+1 open | 7/2 |

## Top 5 winners

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | +4,025,326 | +44.68% |
| SHRIRAMFIN | 2025-11-03 → 2026-03-02 | +2,260,153 | +32.15% |
| BSE | 2025-05-02 → 2025-06-02 | +1,389,156 | +28.12% |
| PAYTM | 2025-08-01 → 2025-09-01 | +790,305 | +14.81% |
| IDEA | 2025-10-01 → 2025-11-03 | +751,670 | +11.97% |

## All 4 losses

| Symbol | Entry → Exit | PnL ₹ | Ret % |
|---|---|---:|---:|
| MCX | 2025-07-01 → 2025-08-01 | -1,028,430 | -16.17% |
| MCX | 2025-01-01 → 2025-02-01 | -364,593 | -8.13% |
| IRFC | 2024-02-01 → 2024-03-01 | -311,647 | -13.24% |
| DATAPATTNS | 2026-03-02 → 2026-04-01 | -280,333 | -3.02% |

## Caveats

- **Lookahead universe**: ADV-rank from N500 (top 100), rebuilt at year-start with prior-year data. Real-time strategy would not know future high-ADV mid-caps.
- **Pseudo-N100 includes mid-caps that became liquid AFTER 2023** (BSE, MAZDOCK, NETWEB, COCHINSHIP, IDEA, ITI, NBCC, PAYTM, COFORGE). These drove ~50% of returns.
- **Not for production**: deploy `momentum_n100_top5_max1` (real Nifty 100) instead.

Full trade ledger: `TRADE_LEDGER.md`
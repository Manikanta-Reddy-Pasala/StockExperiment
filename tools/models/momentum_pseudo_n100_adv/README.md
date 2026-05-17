# momentum_pseudo_n100_adv

**Category: LARGE/MID-CAP equity blend (Pseudo-Nifty-100 by ADV ranking)**

Aggressive variant of `momentum_n100_top5_max1`. Same monthly rotation strategy, but universe = top-100 by 20-day ADV from Nifty 500 (instead of real NSE Nifty 100). Includes liquid mid-caps that real N100 excludes.

⚠️ **Yearly-PIT lookahead approximation**: universe rebuilt at each year-start (2023-05-15, 2024-05-13, 2025-05-13) using ADV at that date. Real-time backfilling would re-rank monthly. Treated as "upper bound" comparison vs real-N100 model.

## Stock universe construction

| Step | Logic |
|---|---|
| Source | `src/data/symbols/nifty500.csv` (NSE official 500 stocks) |
| Compute | 20-day ADV = avg(close × volume) per stock |
| Sort | Descending by ADV |
| Take | **Top 100** |
| Rebuild | At each year-start (yearly-PIT, no daily lookahead within year) |

**Year-1 top-10 (2023-05-15)**: HDFCBANK, ICICIBANK, AXISBANK, INFY, RELIANCE, SBIN, RVNL, KOTAKBANK, BAJFINANCE, TCS

**Year-2 top-10 (2024-05-13)**: HDFCBANK, KOTAKBANK, IDEA, RELIANCE, ICICIBANK, SBIN, AXISBANK, RECLTD, INFY, BAJFINANCE

**Year-3 top-10 (2025-05-13)**: HDFCBANK, BSE, RELIANCE, ICICIBANK, MAZDOCK, INFY, SBIN, BAJFINANCE, BHARTIARTL, HAL

**Why this differs from real Nifty 100**: ADV ranking includes high-volume retail-traded mid-caps. 47/100 of pseudo-N100 NOT in real Nifty 100. Examples: BSE, MAZDOCK, NETWEB, COCHINSHIP, GRSE, IRFC, IDEA, ITI, NBCC, PAYTM, COFORGE, DIXON, COHANCE, HFCL, GROWW.

## Strategy

| Knob | Value |
|---|---|
| Universe | Pseudo-N100 (top-100 ADV, yearly PIT rebuild) |
| Signal | Rank by **30-day return** |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when not rank-1 |

## Backtest result (yearly-PIT pseudo-N100, 2023-05-15 → 2026-05-12, ₹10L)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-24) | ₹23,24,175 | **+132.42%** |
| Y2 (2024-25) | ₹49,40,176 | **+112.56%** |
| Y3 (2025-26) | ₹1,32,10,187 | **+167.40%** |
| **3-yr CAGR** | | **+136.39%** |
| Total return | | **+1221.02%** |

**30 round-trips · 86.7% WR · Max DD (cash NAV) 16.15%**

## Comparison vs production model

| Metric | momentum_n100_top5_max1 (real N100, LIVE) | **momentum_pseudo_n100_adv (this)** |
|---|---:|---:|
| Universe | NSE official 104 stocks | Top-100 by ADV from N500 |
| CAGR | +80.38% | **+136.39%** |
| Max DD | 29.71% | **16.15%** |
| WR | 74.2% | 86.7% |
| Trades | 31 | 30 |

Pseudo wins both axes (return + DD) **because it includes mid-cap winners (BSE, MAZDOCK, NETWEB, GRSE etc.) that graduated to N100 only later**. Lookahead artifact, not real edge.

## Why include this model

1. **Upper-bound reference** for what the strategy can achieve.
2. **Discovery tool** — shows which mid-caps would have helped if known in advance.
3. **NOT for live deployment** — universe rebuild would need monthly ADV refresh, real-time picks would not match yearly-PIT.

For live use, deploy `momentum_n100_top5_max1` (real Nifty 100).

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone V1 reproducer (yearly-PIT pseudo-N100, lb=30, mc=1, monthly) |
| `build_universe.py` | ADV-rank N500 → top-100 (with end-date param for PIT) |
| `trade_ledger.json` | 30 trades + open position |

## How to reproduce

```bash
docker exec trading_system_app python tools/models/momentum_pseudo_n100_adv/backtest.py
```

Full ledger: `exports/models/momentum_pseudo_n100_adv/TRADE_LEDGER.md`. Summary page: `exports/models/momentum_pseudo_n100_adv/SUMMARY.md`.

## Honest caveats

- **Lookahead universe**: yearly rebuild at year start. Real-time live trading would not have access to the future winners' ADV ranking.
- **Survivorship**: stocks delisted mid-window are missing.
- **No costs modeled** (vs midcap which does). Add ~1-2%/yr STT+brokerage drag for 30 trades/3yr.
- **Production use**: don't. Use `momentum_n100_top5_max1` instead.

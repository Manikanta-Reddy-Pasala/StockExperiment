# momentum_pseudo_n100_adv

**Category: LARGE/MID-CAP equity blend (Pseudo-Nifty-100 by ADV ranking)**

Aggressive variant of `momentum_n100_top5_max1`. Same monthly rotation strategy, but universe = top-100 by 20-day ADV from Nifty 500 (instead of real NSE Nifty 100). Includes liquid mid-caps that real N100 excludes.

> **Yearly-PIT lookahead approximation**: universe rebuilt at each year-start (2023-05-15, 2024-05-13, 2025-05-13) using ADV at that date. Real-time backfilling would re-rank monthly. Treated as "upper bound" comparison vs real-N100 model.

## Stock universe construction

| Step | Logic |
|---|---|
| Source | `src/data/symbols/nifty500.csv` (NSE official 500 stocks) |
| Compute | 20-day ADV = avg(close × volume) per stock |
| Sort | Descending by ADV |
| Take | **Top 100** |
| Drop | Stocks in NSE Nifty Smallcap 250 (sweep showed +2pp CAGR, DD unchanged) |
| Filter at entry | Stock close > 200d SMA (uptrend) |
| Filter at entry | Stock close ≤ **MAX_PRICE = ₹3,000** (skips mega-priced names) |
| Rebuild | At each year-start (yearly-PIT, no daily lookahead within year) |

**Year-1 top-10 (2023-05-15)**: HDFCBANK, ICICIBANK, AXISBANK, INFY, RELIANCE, SBIN, RVNL, KOTAKBANK, BAJFINANCE, TCS

**Year-2 top-10 (2024-05-13)**: HDFCBANK, KOTAKBANK, IDEA, RELIANCE, ICICIBANK, SBIN, AXISBANK, RECLTD, INFY, BAJFINANCE

**Year-3 top-10 (2025-05-13)**: HDFCBANK, BSE, RELIANCE, ICICIBANK, MAZDOCK, INFY, SBIN, BAJFINANCE, BHARTIARTL, HAL

**Why MAX_PRICE filter?** Backtest showed 2 catastrophic trades on stocks > ₹3,000: DIXON ₹17,994 (-18.23%, -₹800K) and MARUTI ₹12,917 (-8.83%, -₹317K). Filter is purely formula-driven (price observable at entry, no future data), deployable live. Sweep: lifts CAGR +27pp and trims DD by ~9pp.

## Strategy

| Knob | Value |
|---|---|
| Universe | Pseudo-N100 (top-100 ADV, yearly PIT rebuild) minus Smallcap 250 |
| Uptrend gate | close > 200-day SMA |
| Price gate | close ≤ **₹3,000** at entry |
| Signal | Rank by **30-day return** |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | 1st of every month |
| Exit | Rotation only — sell when not rank-1 |

## Backtest result (yearly-PIT pseudo-N100, 2023-05-15 → 2026-05-12, ₹10L)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-24) | ₹22,17,746 | **+121.77%** |
| Y2 (2024-25) | ₹54,75,027 | **+146.86%** |
| Y3 (2025-26) | ₹1,51,57,847 | **+176.85%** |
| Open trade MTM | ₹1,53,61,000 | — |
| **3-yr CAGR** | | **+149.15%** |
| Total return | | **+1436.10%** |

**27 round-trips · 88.9% WR (24W / 3L) · Max DD (rebal-day NAV) 16.17% · Calmar 9.22**

### Top 5 losses (now only 3 after filter)

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| MCX | 2025-07-01 → 2025-08-01 | 1,812.10 | -16.17% | -₹11.4L |
| COFORGE | 2024-12-02 → 2025-02-01 | 1,742.14 | -7.28% | -₹3.3L |
| IRFC | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -₹3.1L |

### Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +₹46.8L |
| SHRIRAMFIN | 2025-11-03 → 2026-03-02 | 796.45 | +32.15% | +₹25.0L |
| BSE | 2025-05-02 → 2025-06-02 | 2,102.17 | +28.12% | +₹15.4L |
| PAYTM | 2025-08-01 → 2025-09-01 | 1,076.40 | +14.81% | +₹8.7L |
| IDEA | 2025-10-01 → 2025-11-03 | 8.52 | +11.97% | +₹8.3L |

## Comparison vs production model

| Metric | momentum_n100_top5_max1 (real N100, LIVE) | **momentum_pseudo_n100_adv (this)** |
|---|---:|---:|
| Universe | NSE official 104 stocks | Top-100 by ADV from N500 minus Smallcap |
| CAGR | +85.85% | **+149.15%** |
| Max DD (rebal) | 33.89% | **16.17%** |
| WR | 69.0% | **88.9%** |
| Trades | 29 | 27 |
| MAX_PRICE filter | ₹3,000 | ₹3,000 |

Pseudo wins both axes (return + DD) **partly because it includes mid-cap winners (BSE, MAZDOCK, NETWEB, GRSE etc.) that graduated to N100 only later**. Yearly-PIT lookahead artifact, not purely real edge.

## Why include this model

1. **Upper-bound reference** for what the strategy can achieve.
2. **Discovery tool** — shows which mid-caps would have helped if known in advance.
3. **NOT for live deployment** — universe rebuild would need monthly ADV refresh, real-time picks would not match yearly-PIT. (MAX_PRICE filter itself IS live-deployable.)

For live use, deploy `momentum_n100_top5_max1` (real Nifty 100).

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer (yearly-PIT pseudo-N100, lb=30, mc=1, monthly, MAX_PRICE=3000) |
| `build_universe.py` | ADV-rank N500 → top-100 (with end-date param for PIT) |
| `trade_ledger.json` | 27 trades + open position |

## How to reproduce

```bash
docker exec trading_system_app python tools/models/momentum_pseudo_n100_adv/backtest.py
```

Full ledger: `exports/models/momentum_pseudo_n100_adv/TRADE_LEDGER.md`. Summary page: `exports/models/momentum_pseudo_n100_adv/SUMMARY.md`.

## Honest caveats

- **Lookahead universe**: yearly rebuild at year start. Real-time live trading would not have access to the future winners' ADV ranking. MAX_PRICE filter itself is PIT-safe.
- **Survivorship**: stocks delisted mid-window are missing.
- **No costs modeled** (vs midcap which does). Add ~1-2%/yr STT+brokerage drag for 27 trades/3yr.
- **Production use**: don't. Use `momentum_n100_top5_max1` instead.

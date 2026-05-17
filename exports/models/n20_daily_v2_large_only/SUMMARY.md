# n20_daily_v2_large_only — SUMMARY

**v2 of `n20_daily_30d_mc1_uptrend`** with NSE Nifty 100 (Large-cap) membership filter. Halves Max DD with -16pp CAGR cost.

## Stock pick logic (plain English)

1. **Universe build (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep only stocks where close > 200-day SMA
3. **NEW v2 — Large-cap filter**: keep only stocks in NSE Nifty 100 (`src/data/symbols/nifty100.csv`)
4. **Rank by 30-day return** (highest first)
5. **Pick top-1** from filtered set; if empty, hold cash
6. **Rebalance daily** (re-rank + rotate)

## Key knobs

| Knob | Value |
|---|---|
| Universe pool | Top-20 by 20-day ADV from N500 |
| Uptrend filter | close > 200d SMA |
| **NSE Nifty 100 filter (NEW)** | **Stock must be in NSE Nifty 100 list** |
| Lookback | 30 days |
| Position | top-1, max_concurrent=1 |
| Rebalance | Daily |
| Cash policy | Sit in cash if no large-cap candidate matches |

## Headline result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | v1 baseline | **v2 Large-only** | Δ |
|---|---:|---:|---:|
| Final NAV | ₹1.70 Cr | **₹13,959,936** | -₹30 L |
| CAGR | +157.27% | **+140.78%** | -16.5pp |
| Max DD (NAV) | 50.61% | **26.92%** | **-23.7pp** ✅ |
| Max DD (cash) | 50.61% | 25.52% | -25.1pp ✅ |
| Calmar | 3.10 | **5.23** | **+2.13** ✅ |
| Trades | 134 | 139 | +5 |
| WR | 47.8% | 43.1% | -4.7pp |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 139 | 59 | 78 | 43% | +11,076,686 |

All trades are Large-cap by construction. 

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹5,558,933 | **+455.89%** | 36 |
| 2024-25 | ₹5,558,933 | ₹10,791,611 | **+94.13%** | 52 |
| 2025-26 | ₹10,791,611 | ₹13,959,936 | **+29.36%** | 51 |

## Why Large-only chosen

Sweep tested 15+ pure-number DD-reduction filters (hard SL, trail SL, mc>1 diversification, vol caps, port-DD circuit breakers, combos). ALL harmed CAGR more than they cut DD. Only NSE Nifty 100 membership filter halved DD with acceptable CAGR cost.

| Filter approach tested | CAGR | DD | Calmar |
|---|---:|---:|---:|
| v1 baseline (no filter) | +157.27% | 50.61% | 3.10 |
| Hard SL -5%/-7% | +157.11% | 50.61% | 3.10 |
| Trail SL -10% | +157.11% | 50.61% | 3.10 |
| mc=2 (2 positions) | +75.39% | 32.32% | 2.33 |
| mc=3 | +42.25% | 29.59% | 1.43 |
| Max daily vol 4% | +68.07% | 32.42% | 2.10 |
| Max daily vol 3% | +31.52% | 41.03% | 0.77 |
| Halt on port DD>15% | +24.56% | 17.88% | 1.37 |
| Halt on port DD>20% | +34.53% | 20.06% | 1.72 |
| min 30d return >=10% | +154.72% | 48.04% | 3.22 |
| min 30d return >=20% | +123.72% | 47.15% | 2.62 |
| **NSE Nifty 100 filter (v2)** | **+140.78%** | **26.92%** | **5.23** ✅ |

**Insight**: pure-number filters (SL, vol-cap, mc>1) fail because strategy's edge IS the concentrated high-vol top-1 momentum. Cutting volatility cuts the edge. Only cap-membership filter (categorical, not numeric) preserves edge while dropping structurally volatile small/mid stocks.

## Caveats

- WR drops 47.8% → 43.1%. Strategy enters more often but wins less often. Net Calmar improves.
- 26.9% DD still substantial for single-stock concentration.
- 139 trades / 3yr = ~46/yr round-trip → ~3-5%/yr cost drag.
- NSE Nifty 100 list refreshes quarterly (Mar/Sep rebalance). Run `tools/refresh_nifty100.py` to keep current.
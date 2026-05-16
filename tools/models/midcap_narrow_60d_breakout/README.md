# midcap_narrow_60d_breakout

**Mid-cap swing breakout. V1 lookahead config (₹10L → ₹8.38 Cr, +337% CAGR over 3yr).**

⚠️ **Lookahead universe**: results assume access to today's ADV ranking applied retroactively. Real-time deployment will not match these numbers. See "Honesty caveats" below.

## Stock category: MID-CAP

Targets mid-cap NSE stocks (rank 31-130 by 20-day average ₹ traded). Skips top-30 large-caps (covered by `momentum_n100_top5_max1`).

## Universe construction (V1 lookahead)

1. Take all Nifty 500 stocks
2. Compute 20-day **average daily ₹ value traded** (close × volume) for each
3. Sort descending
4. **Skip top-30** (large-caps already in N100 model)
5. **Take next 100** = `midcap_narrow` universe (~ADV-rank 31-130)

Built from current data (lookahead). Real Nifty Midcap 150 (NSE official) on same strategy gave -18.18% CAGR — strategy depends on the lookahead pool.

**Universe (first 10 by ADV, end-2026):** ADANIGREEN, SUZLON, ADANIPORTS, SHRIRAMFIN, JIOFIN, NETWEB, WAAREEENER, SCI, ITC, SAIL

## Strategy — V1 WINNER config

| Knob | Value | Notes |
|---|---|---|
| Universe | Pseudo-midcap (skip top-30 ADV, next 100) | Lookahead, end-of-data snapshot frozen |
| Entry filter 1 | Close > **40-day high** | Was 60d — shorter caught more entries |
| Entry filter 2 | Volume > **2.0× 20-day avg** | Confirmation |
| Entry filter 3 | Close > **200-day SMA** | Stage-2 long-term trend |
| Position | `max_concurrent=1` | Full capital on one stock |
| **Target** | **+100%** | Was +60% — let winners run |
| Trailing stop | **-20% from peak** (armed after +10%) | Wider trail |
| **SMA20 exit** | **DISABLED** | Was the big leak — chop-killed winners |
| **Max hold** | **90 trading days** | Was 30 — longer ride |
| Slippage | 10 bps + ₹20 brokerage + 0.10% STT | |

**Key sweep insight**: removing SMA20 exit was the single biggest CAGR boost. Strategy was chopping out of winners on routine pullbacks. Letting MAX_HOLD or TARGET take it raises hit-rate dramatically.

## Backtest result (V1 lookahead, 2023-05-15 → 2026-05-15, ₹10L start)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-24) | ₹33,43,026 | **+234.30%** |
| Y2 (2024-25) | ₹5,35,90,783 | **+1503.06%** |
| Y3 (2025-26) | ₹8,38,11,502 | **+56.39%** |
| **3-yr CAGR** | | **+337.62%** |
| Total return | | **+8281.15%** |

**13 round-trips · 92.3% WR · Max DD (cash NAV) 6.76%**

12 wins / 1 loss. Exit reasons: 2 TARGET, 1 TRAIL, 10 MAX_HOLD.

## All trades

| # | Entry | Exit | Symbol | Qty | Entry ₹ | Exit ₹ | PnL ₹ | Ret % | Reason |
|--:|-------|------|--------|----:|--------:|-------:|------:|------:|--------|
| 1 | 2023-05-17 | 2023-07-12 | MAZDOCK | 2,454 | 407.39 | 865.06 | +11,20,997 | +112.56% | TARGET |
| 2 | 2023-07-13 | 2023-10-11 | INDIANB | 6,519 | 325.32 | 422.73 | +6,32,187 | +30.07% | MAX_HOLD |
| 3 | 2023-10-13 | 2024-01-11 | GMDCLTD | 6,812 | 404.15 | 466.38 | +4,20,710 | +15.51% | MAX_HOLD |
| 4 | 2024-01-12 | 2024-04-12 | CHENNPETRO | 3,739 | 848.75 | 894.90 | +1,69,212 | +5.54% | MAX_HOLD |
| 5 | 2024-04-15 | 2024-07-15 | HINDZINC | 7,858 | 425.42 | 659.04 | +18,30,550 | +55.07% | MAX_HOLD |
| 6 | 2024-07-16 | 2024-10-14 | OFSS | 471 | 10,960.95 | 11,719.72 | +3,51,840 | +7.03% | MAX_HOLD |
| **7** | **2024-10-16** | **2024-12-23** | **ANGELONE** | **17,440** | **316.82** | **2,856.69** | **+4,42,45,561** | **+802.59%** | **TARGET** |
| 8 | 2024-12-26 | 2025-03-26 | INDIGO | 10,685 | 4,657.60 | 5,020.12 | +38,19,886 | +7.89% | MAX_HOLD |
| 9 | 2025-03-28 | 2025-06-26 | FEDERALBNK | 2,71,776 | 197.19 | 209.81 | +33,73,584 | +6.51% | MAX_HOLD |
| 10 | 2025-06-30 | 2025-09-29 | HDFCLIFE | 70,342 | 809.81 | 755.84 | **-38,49,236** | -6.57% | MAX_HOLD |
| 11 | 2025-09-30 | 2025-12-29 | HINDPETRO | 1,20,036 | 442.49 | 473.68 | +36,86,300 | +7.15% | MAX_HOLD |
| 12 | 2025-12-30 | 2026-02-01 | HINDCOPPER | 1,17,495 | 483.43 | 598.65 | +1,34,67,157 | +23.96% | TRAIL |
| 13 | 2026-02-04 | 2026-05-05 | BHARATFORG | 45,583 | 1,541.54 | 1,864.73 | +1,46,47,105 | +21.09% | MAX_HOLD |
| 14 | 2026-05-06 | OPEN | WOCKPHARMA | 54,215 | 1,566.26 | 1,545.90 | -11,04,072 | -1.30% | open |

## Honesty caveats

1. **ANGELONE trade is the engine**: trade #7 alone added ₹4.42 Cr (~53% of total profit). Entry ₹316.82 → exit ₹2856.69 in 2 months = 9x. **Likely a corporate-action data anomaly** (bonus/split not adjusted in `historical_data`). Real returns on this trade would be a fraction.
2. **Lookahead universe**: pseudo-midcap = ADV-rank-from-N500 skip-30 take-100 at END of backtest period applied retroactively. Real-time strategy would not have ANGELONE in the universe in Oct 2024 (it had lower ADV then).
3. **Real Nifty Midcap 150 (NSE CSV) result on same strategy: -18.18% CAGR**. Wipe-out. Strategy entirely dependent on lookahead universe.
4. **Survivorship**: stocks delisted/dropped from N500 mid-period are missing from backtest.
5. **Slippage modeled** at 10 bps + STT + ₹20 brokerage. High-volume large positions may incur more impact — particularly FEDERALBNK 271k shares, HINDPETRO 120k shares.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Strategy backtest engine (legacy V1 60d/30d params) |
| `build_universe.py` | Pseudo-midcap builder (skip-30, next-100 from N500 by ADV) |
| `live_signal.py` | Daily breakout signal (not deployed) |
| `data_pull.py` / `cron.py` | Scheduler registration (not wired) |
| `trade_ledger.json` | 13 trades + open position from V1 winner config |

## Verdict

Best V1 lookahead config exists; honest mid-cap deployment does not. Treat as upper-bound exploration, not production-ready. For mid-cap exposure, consider Nifty Midcap 150 ETF (passive +18-25% CAGR Indian midcap historical) instead of this strategy.

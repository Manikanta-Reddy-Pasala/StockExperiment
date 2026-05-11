# UPL (UPL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-11-26 15:25:00 (47181 bars)
- **Last close:** 760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 98 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 20 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 78
- **Target hits / Stop hits / Partials:** 20 / 78 / 38
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 16.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 23 | 37.1% | 7 | 39 | 16 | 0.04% | 2.2% |
| BUY @ 2nd Alert (retest1) | 62 | 23 | 37.1% | 7 | 39 | 16 | 0.04% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 74 | 35 | 47.3% | 13 | 39 | 22 | 0.19% | 14.3% |
| SELL @ 2nd Alert (retest1) | 74 | 35 | 47.3% | 13 | 39 | 22 | 0.19% | 14.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 136 | 58 | 42.6% | 20 | 78 | 38 | 0.12% | 16.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:55:00 | 651.95 | 649.39 | 0.00 | ORB-long ORB[645.62,650.37] vol=1.7x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 10:10:00 | 653.64 | 650.26 | 0.00 | T1 1.5R @ 653.64 |
| Stop hit — per-position SL triggered | 2023-05-17 10:30:00 | 651.95 | 650.88 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 651.28 | 654.86 | 0.00 | ORB-short ORB[652.53,657.51] vol=1.7x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:40:00 | 649.49 | 654.12 | 0.00 | T1 1.5R @ 649.49 |
| Stop hit — per-position SL triggered | 2023-05-25 09:45:00 | 651.28 | 653.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 09:40:00 | 654.78 | 653.16 | 0.00 | ORB-long ORB[648.45,654.11] vol=1.9x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-05-26 10:00:00 | 653.38 | 653.61 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:50:00 | 657.32 | 659.97 | 0.00 | ORB-short ORB[658.81,665.71] vol=2.7x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-05-30 11:15:00 | 658.35 | 659.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 665.81 | 664.14 | 0.00 | ORB-long ORB[662.26,664.76] vol=2.2x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 09:35:00 | 667.08 | 665.59 | 0.00 | T1 1.5R @ 667.08 |
| Target hit | 2023-06-06 09:50:00 | 666.19 | 666.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2023-06-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 10:35:00 | 668.98 | 666.71 | 0.00 | ORB-long ORB[663.41,667.87] vol=1.5x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:40:00 | 670.40 | 667.22 | 0.00 | T1 1.5R @ 670.40 |
| Stop hit — per-position SL triggered | 2023-06-08 10:45:00 | 668.98 | 667.42 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:45:00 | 660.68 | 658.55 | 0.00 | ORB-long ORB[653.34,658.52] vol=3.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-06-16 10:05:00 | 659.20 | 659.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:00:00 | 655.12 | 657.05 | 0.00 | ORB-short ORB[655.64,658.86] vol=1.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-06-21 11:25:00 | 655.96 | 656.76 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-07-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:30:00 | 655.07 | 655.82 | 0.00 | ORB-short ORB[656.31,660.87] vol=2.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 12:45:00 | 653.31 | 655.39 | 0.00 | T1 1.5R @ 653.31 |
| Target hit | 2023-07-03 15:20:00 | 652.48 | 654.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 660.06 | 658.25 | 0.00 | ORB-long ORB[653.34,659.24] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-07-04 09:50:00 | 658.79 | 658.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 09:35:00 | 642.69 | 645.70 | 0.00 | ORB-short ORB[644.61,648.45] vol=2.9x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:35:00 | 640.39 | 644.26 | 0.00 | T1 1.5R @ 640.39 |
| Target hit | 2023-07-07 15:20:00 | 635.02 | 640.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-07-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:05:00 | 613.77 | 616.25 | 0.00 | ORB-short ORB[615.83,623.03] vol=3.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 12:25:00 | 611.85 | 614.80 | 0.00 | T1 1.5R @ 611.85 |
| Target hit | 2023-07-13 15:20:00 | 606.39 | 611.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-07-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 11:05:00 | 617.70 | 615.38 | 0.00 | ORB-long ORB[611.28,616.79] vol=9.2x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:20:00 | 619.50 | 615.70 | 0.00 | T1 1.5R @ 619.50 |
| Stop hit — per-position SL triggered | 2023-07-17 13:10:00 | 617.70 | 616.45 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 612.48 | 613.95 | 0.00 | ORB-short ORB[613.24,617.46] vol=2.3x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 09:45:00 | 611.19 | 613.79 | 0.00 | T1 1.5R @ 611.19 |
| Stop hit — per-position SL triggered | 2023-07-20 09:50:00 | 612.48 | 613.75 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 10:40:00 | 609.60 | 612.39 | 0.00 | ORB-short ORB[610.75,614.40] vol=3.2x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-07-21 13:40:00 | 610.50 | 610.43 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:30:00 | 609.46 | 607.24 | 0.00 | ORB-long ORB[601.97,609.02] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2023-07-26 09:35:00 | 608.35 | 607.35 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:55:00 | 603.60 | 600.94 | 0.00 | ORB-long ORB[596.89,603.08] vol=1.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2023-07-28 12:05:00 | 602.40 | 601.42 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:50:00 | 603.36 | 601.13 | 0.00 | ORB-long ORB[598.76,602.69] vol=1.6x ATR=1.32 |
| Stop hit — per-position SL triggered | 2023-07-31 09:55:00 | 602.04 | 601.29 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 595.02 | 598.62 | 0.00 | ORB-short ORB[596.84,603.89] vol=2.5x ATR=1.17 |
| Stop hit — per-position SL triggered | 2023-08-02 11:15:00 | 596.19 | 598.22 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:25:00 | 580.58 | 581.14 | 0.00 | ORB-short ORB[580.68,584.08] vol=3.0x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-08-08 10:40:00 | 581.49 | 581.13 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 09:30:00 | 581.78 | 580.89 | 0.00 | ORB-long ORB[578.23,581.01] vol=14.1x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-08-09 09:35:00 | 580.82 | 580.92 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 11:00:00 | 579.62 | 581.52 | 0.00 | ORB-short ORB[581.73,586.96] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 13:00:00 | 578.12 | 580.69 | 0.00 | T1 1.5R @ 578.12 |
| Target hit | 2023-08-11 15:20:00 | 574.87 | 578.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2023-08-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:25:00 | 561.97 | 564.59 | 0.00 | ORB-short ORB[563.17,567.20] vol=1.8x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:20:00 | 560.40 | 563.18 | 0.00 | T1 1.5R @ 560.40 |
| Stop hit — per-position SL triggered | 2023-08-17 12:25:00 | 561.97 | 562.71 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:05:00 | 555.93 | 559.95 | 0.00 | ORB-short ORB[558.90,563.70] vol=1.9x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-08-18 10:10:00 | 557.02 | 559.68 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:50:00 | 564.32 | 561.01 | 0.00 | ORB-long ORB[556.46,560.20] vol=3.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-08-21 11:35:00 | 563.11 | 561.64 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 565.57 | 564.48 | 0.00 | ORB-long ORB[562.50,564.99] vol=2.1x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 09:35:00 | 566.74 | 565.23 | 0.00 | T1 1.5R @ 566.74 |
| Target hit | 2023-08-24 09:50:00 | 566.91 | 566.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2023-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:30:00 | 557.42 | 560.89 | 0.00 | ORB-short ORB[558.57,563.65] vol=1.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-08-25 11:05:00 | 558.70 | 560.14 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 10:15:00 | 562.12 | 560.87 | 0.00 | ORB-long ORB[558.28,561.16] vol=2.0x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 11:00:00 | 563.52 | 561.26 | 0.00 | T1 1.5R @ 563.52 |
| Stop hit — per-position SL triggered | 2023-08-28 11:50:00 | 562.12 | 561.52 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:45:00 | 580.10 | 578.08 | 0.00 | ORB-long ORB[575.16,579.96] vol=1.6x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:20:00 | 581.90 | 579.01 | 0.00 | T1 1.5R @ 581.90 |
| Stop hit — per-position SL triggered | 2023-08-30 10:40:00 | 580.10 | 579.20 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 09:30:00 | 572.24 | 573.62 | 0.00 | ORB-short ORB[572.57,577.56] vol=2.1x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-08-31 09:35:00 | 573.43 | 573.60 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:30:00 | 588.16 | 586.45 | 0.00 | ORB-long ORB[582.31,587.82] vol=1.6x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:05:00 | 590.08 | 588.06 | 0.00 | T1 1.5R @ 590.08 |
| Stop hit — per-position SL triggered | 2023-09-07 10:30:00 | 588.16 | 588.34 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:20:00 | 585.47 | 587.40 | 0.00 | ORB-short ORB[587.49,589.84] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-09-08 10:25:00 | 586.47 | 587.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:15:00 | 587.82 | 586.25 | 0.00 | ORB-long ORB[583.32,586.82] vol=2.3x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 11:50:00 | 589.57 | 587.43 | 0.00 | T1 1.5R @ 589.57 |
| Target hit | 2023-09-11 12:55:00 | 588.64 | 588.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 594.39 | 591.35 | 0.00 | ORB-long ORB[585.14,593.10] vol=3.2x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:50:00 | 596.71 | 593.38 | 0.00 | T1 1.5R @ 596.71 |
| Target hit | 2023-09-14 15:20:00 | 606.15 | 602.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2023-09-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:00:00 | 592.96 | 598.23 | 0.00 | ORB-short ORB[599.62,603.84] vol=2.6x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-09-22 10:10:00 | 594.97 | 597.84 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:35:00 | 596.22 | 594.03 | 0.00 | ORB-long ORB[589.93,594.25] vol=1.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-09-26 12:35:00 | 595.06 | 595.30 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:15:00 | 586.39 | 589.01 | 0.00 | ORB-short ORB[588.50,593.34] vol=1.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-09-27 10:55:00 | 587.83 | 587.57 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 09:40:00 | 593.48 | 589.15 | 0.00 | ORB-long ORB[584.42,589.02] vol=1.7x ATR=1.97 |
| Stop hit — per-position SL triggered | 2023-09-29 10:40:00 | 591.51 | 591.63 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 10:45:00 | 584.51 | 586.69 | 0.00 | ORB-short ORB[585.76,590.85] vol=2.2x ATR=1.35 |
| Stop hit — per-position SL triggered | 2023-10-03 11:15:00 | 585.86 | 586.32 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:35:00 | 583.51 | 581.57 | 0.00 | ORB-long ORB[580.34,582.84] vol=2.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 582.57 | 582.34 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:35:00 | 591.80 | 589.30 | 0.00 | ORB-long ORB[584.32,587.06] vol=5.5x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 11:05:00 | 593.81 | 591.02 | 0.00 | T1 1.5R @ 593.81 |
| Target hit | 2023-10-10 15:20:00 | 596.07 | 595.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-10-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:20:00 | 600.77 | 600.92 | 0.00 | ORB-short ORB[601.64,603.75] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2023-10-12 10:40:00 | 601.83 | 600.95 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:10:00 | 598.33 | 596.97 | 0.00 | ORB-long ORB[591.28,597.99] vol=2.0x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-10-13 10:35:00 | 596.90 | 597.15 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 09:40:00 | 600.10 | 598.23 | 0.00 | ORB-long ORB[596.07,599.38] vol=3.5x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 10:05:00 | 601.78 | 599.11 | 0.00 | T1 1.5R @ 601.78 |
| Target hit | 2023-10-16 15:20:00 | 605.43 | 603.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2023-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 09:35:00 | 604.08 | 605.78 | 0.00 | ORB-short ORB[604.85,609.02] vol=1.7x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:10:00 | 602.64 | 605.06 | 0.00 | T1 1.5R @ 602.64 |
| Target hit | 2023-10-17 15:20:00 | 599.48 | 601.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 596.65 | 600.22 | 0.00 | ORB-short ORB[597.70,601.92] vol=4.3x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 597.80 | 600.08 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:30:00 | 585.04 | 587.66 | 0.00 | ORB-short ORB[586.24,592.52] vol=2.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2023-10-19 09:45:00 | 586.72 | 587.06 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:05:00 | 575.79 | 579.35 | 0.00 | ORB-short ORB[577.66,583.22] vol=1.6x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:10:00 | 573.63 | 578.56 | 0.00 | T1 1.5R @ 573.63 |
| Stop hit — per-position SL triggered | 2023-10-23 10:25:00 | 575.79 | 578.02 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-27 10:20:00 | 536.70 | 540.60 | 0.00 | ORB-short ORB[539.72,543.65] vol=1.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2023-10-27 10:30:00 | 538.38 | 540.35 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 09:35:00 | 528.06 | 525.87 | 0.00 | ORB-long ORB[521.83,527.39] vol=2.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-11-03 09:45:00 | 526.55 | 526.50 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 09:30:00 | 527.34 | 529.12 | 0.00 | ORB-short ORB[528.16,531.37] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-11-06 10:10:00 | 528.50 | 528.33 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:30:00 | 532.38 | 531.38 | 0.00 | ORB-long ORB[528.97,532.24] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-11-07 09:40:00 | 531.49 | 531.44 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:50:00 | 534.92 | 532.54 | 0.00 | ORB-long ORB[530.56,533.34] vol=2.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-11-08 10:55:00 | 533.98 | 532.61 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 11:00:00 | 530.51 | 531.89 | 0.00 | ORB-short ORB[532.19,535.07] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:10:00 | 529.21 | 531.35 | 0.00 | T1 1.5R @ 529.21 |
| Target hit | 2023-11-09 15:20:00 | 527.15 | 528.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2023-11-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:30:00 | 540.87 | 538.31 | 0.00 | ORB-long ORB[533.67,540.34] vol=2.3x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-11-15 09:45:00 | 539.45 | 538.87 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 10:25:00 | 534.11 | 535.82 | 0.00 | ORB-short ORB[535.55,538.85] vol=1.8x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-11-16 11:00:00 | 535.07 | 535.33 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:50:00 | 539.43 | 536.75 | 0.00 | ORB-long ORB[533.82,536.36] vol=5.0x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-11-21 10:00:00 | 538.33 | 537.17 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:45:00 | 545.33 | 543.43 | 0.00 | ORB-long ORB[540.15,544.80] vol=2.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2023-11-22 12:10:00 | 543.98 | 544.51 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:35:00 | 541.59 | 542.61 | 0.00 | ORB-short ORB[542.07,545.19] vol=2.3x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:50:00 | 540.12 | 541.71 | 0.00 | T1 1.5R @ 540.12 |
| Target hit | 2023-11-24 11:15:00 | 540.53 | 540.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 11:15:00 | 535.02 | 536.38 | 0.00 | ORB-short ORB[535.83,538.09] vol=2.0x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:30:00 | 533.85 | 536.12 | 0.00 | T1 1.5R @ 533.85 |
| Stop hit — per-position SL triggered | 2023-11-28 11:40:00 | 535.02 | 535.91 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-11-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 11:10:00 | 546.34 | 546.48 | 0.00 | ORB-short ORB[546.53,549.55] vol=1.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-11-30 11:30:00 | 547.48 | 546.51 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 565.33 | 569.25 | 0.00 | ORB-short ORB[565.57,571.61] vol=1.8x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-12-08 11:20:00 | 566.64 | 568.95 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 10:15:00 | 571.13 | 569.07 | 0.00 | ORB-long ORB[559.77,568.35] vol=1.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-12-11 10:25:00 | 569.54 | 569.20 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 570.37 | 574.45 | 0.00 | ORB-short ORB[573.24,578.14] vol=1.9x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-12-13 09:40:00 | 572.19 | 573.57 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:30:00 | 578.66 | 580.75 | 0.00 | ORB-short ORB[582.26,587.06] vol=1.5x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 11:10:00 | 576.40 | 579.67 | 0.00 | T1 1.5R @ 576.40 |
| Target hit | 2023-12-19 15:20:00 | 574.06 | 577.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2023-12-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 09:40:00 | 573.63 | 575.85 | 0.00 | ORB-short ORB[574.59,580.34] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-12-20 09:50:00 | 575.08 | 575.71 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:35:00 | 569.36 | 567.26 | 0.00 | ORB-long ORB[563.27,566.82] vol=1.7x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 09:40:00 | 571.30 | 569.06 | 0.00 | T1 1.5R @ 571.30 |
| Target hit | 2024-01-01 10:00:00 | 569.55 | 569.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2024-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:30:00 | 578.09 | 573.87 | 0.00 | ORB-long ORB[568.54,573.34] vol=3.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-01-02 09:50:00 | 576.26 | 575.96 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 565.62 | 567.67 | 0.00 | ORB-short ORB[566.05,572.48] vol=1.9x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-01-03 09:45:00 | 567.18 | 567.59 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:50:00 | 561.25 | 562.98 | 0.00 | ORB-short ORB[562.31,565.19] vol=1.8x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:05:00 | 559.39 | 562.43 | 0.00 | T1 1.5R @ 559.39 |
| Stop hit — per-position SL triggered | 2024-01-04 10:10:00 | 561.25 | 562.40 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:45:00 | 532.38 | 538.29 | 0.00 | ORB-short ORB[538.95,544.80] vol=2.0x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-01-09 09:55:00 | 534.31 | 537.57 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 11:15:00 | 541.97 | 540.09 | 0.00 | ORB-long ORB[537.70,540.63] vol=3.0x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-01-11 11:50:00 | 541.04 | 540.28 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 10:00:00 | 528.93 | 532.24 | 0.00 | ORB-short ORB[531.04,535.69] vol=2.3x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:05:00 | 526.98 | 531.51 | 0.00 | T1 1.5R @ 526.98 |
| Target hit | 2024-01-23 15:20:00 | 513.63 | 521.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2024-01-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:55:00 | 519.48 | 521.65 | 0.00 | ORB-short ORB[521.54,525.28] vol=1.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:20:00 | 517.46 | 520.90 | 0.00 | T1 1.5R @ 517.46 |
| Target hit | 2024-01-25 15:20:00 | 515.16 | 517.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2024-01-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:30:00 | 524.47 | 522.11 | 0.00 | ORB-long ORB[519.29,524.18] vol=1.7x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-01-30 10:45:00 | 523.14 | 522.23 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 09:55:00 | 455.78 | 459.62 | 0.00 | ORB-short ORB[458.52,462.16] vol=1.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 10:00:00 | 453.89 | 458.02 | 0.00 | T1 1.5R @ 453.89 |
| Target hit | 2024-02-08 15:20:00 | 446.05 | 450.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 09:30:00 | 446.34 | 444.25 | 0.00 | ORB-long ORB[441.88,445.90] vol=1.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-02-12 09:40:00 | 444.69 | 444.60 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:10:00 | 475.78 | 478.00 | 0.00 | ORB-short ORB[475.88,479.81] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-02-21 10:15:00 | 476.84 | 477.89 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:35:00 | 472.04 | 469.37 | 0.00 | ORB-long ORB[467.30,469.93] vol=1.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-02-26 09:45:00 | 470.97 | 470.16 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 09:30:00 | 466.14 | 468.34 | 0.00 | ORB-short ORB[467.82,471.47] vol=4.3x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 09:40:00 | 464.32 | 467.29 | 0.00 | T1 1.5R @ 464.32 |
| Stop hit — per-position SL triggered | 2024-02-27 12:25:00 | 466.14 | 465.18 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 461.40 | 464.04 | 0.00 | ORB-short ORB[463.17,466.19] vol=1.5x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-02-28 11:10:00 | 462.28 | 463.66 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:35:00 | 456.31 | 458.86 | 0.00 | ORB-short ORB[456.60,460.58] vol=1.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-03-04 09:45:00 | 457.59 | 458.60 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 450.60 | 453.17 | 0.00 | ORB-short ORB[452.09,458.04] vol=1.8x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:55:00 | 448.79 | 451.75 | 0.00 | T1 1.5R @ 448.79 |
| Stop hit — per-position SL triggered | 2024-03-06 10:45:00 | 450.60 | 450.56 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 09:30:00 | 448.69 | 451.77 | 0.00 | ORB-short ORB[451.80,455.64] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 09:45:00 | 445.59 | 449.88 | 0.00 | T1 1.5R @ 445.59 |
| Stop hit — per-position SL triggered | 2024-03-18 14:45:00 | 448.69 | 447.67 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-03-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:30:00 | 443.36 | 440.82 | 0.00 | ORB-long ORB[436.46,441.59] vol=2.3x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-03-22 09:35:00 | 441.97 | 441.08 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-04-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:05:00 | 449.60 | 446.08 | 0.00 | ORB-long ORB[442.21,447.20] vol=1.5x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-04-02 10:30:00 | 447.97 | 446.73 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 454.01 | 458.04 | 0.00 | ORB-short ORB[458.76,461.30] vol=1.7x ATR=1.37 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 455.38 | 457.95 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 476.22 | 473.00 | 0.00 | ORB-long ORB[469.45,474.83] vol=1.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-04-10 10:55:00 | 474.68 | 473.34 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 480.44 | 485.82 | 0.00 | ORB-short ORB[484.18,490.27] vol=1.7x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-04-12 09:45:00 | 482.35 | 485.60 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:45:00 | 469.07 | 467.02 | 0.00 | ORB-long ORB[463.36,468.54] vol=2.0x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 11:10:00 | 471.85 | 467.76 | 0.00 | T1 1.5R @ 471.85 |
| Stop hit — per-position SL triggered | 2024-04-16 11:30:00 | 469.07 | 467.85 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 479.72 | 478.63 | 0.00 | ORB-long ORB[475.40,478.61] vol=2.7x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 09:40:00 | 481.66 | 479.19 | 0.00 | T1 1.5R @ 481.66 |
| Stop hit — per-position SL triggered | 2024-04-24 11:05:00 | 479.72 | 480.59 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 11:10:00 | 484.03 | 482.32 | 0.00 | ORB-long ORB[476.12,483.36] vol=1.8x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-04-25 11:35:00 | 482.86 | 482.44 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-04-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 11:00:00 | 489.36 | 486.39 | 0.00 | ORB-long ORB[485.52,489.21] vol=4.0x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-04-26 11:05:00 | 487.97 | 486.47 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 09:35:00 | 483.12 | 484.52 | 0.00 | ORB-short ORB[484.03,489.60] vol=4.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 14:15:00 | 480.86 | 483.15 | 0.00 | T1 1.5R @ 480.86 |
| Target hit | 2024-05-02 15:20:00 | 477.94 | 481.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:15:00 | 478.18 | 480.39 | 0.00 | ORB-short ORB[479.05,483.31] vol=4.9x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:00:00 | 476.23 | 479.81 | 0.00 | T1 1.5R @ 476.23 |
| Target hit | 2024-05-03 15:20:00 | 473.39 | 476.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 96 — SELL (started 2024-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:55:00 | 470.60 | 473.02 | 0.00 | ORB-short ORB[472.14,477.99] vol=7.1x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-05-06 10:10:00 | 472.33 | 472.19 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-05-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:30:00 | 461.88 | 460.29 | 0.00 | ORB-long ORB[457.70,461.68] vol=2.0x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-05-09 09:35:00 | 460.62 | 460.45 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-05-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 11:00:00 | 459.38 | 452.64 | 0.00 | ORB-long ORB[445.76,451.80] vol=6.6x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:10:00 | 462.84 | 454.37 | 0.00 | T1 1.5R @ 462.84 |
| Stop hit — per-position SL triggered | 2024-05-10 11:50:00 | 459.38 | 456.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-17 09:55:00 | 651.95 | 2023-05-17 10:10:00 | 653.64 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-05-17 09:55:00 | 651.95 | 2023-05-17 10:30:00 | 651.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-25 09:30:00 | 651.28 | 2023-05-25 09:40:00 | 649.49 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-25 09:30:00 | 651.28 | 2023-05-25 09:45:00 | 651.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-26 09:40:00 | 654.78 | 2023-05-26 10:00:00 | 653.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-30 10:50:00 | 657.32 | 2023-05-30 11:15:00 | 658.35 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-06 09:30:00 | 665.81 | 2023-06-06 09:35:00 | 667.08 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2023-06-06 09:30:00 | 665.81 | 2023-06-06 09:50:00 | 666.19 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2023-06-08 10:35:00 | 668.98 | 2023-06-08 10:40:00 | 670.40 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-06-08 10:35:00 | 668.98 | 2023-06-08 10:45:00 | 668.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 09:45:00 | 660.68 | 2023-06-16 10:05:00 | 659.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-21 11:00:00 | 655.12 | 2023-06-21 11:25:00 | 655.96 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-07-03 10:30:00 | 655.07 | 2023-07-03 12:45:00 | 653.31 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-03 10:30:00 | 655.07 | 2023-07-03 15:20:00 | 652.48 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-04 09:40:00 | 660.06 | 2023-07-04 09:50:00 | 658.79 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-07-07 09:35:00 | 642.69 | 2023-07-07 10:35:00 | 640.39 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-07-07 09:35:00 | 642.69 | 2023-07-07 15:20:00 | 635.02 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2023-07-13 10:05:00 | 613.77 | 2023-07-13 12:25:00 | 611.85 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-13 10:05:00 | 613.77 | 2023-07-13 15:20:00 | 606.39 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2023-07-17 11:05:00 | 617.70 | 2023-07-17 11:20:00 | 619.50 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-17 11:05:00 | 617.70 | 2023-07-17 13:10:00 | 617.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-20 09:40:00 | 612.48 | 2023-07-20 09:45:00 | 611.19 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-07-20 09:40:00 | 612.48 | 2023-07-20 09:50:00 | 612.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-21 10:40:00 | 609.60 | 2023-07-21 13:40:00 | 610.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-07-26 09:30:00 | 609.46 | 2023-07-26 09:35:00 | 608.35 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-28 10:55:00 | 603.60 | 2023-07-28 12:05:00 | 602.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-31 09:50:00 | 603.36 | 2023-07-31 09:55:00 | 602.04 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-02 11:00:00 | 595.02 | 2023-08-02 11:15:00 | 596.19 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-08 10:25:00 | 580.58 | 2023-08-08 10:40:00 | 581.49 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-08-09 09:30:00 | 581.78 | 2023-08-09 09:35:00 | 580.82 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-08-11 11:00:00 | 579.62 | 2023-08-11 13:00:00 | 578.12 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-11 11:00:00 | 579.62 | 2023-08-11 15:20:00 | 574.87 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2023-08-17 10:25:00 | 561.97 | 2023-08-17 11:20:00 | 560.40 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-08-17 10:25:00 | 561.97 | 2023-08-17 12:25:00 | 561.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-18 10:05:00 | 555.93 | 2023-08-18 10:10:00 | 557.02 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-21 10:50:00 | 564.32 | 2023-08-21 11:35:00 | 563.11 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-24 09:30:00 | 565.57 | 2023-08-24 09:35:00 | 566.74 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-08-24 09:30:00 | 565.57 | 2023-08-24 09:50:00 | 566.91 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2023-08-25 10:30:00 | 557.42 | 2023-08-25 11:05:00 | 558.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-28 10:15:00 | 562.12 | 2023-08-28 11:00:00 | 563.52 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-08-28 10:15:00 | 562.12 | 2023-08-28 11:50:00 | 562.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 09:45:00 | 580.10 | 2023-08-30 10:20:00 | 581.90 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-08-30 09:45:00 | 580.10 | 2023-08-30 10:40:00 | 580.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-31 09:30:00 | 572.24 | 2023-08-31 09:35:00 | 573.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-07 09:30:00 | 588.16 | 2023-09-07 10:05:00 | 590.08 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-09-07 09:30:00 | 588.16 | 2023-09-07 10:30:00 | 588.16 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 10:20:00 | 585.47 | 2023-09-08 10:25:00 | 586.47 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-11 10:15:00 | 587.82 | 2023-09-11 11:50:00 | 589.57 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-11 10:15:00 | 587.82 | 2023-09-11 12:55:00 | 588.64 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-09-14 09:30:00 | 594.39 | 2023-09-14 09:50:00 | 596.71 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-14 09:30:00 | 594.39 | 2023-09-14 15:20:00 | 606.15 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2023-09-22 10:00:00 | 592.96 | 2023-09-22 10:10:00 | 594.97 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-09-26 10:35:00 | 596.22 | 2023-09-26 12:35:00 | 595.06 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-27 10:15:00 | 586.39 | 2023-09-27 10:55:00 | 587.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-29 09:40:00 | 593.48 | 2023-09-29 10:40:00 | 591.51 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-10-03 10:45:00 | 584.51 | 2023-10-03 11:15:00 | 585.86 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-06 10:35:00 | 583.51 | 2023-10-06 11:15:00 | 582.57 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-10-10 09:35:00 | 591.80 | 2023-10-10 11:05:00 | 593.81 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-10-10 09:35:00 | 591.80 | 2023-10-10 15:20:00 | 596.07 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2023-10-12 10:20:00 | 600.77 | 2023-10-12 10:40:00 | 601.83 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-13 10:10:00 | 598.33 | 2023-10-13 10:35:00 | 596.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-16 09:40:00 | 600.10 | 2023-10-16 10:05:00 | 601.78 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-10-16 09:40:00 | 600.10 | 2023-10-16 15:20:00 | 605.43 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2023-10-17 09:35:00 | 604.08 | 2023-10-17 10:10:00 | 602.64 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-10-17 09:35:00 | 604.08 | 2023-10-17 15:20:00 | 599.48 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2023-10-18 11:10:00 | 596.65 | 2023-10-18 11:15:00 | 597.80 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-19 09:30:00 | 585.04 | 2023-10-19 09:45:00 | 586.72 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-23 10:05:00 | 575.79 | 2023-10-23 10:10:00 | 573.63 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-10-23 10:05:00 | 575.79 | 2023-10-23 10:25:00 | 575.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-27 10:20:00 | 536.70 | 2023-10-27 10:30:00 | 538.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-03 09:35:00 | 528.06 | 2023-11-03 09:45:00 | 526.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-06 09:30:00 | 527.34 | 2023-11-06 10:10:00 | 528.50 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-07 09:30:00 | 532.38 | 2023-11-07 09:40:00 | 531.49 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-11-08 10:50:00 | 534.92 | 2023-11-08 10:55:00 | 533.98 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-09 11:00:00 | 530.51 | 2023-11-09 11:10:00 | 529.21 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-11-09 11:00:00 | 530.51 | 2023-11-09 15:20:00 | 527.15 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2023-11-15 09:30:00 | 540.87 | 2023-11-15 09:45:00 | 539.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-16 10:25:00 | 534.11 | 2023-11-16 11:00:00 | 535.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-11-21 09:50:00 | 539.43 | 2023-11-21 10:00:00 | 538.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-22 09:45:00 | 545.33 | 2023-11-22 12:10:00 | 543.98 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-24 09:35:00 | 541.59 | 2023-11-24 09:50:00 | 540.12 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-11-24 09:35:00 | 541.59 | 2023-11-24 11:15:00 | 540.53 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-28 11:15:00 | 535.02 | 2023-11-28 11:30:00 | 533.85 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-11-28 11:15:00 | 535.02 | 2023-11-28 11:40:00 | 535.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-30 11:10:00 | 546.34 | 2023-11-30 11:30:00 | 547.48 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-08 11:00:00 | 565.33 | 2023-12-08 11:20:00 | 566.64 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-11 10:15:00 | 571.13 | 2023-12-11 10:25:00 | 569.54 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-13 09:30:00 | 570.37 | 2023-12-13 09:40:00 | 572.19 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-12-19 10:30:00 | 578.66 | 2023-12-19 11:10:00 | 576.40 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-19 10:30:00 | 578.66 | 2023-12-19 15:20:00 | 574.06 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2023-12-20 09:40:00 | 573.63 | 2023-12-20 09:50:00 | 575.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-01 09:35:00 | 569.36 | 2024-01-01 09:40:00 | 571.30 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-01-01 09:35:00 | 569.36 | 2024-01-01 10:00:00 | 569.55 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2024-01-02 09:30:00 | 578.09 | 2024-01-02 09:50:00 | 576.26 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-03 09:40:00 | 565.62 | 2024-01-03 09:45:00 | 567.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-04 09:50:00 | 561.25 | 2024-01-04 10:05:00 | 559.39 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-01-04 09:50:00 | 561.25 | 2024-01-04 10:10:00 | 561.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-09 09:45:00 | 532.38 | 2024-01-09 09:55:00 | 534.31 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-11 11:15:00 | 541.97 | 2024-01-11 11:50:00 | 541.04 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-23 10:00:00 | 528.93 | 2024-01-23 10:05:00 | 526.98 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-01-23 10:00:00 | 528.93 | 2024-01-23 15:20:00 | 513.63 | TARGET_HIT | 0.50 | 2.89% |
| SELL | retest1 | 2024-01-25 09:55:00 | 519.48 | 2024-01-25 10:20:00 | 517.46 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-01-25 09:55:00 | 519.48 | 2024-01-25 15:20:00 | 515.16 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-01-30 10:30:00 | 524.47 | 2024-01-30 10:45:00 | 523.14 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-08 09:55:00 | 455.78 | 2024-02-08 10:00:00 | 453.89 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-08 09:55:00 | 455.78 | 2024-02-08 15:20:00 | 446.05 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2024-02-12 09:30:00 | 446.34 | 2024-02-12 09:40:00 | 444.69 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-02-21 10:10:00 | 475.78 | 2024-02-21 10:15:00 | 476.84 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-02-26 09:35:00 | 472.04 | 2024-02-26 09:45:00 | 470.97 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-27 09:30:00 | 466.14 | 2024-02-27 09:40:00 | 464.32 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-27 09:30:00 | 466.14 | 2024-02-27 12:25:00 | 466.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:50:00 | 461.40 | 2024-02-28 11:10:00 | 462.28 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-03-04 09:35:00 | 456.31 | 2024-03-04 09:45:00 | 457.59 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-06 09:30:00 | 450.60 | 2024-03-06 09:55:00 | 448.79 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-03-06 09:30:00 | 450.60 | 2024-03-06 10:45:00 | 450.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-18 09:30:00 | 448.69 | 2024-03-18 09:45:00 | 445.59 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-03-18 09:30:00 | 448.69 | 2024-03-18 14:45:00 | 448.69 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-22 09:30:00 | 443.36 | 2024-03-22 09:35:00 | 441.97 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-02 10:05:00 | 449.60 | 2024-04-02 10:30:00 | 447.97 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-04 10:50:00 | 454.01 | 2024-04-04 10:55:00 | 455.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-10 10:35:00 | 476.22 | 2024-04-10 10:55:00 | 474.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-04-12 09:40:00 | 480.44 | 2024-04-12 09:45:00 | 482.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-04-16 09:45:00 | 469.07 | 2024-04-16 11:10:00 | 471.85 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-04-16 09:45:00 | 469.07 | 2024-04-16 11:30:00 | 469.07 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 09:30:00 | 479.72 | 2024-04-24 09:40:00 | 481.66 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-04-24 09:30:00 | 479.72 | 2024-04-24 11:05:00 | 479.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 11:10:00 | 484.03 | 2024-04-25 11:35:00 | 482.86 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-26 11:00:00 | 489.36 | 2024-04-26 11:05:00 | 487.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-02 09:35:00 | 483.12 | 2024-05-02 14:15:00 | 480.86 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-02 09:35:00 | 483.12 | 2024-05-02 15:20:00 | 477.94 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2024-05-03 10:15:00 | 478.18 | 2024-05-03 11:00:00 | 476.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-03 10:15:00 | 478.18 | 2024-05-03 15:20:00 | 473.39 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2024-05-06 09:55:00 | 470.60 | 2024-05-06 10:10:00 | 472.33 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-09 09:30:00 | 461.88 | 2024-05-09 09:35:00 | 460.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-10 11:00:00 | 459.38 | 2024-05-10 11:10:00 | 462.84 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-05-10 11:00:00 | 459.38 | 2024-05-10 11:50:00 | 459.38 | STOP_HIT | 0.50 | 0.00% |

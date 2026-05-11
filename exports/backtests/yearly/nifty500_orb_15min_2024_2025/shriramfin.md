# Shriram Finance Ltd. (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-12-01 15:25:00 (28921 bars)
- **Last close:** 851.95
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
| ENTRY1 | 45 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 7 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 38
- **Target hits / Stop hits / Partials:** 7 / 38 / 21
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 13.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 17 | 43.6% | 4 | 22 | 13 | 0.20% | 7.9% |
| BUY @ 2nd Alert (retest1) | 39 | 17 | 43.6% | 4 | 22 | 13 | 0.20% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 11 | 40.7% | 3 | 16 | 8 | 0.22% | 6.0% |
| SELL @ 2nd Alert (retest1) | 27 | 11 | 40.7% | 3 | 16 | 8 | 0.22% | 6.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 66 | 28 | 42.4% | 7 | 38 | 21 | 0.21% | 13.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:55:00 | 473.06 | 470.28 | 0.00 | ORB-long ORB[465.70,469.20] vol=1.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 12:25:00 | 475.25 | 471.29 | 0.00 | T1 1.5R @ 475.25 |
| Stop hit — per-position SL triggered | 2024-05-23 13:15:00 | 473.06 | 471.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 473.93 | 474.71 | 0.00 | ORB-short ORB[474.00,479.97] vol=2.0x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-05-24 11:15:00 | 475.08 | 474.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:05:00 | 466.30 | 470.25 | 0.00 | ORB-short ORB[471.63,476.00] vol=1.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:30:00 | 464.29 | 468.85 | 0.00 | T1 1.5R @ 464.29 |
| Target hit | 2024-05-30 15:20:00 | 461.07 | 462.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:15:00 | 555.44 | 551.29 | 0.00 | ORB-long ORB[546.47,553.90] vol=2.4x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:20:00 | 558.45 | 551.76 | 0.00 | T1 1.5R @ 558.45 |
| Stop hit — per-position SL triggered | 2024-06-18 11:50:00 | 555.44 | 554.26 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 576.93 | 582.25 | 0.00 | ORB-short ORB[580.21,588.00] vol=1.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:50:00 | 573.87 | 580.71 | 0.00 | T1 1.5R @ 573.87 |
| Stop hit — per-position SL triggered | 2024-07-02 10:35:00 | 576.93 | 578.74 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:05:00 | 562.79 | 565.13 | 0.00 | ORB-short ORB[563.60,571.20] vol=2.1x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 564.20 | 565.09 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 549.19 | 555.42 | 0.00 | ORB-short ORB[557.33,560.93] vol=1.5x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 546.63 | 553.65 | 0.00 | T1 1.5R @ 546.63 |
| Stop hit — per-position SL triggered | 2024-07-10 11:55:00 | 549.19 | 549.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:30:00 | 568.73 | 565.70 | 0.00 | ORB-long ORB[561.44,567.39] vol=2.3x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:55:00 | 571.18 | 566.65 | 0.00 | T1 1.5R @ 571.18 |
| Target hit | 2024-07-15 15:20:00 | 574.00 | 571.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:50:00 | 567.82 | 572.87 | 0.00 | ORB-short ORB[573.00,580.00] vol=2.4x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-07-16 10:00:00 | 569.85 | 572.47 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 11:00:00 | 564.75 | 560.75 | 0.00 | ORB-long ORB[552.81,560.99] vol=2.3x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:25:00 | 567.48 | 561.30 | 0.00 | T1 1.5R @ 567.48 |
| Stop hit — per-position SL triggered | 2024-07-22 11:50:00 | 564.75 | 561.71 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:50:00 | 589.00 | 588.19 | 0.00 | ORB-long ORB[583.99,588.38] vol=1.8x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:40:00 | 591.20 | 589.12 | 0.00 | T1 1.5R @ 591.20 |
| Stop hit — per-position SL triggered | 2024-08-01 10:50:00 | 589.00 | 589.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:15:00 | 600.22 | 591.00 | 0.00 | ORB-long ORB[584.63,592.33] vol=2.8x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 11:35:00 | 603.74 | 593.00 | 0.00 | T1 1.5R @ 603.74 |
| Stop hit — per-position SL triggered | 2024-08-02 11:55:00 | 600.22 | 594.14 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 595.62 | 590.66 | 0.00 | ORB-long ORB[585.62,594.00] vol=2.6x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:45:00 | 598.23 | 593.00 | 0.00 | T1 1.5R @ 598.23 |
| Target hit | 2024-08-12 14:55:00 | 596.43 | 596.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-08-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:25:00 | 590.65 | 588.68 | 0.00 | ORB-long ORB[580.41,589.26] vol=2.0x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-08-16 10:30:00 | 588.53 | 588.72 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 634.10 | 631.96 | 0.00 | ORB-long ORB[623.37,629.85] vol=2.4x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-08-21 12:00:00 | 631.64 | 633.30 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 11:10:00 | 649.97 | 647.59 | 0.00 | ORB-long ORB[642.81,649.00] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-09-02 12:45:00 | 648.07 | 648.52 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 673.25 | 674.46 | 0.00 | ORB-short ORB[673.33,677.00] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-09-16 09:40:00 | 674.74 | 674.40 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 688.00 | 684.37 | 0.00 | ORB-long ORB[681.49,685.44] vol=2.2x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:45:00 | 690.53 | 685.69 | 0.00 | T1 1.5R @ 690.53 |
| Stop hit — per-position SL triggered | 2024-09-17 09:50:00 | 688.00 | 685.92 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 706.61 | 712.56 | 0.00 | ORB-short ORB[710.21,716.30] vol=1.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-09-24 09:45:00 | 708.64 | 710.98 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 11:05:00 | 720.70 | 718.65 | 0.00 | ORB-long ORB[711.56,720.65] vol=2.1x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-10-01 11:10:00 | 718.32 | 718.66 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:30:00 | 702.92 | 706.83 | 0.00 | ORB-short ORB[704.81,714.98] vol=1.6x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:45:00 | 699.52 | 705.33 | 0.00 | T1 1.5R @ 699.52 |
| Target hit | 2024-10-03 15:20:00 | 680.19 | 690.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2024-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:00:00 | 670.32 | 674.84 | 0.00 | ORB-short ORB[676.00,683.19] vol=2.1x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-10-10 10:15:00 | 672.79 | 673.95 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 681.23 | 676.33 | 0.00 | ORB-long ORB[670.75,677.76] vol=1.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:00:00 | 684.29 | 678.75 | 0.00 | T1 1.5R @ 684.29 |
| Stop hit — per-position SL triggered | 2024-10-14 10:10:00 | 681.23 | 679.73 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:30:00 | 668.20 | 672.46 | 0.00 | ORB-short ORB[669.00,677.19] vol=1.6x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:40:00 | 665.45 | 670.17 | 0.00 | T1 1.5R @ 665.45 |
| Stop hit — per-position SL triggered | 2024-10-17 10:30:00 | 668.20 | 667.55 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 619.92 | 624.39 | 0.00 | ORB-short ORB[625.73,631.52] vol=2.1x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:55:00 | 616.68 | 622.86 | 0.00 | T1 1.5R @ 616.68 |
| Stop hit — per-position SL triggered | 2024-11-05 11:15:00 | 619.92 | 622.30 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:05:00 | 621.62 | 630.11 | 0.00 | ORB-short ORB[629.40,638.78] vol=1.8x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-11-07 10:20:00 | 624.13 | 628.56 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:30:00 | 598.57 | 601.05 | 0.00 | ORB-short ORB[598.94,605.07] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-11-12 09:45:00 | 600.52 | 600.40 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:35:00 | 570.32 | 570.16 | 0.00 | ORB-long ORB[563.71,569.63] vol=2.3x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-11-19 10:50:00 | 568.57 | 570.17 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 613.06 | 609.64 | 0.00 | ORB-long ORB[606.21,609.90] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 611.48 | 610.10 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 615.34 | 622.44 | 0.00 | ORB-short ORB[623.20,631.81] vol=2.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 617.33 | 622.20 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 645.94 | 648.61 | 0.00 | ORB-short ORB[646.54,653.00] vol=2.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-12-12 11:10:00 | 647.69 | 648.55 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:45:00 | 584.00 | 581.73 | 0.00 | ORB-long ORB[579.06,583.73] vol=1.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 582.04 | 582.40 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 583.38 | 579.50 | 0.00 | ORB-long ORB[577.10,580.40] vol=1.9x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 581.88 | 580.11 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 595.30 | 591.83 | 0.00 | ORB-long ORB[586.32,594.40] vol=2.2x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:25:00 | 598.04 | 593.06 | 0.00 | T1 1.5R @ 598.04 |
| Stop hit — per-position SL triggered | 2025-01-02 11:45:00 | 595.30 | 593.63 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:55:00 | 570.26 | 577.24 | 0.00 | ORB-short ORB[578.30,584.83] vol=2.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:30:00 | 567.37 | 573.71 | 0.00 | T1 1.5R @ 567.37 |
| Target hit | 2025-01-09 15:20:00 | 563.60 | 567.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-01-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:45:00 | 539.80 | 536.52 | 0.00 | ORB-long ORB[530.25,535.80] vol=1.9x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:00:00 | 543.44 | 537.38 | 0.00 | T1 1.5R @ 543.44 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 539.80 | 538.37 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 577.35 | 573.29 | 0.00 | ORB-long ORB[567.00,575.00] vol=2.9x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 574.66 | 572.34 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 09:35:00 | 571.30 | 576.19 | 0.00 | ORB-short ORB[573.80,581.75] vol=1.9x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-02-24 09:50:00 | 574.12 | 574.62 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-03-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:45:00 | 622.05 | 620.23 | 0.00 | ORB-long ORB[613.40,621.30] vol=2.2x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:40:00 | 625.50 | 620.86 | 0.00 | T1 1.5R @ 625.50 |
| Target hit | 2025-03-04 15:20:00 | 633.30 | 626.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 649.00 | 643.63 | 0.00 | ORB-long ORB[637.00,645.80] vol=2.9x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-03-06 09:40:00 | 646.32 | 644.12 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:30:00 | 636.15 | 631.70 | 0.00 | ORB-long ORB[626.05,634.65] vol=2.3x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-03-18 09:35:00 | 634.06 | 631.96 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:45:00 | 647.70 | 643.92 | 0.00 | ORB-long ORB[637.65,647.00] vol=1.6x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:00:00 | 651.57 | 645.86 | 0.00 | T1 1.5R @ 651.57 |
| Target hit | 2025-03-19 15:20:00 | 667.80 | 664.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 685.85 | 683.43 | 0.00 | ORB-long ORB[674.00,683.55] vol=1.7x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-03-24 11:30:00 | 683.51 | 683.74 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 10:45:00 | 659.90 | 654.98 | 0.00 | ORB-long ORB[650.90,657.75] vol=1.5x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-04-04 10:55:00 | 657.31 | 655.20 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:50:00 | 699.55 | 706.48 | 0.00 | ORB-short ORB[703.35,713.80] vol=1.5x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:05:00 | 695.19 | 704.04 | 0.00 | T1 1.5R @ 695.19 |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 699.55 | 698.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-23 10:55:00 | 473.06 | 2024-05-23 12:25:00 | 475.25 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-05-23 10:55:00 | 473.06 | 2024-05-23 13:15:00 | 473.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-24 11:05:00 | 473.93 | 2024-05-24 11:15:00 | 475.08 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-30 10:05:00 | 466.30 | 2024-05-30 10:30:00 | 464.29 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-30 10:05:00 | 466.30 | 2024-05-30 15:20:00 | 461.07 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2024-06-18 10:15:00 | 555.44 | 2024-06-18 10:20:00 | 558.45 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-18 10:15:00 | 555.44 | 2024-06-18 11:50:00 | 555.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:40:00 | 576.93 | 2024-07-02 09:50:00 | 573.87 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-02 09:40:00 | 576.93 | 2024-07-02 10:35:00 | 576.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:05:00 | 562.79 | 2024-07-08 11:10:00 | 564.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 10:05:00 | 549.19 | 2024-07-10 10:20:00 | 546.63 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-10 10:05:00 | 549.19 | 2024-07-10 11:55:00 | 549.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 10:30:00 | 568.73 | 2024-07-15 10:55:00 | 571.18 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-15 10:30:00 | 568.73 | 2024-07-15 15:20:00 | 574.00 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-07-16 09:50:00 | 567.82 | 2024-07-16 10:00:00 | 569.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-22 11:00:00 | 564.75 | 2024-07-22 11:25:00 | 567.48 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-22 11:00:00 | 564.75 | 2024-07-22 11:50:00 | 564.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 09:50:00 | 589.00 | 2024-08-01 10:40:00 | 591.20 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-01 09:50:00 | 589.00 | 2024-08-01 10:50:00 | 589.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-02 11:15:00 | 600.22 | 2024-08-02 11:35:00 | 603.74 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-02 11:15:00 | 600.22 | 2024-08-02 11:55:00 | 600.22 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 11:15:00 | 595.62 | 2024-08-12 11:45:00 | 598.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-08-12 11:15:00 | 595.62 | 2024-08-12 14:55:00 | 596.43 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-08-16 10:25:00 | 590.65 | 2024-08-16 10:30:00 | 588.53 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-21 10:10:00 | 634.10 | 2024-08-21 12:00:00 | 631.64 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-02 11:10:00 | 649.97 | 2024-09-02 12:45:00 | 648.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-16 09:30:00 | 673.25 | 2024-09-16 09:40:00 | 674.74 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-17 09:35:00 | 688.00 | 2024-09-17 09:45:00 | 690.53 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-17 09:35:00 | 688.00 | 2024-09-17 09:50:00 | 688.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 09:30:00 | 706.61 | 2024-09-24 09:45:00 | 708.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-01 11:05:00 | 720.70 | 2024-10-01 11:10:00 | 718.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-03 09:30:00 | 702.92 | 2024-10-03 09:45:00 | 699.52 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-03 09:30:00 | 702.92 | 2024-10-03 15:20:00 | 680.19 | TARGET_HIT | 0.50 | 3.23% |
| SELL | retest1 | 2024-10-10 10:00:00 | 670.32 | 2024-10-10 10:15:00 | 672.79 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-14 09:45:00 | 681.23 | 2024-10-14 10:00:00 | 684.29 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-14 09:45:00 | 681.23 | 2024-10-14 10:10:00 | 681.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:30:00 | 668.20 | 2024-10-17 09:40:00 | 665.45 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 09:30:00 | 668.20 | 2024-10-17 10:30:00 | 668.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 10:35:00 | 619.92 | 2024-11-05 10:55:00 | 616.68 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-05 10:35:00 | 619.92 | 2024-11-05 11:15:00 | 619.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:05:00 | 621.62 | 2024-11-07 10:20:00 | 624.13 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-12 09:30:00 | 598.57 | 2024-11-12 09:45:00 | 600.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-19 10:35:00 | 570.32 | 2024-11-19 10:50:00 | 568.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-28 09:50:00 | 613.06 | 2024-11-28 10:05:00 | 611.48 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-05 10:55:00 | 615.34 | 2024-12-05 11:00:00 | 617.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-12 11:00:00 | 645.94 | 2024-12-12 11:10:00 | 647.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-30 09:45:00 | 584.00 | 2024-12-30 10:05:00 | 582.04 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-01 11:05:00 | 583.38 | 2025-01-01 11:25:00 | 581.88 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-02 10:50:00 | 595.30 | 2025-01-02 11:25:00 | 598.04 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-02 10:50:00 | 595.30 | 2025-01-02 11:45:00 | 595.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:55:00 | 570.26 | 2025-01-09 12:30:00 | 567.37 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-09 10:55:00 | 570.26 | 2025-01-09 15:20:00 | 563.60 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2025-01-29 10:45:00 | 539.80 | 2025-01-29 11:00:00 | 543.44 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-29 10:45:00 | 539.80 | 2025-01-29 11:20:00 | 539.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 10:10:00 | 577.35 | 2025-02-07 10:15:00 | 574.66 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-02-24 09:35:00 | 571.30 | 2025-02-24 09:50:00 | 574.12 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-04 10:45:00 | 622.05 | 2025-03-04 11:40:00 | 625.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-04 10:45:00 | 622.05 | 2025-03-04 15:20:00 | 633.30 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2025-03-06 09:35:00 | 649.00 | 2025-03-06 09:40:00 | 646.32 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-18 09:30:00 | 636.15 | 2025-03-18 09:35:00 | 634.06 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-19 09:45:00 | 647.70 | 2025-03-19 10:00:00 | 651.57 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-03-19 09:45:00 | 647.70 | 2025-03-19 15:20:00 | 667.80 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2025-03-24 10:40:00 | 685.85 | 2025-03-24 11:30:00 | 683.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-04 10:45:00 | 659.90 | 2025-04-04 10:55:00 | 657.31 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-04-23 09:50:00 | 699.55 | 2025-04-23 10:05:00 | 695.19 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-23 09:50:00 | 699.55 | 2025-04-23 11:15:00 | 699.55 | STOP_HIT | 0.50 | 0.00% |

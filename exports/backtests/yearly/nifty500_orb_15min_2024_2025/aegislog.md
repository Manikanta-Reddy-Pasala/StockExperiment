# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-01-03 15:25:00 (12108 bars)
- **Last close:** 810.00
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 32
- **Target hits / Stop hits / Partials:** 3 / 32 / 9
- **Avg / median % per leg:** -0.09% / -0.31%
- **Sum % (uncompounded):** -4.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 2 | 11 | 3 | -0.09% | -1.4% |
| BUY @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 2 | 11 | 3 | -0.09% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 7 | 25.0% | 1 | 21 | 6 | -0.09% | -2.6% |
| SELL @ 2nd Alert (retest1) | 28 | 7 | 25.0% | 1 | 21 | 6 | -0.09% | -2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 44 | 12 | 27.3% | 3 | 32 | 9 | -0.09% | -4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 601.45 | 599.91 | 0.00 | ORB-long ORB[594.55,600.75] vol=2.2x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:50:00 | 605.33 | 604.86 | 0.00 | T1 1.5R @ 605.33 |
| Target hit | 2024-05-16 10:00:00 | 605.10 | 606.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 603.50 | 600.27 | 0.00 | ORB-long ORB[595.05,601.50] vol=1.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-05-17 11:15:00 | 601.41 | 600.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 620.95 | 625.63 | 0.00 | ORB-short ORB[623.20,631.70] vol=1.9x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 624.04 | 625.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 772.60 | 776.55 | 0.00 | ORB-short ORB[774.80,779.80] vol=1.9x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-06-13 09:40:00 | 776.40 | 776.07 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 834.90 | 829.68 | 0.00 | ORB-long ORB[822.00,831.20] vol=3.2x ATR=4.11 |
| Stop hit — per-position SL triggered | 2024-06-18 09:35:00 | 830.79 | 829.69 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 863.25 | 876.11 | 0.00 | ORB-short ORB[874.35,884.50] vol=1.6x ATR=6.19 |
| Stop hit — per-position SL triggered | 2024-07-01 09:35:00 | 869.44 | 874.15 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 869.45 | 874.94 | 0.00 | ORB-short ORB[870.35,882.50] vol=1.8x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-07-04 10:40:00 | 873.94 | 874.46 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:45:00 | 842.10 | 846.56 | 0.00 | ORB-short ORB[842.50,852.00] vol=3.1x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:15:00 | 837.31 | 845.26 | 0.00 | T1 1.5R @ 837.31 |
| Stop hit — per-position SL triggered | 2024-07-23 12:10:00 | 842.10 | 843.88 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:05:00 | 893.75 | 880.94 | 0.00 | ORB-long ORB[871.60,882.70] vol=1.6x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-07-26 15:20:00 | 893.05 | 893.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:45:00 | 880.00 | 884.32 | 0.00 | ORB-short ORB[884.15,889.50] vol=2.8x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-07-30 10:10:00 | 883.83 | 883.74 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:45:00 | 753.45 | 747.82 | 0.00 | ORB-long ORB[737.45,747.30] vol=2.1x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-08-09 10:05:00 | 749.38 | 748.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:00:00 | 777.85 | 780.51 | 0.00 | ORB-short ORB[778.15,788.25] vol=2.3x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-08-20 11:20:00 | 779.82 | 780.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:15:00 | 780.50 | 787.22 | 0.00 | ORB-short ORB[792.00,803.00] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-08-26 11:30:00 | 782.32 | 786.89 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 11:05:00 | 761.55 | 751.12 | 0.00 | ORB-long ORB[745.60,755.70] vol=2.0x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-08-30 11:20:00 | 759.17 | 751.84 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 806.45 | 814.51 | 0.00 | ORB-short ORB[810.70,819.55] vol=1.6x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 811.76 | 815.42 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 11:05:00 | 837.95 | 845.14 | 0.00 | ORB-short ORB[847.00,855.25] vol=2.0x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 12:35:00 | 833.45 | 843.94 | 0.00 | T1 1.5R @ 833.45 |
| Stop hit — per-position SL triggered | 2024-09-12 14:50:00 | 837.95 | 839.34 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 777.30 | 782.54 | 0.00 | ORB-short ORB[780.25,789.00] vol=2.0x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:50:00 | 772.23 | 777.99 | 0.00 | T1 1.5R @ 772.23 |
| Target hit | 2024-09-18 10:30:00 | 771.85 | 771.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2024-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:40:00 | 742.35 | 744.84 | 0.00 | ORB-short ORB[743.05,753.40] vol=2.1x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-09-26 10:05:00 | 744.89 | 744.07 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:15:00 | 677.25 | 680.53 | 0.00 | ORB-short ORB[680.05,686.95] vol=2.2x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-10-14 10:20:00 | 679.99 | 680.42 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:00:00 | 706.45 | 716.04 | 0.00 | ORB-short ORB[718.90,727.50] vol=2.4x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-10-16 12:45:00 | 709.76 | 712.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-10-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 10:40:00 | 733.05 | 726.70 | 0.00 | ORB-long ORB[718.45,727.00] vol=1.6x ATR=5.07 |
| Stop hit — per-position SL triggered | 2024-10-17 14:40:00 | 727.98 | 729.66 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 712.25 | 719.35 | 0.00 | ORB-short ORB[718.25,728.85] vol=2.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-10-21 09:45:00 | 715.42 | 718.31 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 09:30:00 | 738.65 | 732.78 | 0.00 | ORB-long ORB[724.50,733.70] vol=1.7x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-10-23 09:45:00 | 731.40 | 734.27 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 737.20 | 743.13 | 0.00 | ORB-short ORB[743.20,754.00] vol=3.0x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:35:00 | 730.09 | 738.71 | 0.00 | T1 1.5R @ 730.09 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 737.20 | 737.45 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:00:00 | 798.70 | 792.18 | 0.00 | ORB-long ORB[786.55,793.85] vol=2.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-10-31 10:05:00 | 795.87 | 792.51 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-11-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:20:00 | 785.65 | 792.57 | 0.00 | ORB-short ORB[791.15,800.50] vol=2.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-11-08 10:25:00 | 788.53 | 792.00 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 11:10:00 | 814.60 | 818.57 | 0.00 | ORB-short ORB[818.45,828.25] vol=3.1x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:55:00 | 810.98 | 817.30 | 0.00 | T1 1.5R @ 810.98 |
| Stop hit — per-position SL triggered | 2024-11-27 12:35:00 | 814.60 | 816.99 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:05:00 | 841.45 | 843.58 | 0.00 | ORB-short ORB[841.90,854.50] vol=2.0x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 845.44 | 842.40 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 749.80 | 754.47 | 0.00 | ORB-short ORB[753.00,763.60] vol=4.3x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-12-13 09:35:00 | 752.51 | 753.88 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:55:00 | 773.85 | 777.38 | 0.00 | ORB-short ORB[774.00,784.95] vol=2.0x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:25:00 | 769.42 | 776.32 | 0.00 | T1 1.5R @ 769.42 |
| Stop hit — per-position SL triggered | 2024-12-16 13:30:00 | 773.85 | 775.30 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 781.15 | 773.82 | 0.00 | ORB-long ORB[768.05,774.95] vol=4.1x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:55:00 | 785.23 | 778.18 | 0.00 | T1 1.5R @ 785.23 |
| Target hit | 2024-12-17 12:05:00 | 789.10 | 789.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 787.15 | 778.25 | 0.00 | ORB-long ORB[770.00,780.15] vol=2.5x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-12-19 10:45:00 | 782.81 | 784.66 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:15:00 | 818.25 | 814.13 | 0.00 | ORB-long ORB[808.00,815.95] vol=1.9x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-12-30 11:40:00 | 815.64 | 814.67 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:50:00 | 812.70 | 817.74 | 0.00 | ORB-short ORB[816.40,825.85] vol=1.5x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-01-01 09:55:00 | 815.81 | 817.87 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:10:00 | 821.70 | 815.95 | 0.00 | ORB-long ORB[811.30,818.70] vol=3.4x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:50:00 | 825.27 | 818.21 | 0.00 | T1 1.5R @ 825.27 |
| Stop hit — per-position SL triggered | 2025-01-02 12:45:00 | 821.70 | 820.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:35:00 | 601.45 | 2024-05-16 09:50:00 | 605.33 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-05-16 09:35:00 | 601.45 | 2024-05-16 10:00:00 | 605.10 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-05-17 10:55:00 | 603.50 | 2024-05-17 11:15:00 | 601.41 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-24 09:35:00 | 620.95 | 2024-05-24 09:40:00 | 624.04 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-06-13 09:30:00 | 772.60 | 2024-06-13 09:40:00 | 776.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-18 09:30:00 | 834.90 | 2024-06-18 09:35:00 | 830.79 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-07-01 09:30:00 | 863.25 | 2024-07-01 09:35:00 | 869.44 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-07-04 10:25:00 | 869.45 | 2024-07-04 10:40:00 | 873.94 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-07-23 10:45:00 | 842.10 | 2024-07-23 11:15:00 | 837.31 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-07-23 10:45:00 | 842.10 | 2024-07-23 12:10:00 | 842.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:05:00 | 893.75 | 2024-07-26 15:20:00 | 893.05 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest1 | 2024-07-30 09:45:00 | 880.00 | 2024-07-30 10:10:00 | 883.83 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-09 09:45:00 | 753.45 | 2024-08-09 10:05:00 | 749.38 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-08-20 11:00:00 | 777.85 | 2024-08-20 11:20:00 | 779.82 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-26 11:15:00 | 780.50 | 2024-08-26 11:30:00 | 782.32 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-30 11:05:00 | 761.55 | 2024-08-30 11:20:00 | 759.17 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-09 09:30:00 | 806.45 | 2024-09-09 09:35:00 | 811.76 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2024-09-12 11:05:00 | 837.95 | 2024-09-12 12:35:00 | 833.45 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-12 11:05:00 | 837.95 | 2024-09-12 14:50:00 | 837.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 09:30:00 | 777.30 | 2024-09-18 09:50:00 | 772.23 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-09-18 09:30:00 | 777.30 | 2024-09-18 10:30:00 | 771.85 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2024-09-26 09:40:00 | 742.35 | 2024-09-26 10:05:00 | 744.89 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-14 10:15:00 | 677.25 | 2024-10-14 10:20:00 | 679.99 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-16 11:00:00 | 706.45 | 2024-10-16 12:45:00 | 709.76 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-17 10:40:00 | 733.05 | 2024-10-17 14:40:00 | 727.98 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2024-10-21 09:35:00 | 712.25 | 2024-10-21 09:45:00 | 715.42 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-23 09:30:00 | 738.65 | 2024-10-23 09:45:00 | 731.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest1 | 2024-10-25 09:30:00 | 737.20 | 2024-10-25 10:35:00 | 730.09 | PARTIAL | 0.50 | 0.96% |
| SELL | retest1 | 2024-10-25 09:30:00 | 737.20 | 2024-10-25 10:55:00 | 737.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 10:00:00 | 798.70 | 2024-10-31 10:05:00 | 795.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-08 10:20:00 | 785.65 | 2024-11-08 10:25:00 | 788.53 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-27 11:10:00 | 814.60 | 2024-11-27 11:55:00 | 810.98 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-27 11:10:00 | 814.60 | 2024-11-27 12:35:00 | 814.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:05:00 | 841.45 | 2024-12-05 12:05:00 | 845.44 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-13 09:30:00 | 749.80 | 2024-12-13 09:35:00 | 752.51 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-16 10:55:00 | 773.85 | 2024-12-16 11:25:00 | 769.42 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-16 10:55:00 | 773.85 | 2024-12-16 13:30:00 | 773.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 09:45:00 | 781.15 | 2024-12-17 09:55:00 | 785.23 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-17 09:45:00 | 781.15 | 2024-12-17 12:05:00 | 789.10 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-12-19 09:40:00 | 787.15 | 2024-12-19 10:45:00 | 782.81 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-12-30 11:15:00 | 818.25 | 2024-12-30 11:40:00 | 815.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-01 09:50:00 | 812.70 | 2025-01-01 09:55:00 | 815.81 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-02 11:10:00 | 821.70 | 2025-01-02 11:50:00 | 825.27 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-02 11:10:00 | 821.70 | 2025-01-02 12:45:00 | 821.70 | STOP_HIT | 0.50 | 0.00% |

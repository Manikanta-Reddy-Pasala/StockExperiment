# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-04-24 15:25:00 (53105 bars)
- **Last close:** 785.15
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 15 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 63
- **Target hits / Stop hits / Partials:** 15 / 63 / 28
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 10.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 27 | 49.1% | 10 | 28 | 17 | 0.17% | 9.3% |
| BUY @ 2nd Alert (retest1) | 55 | 27 | 49.1% | 10 | 28 | 17 | 0.17% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 16 | 31.4% | 5 | 35 | 11 | 0.02% | 0.9% |
| SELL @ 2nd Alert (retest1) | 51 | 16 | 31.4% | 5 | 35 | 11 | 0.02% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 43 | 40.6% | 15 | 63 | 28 | 0.10% | 10.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:35:00 | 817.13 | 820.92 | 0.00 | ORB-short ORB[821.55,824.43] vol=1.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-05-19 11:15:00 | 818.72 | 820.30 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 10:20:00 | 808.13 | 811.64 | 0.00 | ORB-short ORB[811.60,817.48] vol=1.6x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:35:00 | 805.78 | 809.55 | 0.00 | T1 1.5R @ 805.78 |
| Target hit | 2023-05-31 15:20:00 | 805.65 | 806.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2023-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:15:00 | 803.73 | 803.99 | 0.00 | ORB-short ORB[803.93,807.40] vol=2.1x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-06-02 11:35:00 | 804.81 | 803.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:25:00 | 800.93 | 801.74 | 0.00 | ORB-short ORB[801.25,804.93] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:55:00 | 799.52 | 801.31 | 0.00 | T1 1.5R @ 799.52 |
| Target hit | 2023-06-06 15:05:00 | 799.80 | 798.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2023-06-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:40:00 | 806.98 | 806.36 | 0.00 | ORB-long ORB[801.63,806.88] vol=1.5x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 09:50:00 | 808.57 | 806.67 | 0.00 | T1 1.5R @ 808.57 |
| Target hit | 2023-06-08 11:50:00 | 809.75 | 809.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2023-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 10:40:00 | 802.50 | 803.82 | 0.00 | ORB-short ORB[804.05,807.90] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-06-12 11:10:00 | 803.49 | 803.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 09:50:00 | 811.85 | 810.51 | 0.00 | ORB-long ORB[805.50,811.45] vol=1.6x ATR=1.60 |
| Stop hit — per-position SL triggered | 2023-06-21 10:15:00 | 810.25 | 810.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:30:00 | 823.65 | 822.45 | 0.00 | ORB-long ORB[819.53,823.28] vol=3.1x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-06-22 11:00:00 | 822.14 | 822.54 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 09:50:00 | 877.30 | 866.80 | 0.00 | ORB-long ORB[855.00,863.85] vol=2.3x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-07-03 09:55:00 | 874.76 | 868.22 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-07-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 10:20:00 | 831.45 | 832.55 | 0.00 | ORB-short ORB[831.80,836.50] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2023-07-06 10:25:00 | 832.77 | 832.63 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 11:00:00 | 833.63 | 831.14 | 0.00 | ORB-long ORB[829.40,833.30] vol=2.2x ATR=1.29 |
| Stop hit — per-position SL triggered | 2023-07-11 11:05:00 | 832.34 | 831.32 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 11:15:00 | 828.50 | 827.90 | 0.00 | ORB-long ORB[822.20,827.98] vol=2.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-07-12 11:20:00 | 826.98 | 827.88 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 11:00:00 | 820.08 | 822.84 | 0.00 | ORB-short ORB[821.10,826.80] vol=1.7x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-07-14 12:05:00 | 821.27 | 821.87 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:15:00 | 839.68 | 842.00 | 0.00 | ORB-short ORB[840.00,844.33] vol=2.1x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 12:10:00 | 837.65 | 841.57 | 0.00 | T1 1.5R @ 837.65 |
| Stop hit — per-position SL triggered | 2023-07-19 12:20:00 | 839.68 | 841.31 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:50:00 | 841.20 | 840.54 | 0.00 | ORB-long ORB[838.30,841.10] vol=3.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-07-24 11:10:00 | 840.08 | 840.56 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:30:00 | 846.13 | 843.40 | 0.00 | ORB-long ORB[839.20,844.10] vol=2.1x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-07-25 09:35:00 | 844.67 | 843.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:45:00 | 846.13 | 848.80 | 0.00 | ORB-short ORB[847.05,851.50] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-07-27 10:50:00 | 847.46 | 848.56 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-08-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 11:00:00 | 819.93 | 817.45 | 0.00 | ORB-long ORB[814.63,818.98] vol=2.1x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 11:25:00 | 822.04 | 818.27 | 0.00 | T1 1.5R @ 822.04 |
| Stop hit — per-position SL triggered | 2023-08-04 13:35:00 | 819.93 | 820.01 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:45:00 | 825.18 | 827.81 | 0.00 | ORB-short ORB[826.50,831.55] vol=2.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2023-08-07 11:45:00 | 826.90 | 827.05 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:30:00 | 820.38 | 822.28 | 0.00 | ORB-short ORB[820.95,826.50] vol=2.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 11:00:00 | 818.56 | 821.38 | 0.00 | T1 1.5R @ 818.56 |
| Target hit | 2023-08-09 13:00:00 | 819.55 | 819.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2023-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:55:00 | 812.60 | 814.86 | 0.00 | ORB-short ORB[814.75,819.20] vol=1.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-08-11 10:40:00 | 814.03 | 814.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 11:05:00 | 804.33 | 801.52 | 0.00 | ORB-long ORB[798.80,802.98] vol=2.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-08-17 11:15:00 | 803.19 | 801.75 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:00:00 | 793.73 | 795.96 | 0.00 | ORB-short ORB[796.33,799.00] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-08-22 12:05:00 | 794.76 | 795.37 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:55:00 | 780.65 | 784.61 | 0.00 | ORB-short ORB[783.58,788.75] vol=1.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-08-25 11:05:00 | 782.09 | 784.08 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:50:00 | 783.80 | 783.42 | 0.00 | ORB-long ORB[780.75,783.53] vol=2.2x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 10:25:00 | 785.62 | 783.87 | 0.00 | T1 1.5R @ 785.62 |
| Target hit | 2023-08-28 15:20:00 | 789.15 | 787.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-09-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:10:00 | 784.65 | 785.32 | 0.00 | ORB-short ORB[785.03,789.30] vol=2.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 11:35:00 | 782.39 | 785.07 | 0.00 | T1 1.5R @ 782.39 |
| Stop hit — per-position SL triggered | 2023-09-01 12:45:00 | 784.65 | 783.89 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:55:00 | 788.13 | 790.65 | 0.00 | ORB-short ORB[790.60,795.73] vol=1.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-09-04 11:00:00 | 789.53 | 790.55 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 789.43 | 790.33 | 0.00 | ORB-short ORB[789.53,794.20] vol=1.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 11:00:00 | 787.95 | 789.79 | 0.00 | T1 1.5R @ 787.95 |
| Stop hit — per-position SL triggered | 2023-09-05 11:50:00 | 789.43 | 789.53 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:30:00 | 792.08 | 790.90 | 0.00 | ORB-long ORB[787.45,791.85] vol=1.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 11:50:00 | 794.00 | 792.04 | 0.00 | T1 1.5R @ 794.00 |
| Stop hit — per-position SL triggered | 2023-09-06 12:35:00 | 792.08 | 792.21 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 11:10:00 | 801.38 | 797.09 | 0.00 | ORB-long ORB[793.58,799.70] vol=1.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-09-07 11:30:00 | 800.07 | 797.58 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 10:55:00 | 821.35 | 818.71 | 0.00 | ORB-long ORB[814.30,817.00] vol=1.6x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 12:55:00 | 823.34 | 820.39 | 0.00 | T1 1.5R @ 823.34 |
| Stop hit — per-position SL triggered | 2023-09-13 15:05:00 | 821.35 | 822.03 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 11:00:00 | 831.55 | 828.78 | 0.00 | ORB-long ORB[825.00,829.68] vol=2.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-09-15 11:20:00 | 830.36 | 829.15 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:05:00 | 774.30 | 776.99 | 0.00 | ORB-short ORB[775.00,782.20] vol=1.5x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-09-22 10:40:00 | 776.15 | 776.39 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:00:00 | 758.38 | 762.76 | 0.00 | ORB-short ORB[761.50,764.85] vol=1.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-09-27 10:35:00 | 759.64 | 761.44 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:50:00 | 765.45 | 764.92 | 0.00 | ORB-long ORB[761.85,764.95] vol=9.2x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-09-29 12:50:00 | 764.14 | 765.17 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:35:00 | 757.38 | 751.57 | 0.00 | ORB-long ORB[744.63,753.83] vol=1.9x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 10:30:00 | 760.40 | 755.98 | 0.00 | T1 1.5R @ 760.40 |
| Target hit | 2023-10-04 15:20:00 | 767.33 | 759.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2023-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 09:50:00 | 758.00 | 760.36 | 0.00 | ORB-short ORB[760.28,763.50] vol=3.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-10-10 10:05:00 | 758.98 | 760.02 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 09:50:00 | 769.25 | 772.10 | 0.00 | ORB-short ORB[770.43,777.88] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-10-17 10:55:00 | 770.83 | 771.17 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:40:00 | 764.10 | 766.27 | 0.00 | ORB-short ORB[765.00,770.60] vol=2.2x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:00:00 | 762.37 | 765.88 | 0.00 | T1 1.5R @ 762.37 |
| Target hit | 2023-10-18 15:20:00 | 759.68 | 762.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-10-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 11:00:00 | 759.28 | 757.35 | 0.00 | ORB-long ORB[752.50,757.48] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-10-20 11:55:00 | 758.26 | 757.84 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:50:00 | 760.03 | 761.45 | 0.00 | ORB-short ORB[761.40,764.38] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-10-23 11:00:00 | 761.05 | 761.38 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 10:50:00 | 743.35 | 744.28 | 0.00 | ORB-short ORB[743.55,749.00] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:00:00 | 741.92 | 744.18 | 0.00 | T1 1.5R @ 741.92 |
| Stop hit — per-position SL triggered | 2023-11-06 11:30:00 | 743.35 | 743.91 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 10:00:00 | 741.48 | 742.47 | 0.00 | ORB-short ORB[742.20,747.00] vol=2.4x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-11-07 10:35:00 | 742.41 | 742.28 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:05:00 | 755.03 | 754.57 | 0.00 | ORB-long ORB[752.70,755.00] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:30:00 | 756.08 | 754.71 | 0.00 | T1 1.5R @ 756.08 |
| Stop hit — per-position SL triggered | 2023-11-16 15:00:00 | 755.03 | 755.82 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:10:00 | 751.25 | 753.04 | 0.00 | ORB-short ORB[751.35,756.50] vol=1.5x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-11-20 12:15:00 | 752.26 | 752.72 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:35:00 | 770.33 | 768.93 | 0.00 | ORB-long ORB[766.50,770.18] vol=2.2x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:45:00 | 772.02 | 769.77 | 0.00 | T1 1.5R @ 772.02 |
| Target hit | 2023-11-29 15:20:00 | 780.50 | 773.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 811.80 | 815.09 | 0.00 | ORB-short ORB[813.23,819.23] vol=1.5x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:35:00 | 809.33 | 814.42 | 0.00 | T1 1.5R @ 809.33 |
| Stop hit — per-position SL triggered | 2023-12-06 09:50:00 | 811.80 | 813.31 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 825.15 | 822.22 | 0.00 | ORB-long ORB[815.23,821.50] vol=1.6x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:20:00 | 827.02 | 822.88 | 0.00 | T1 1.5R @ 827.02 |
| Stop hit — per-position SL triggered | 2023-12-08 11:55:00 | 825.15 | 823.43 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:55:00 | 835.00 | 831.07 | 0.00 | ORB-long ORB[825.13,833.13] vol=1.9x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-21 11:00:00 | 837.72 | 831.45 | 0.00 | T1 1.5R @ 837.72 |
| Target hit | 2023-12-21 15:20:00 | 843.75 | 836.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2023-12-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:05:00 | 844.30 | 841.89 | 0.00 | ORB-long ORB[839.30,842.58] vol=1.7x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:55:00 | 846.01 | 843.46 | 0.00 | T1 1.5R @ 846.01 |
| Target hit | 2023-12-27 15:20:00 | 852.50 | 847.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 845.78 | 848.05 | 0.00 | ORB-short ORB[846.55,851.38] vol=3.3x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-01-02 10:25:00 | 847.46 | 847.62 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 11:15:00 | 845.15 | 839.20 | 0.00 | ORB-long ORB[835.35,839.00] vol=1.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-01-04 11:20:00 | 843.86 | 839.27 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:40:00 | 840.00 | 843.76 | 0.00 | ORB-short ORB[841.15,852.45] vol=1.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-01-05 10:05:00 | 842.02 | 843.11 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 837.43 | 838.80 | 0.00 | ORB-short ORB[838.15,841.18] vol=3.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-01-08 11:15:00 | 838.77 | 838.76 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 11:05:00 | 826.00 | 824.85 | 0.00 | ORB-long ORB[820.70,824.15] vol=1.6x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 13:10:00 | 828.15 | 825.61 | 0.00 | T1 1.5R @ 828.15 |
| Target hit | 2024-01-10 15:20:00 | 828.03 | 826.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2024-01-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:05:00 | 826.48 | 827.94 | 0.00 | ORB-short ORB[828.53,831.45] vol=3.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-01-11 10:15:00 | 827.77 | 827.88 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:30:00 | 730.03 | 732.83 | 0.00 | ORB-short ORB[731.25,737.35] vol=3.8x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:55:00 | 727.14 | 730.71 | 0.00 | T1 1.5R @ 727.14 |
| Target hit | 2024-01-23 15:20:00 | 714.50 | 721.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2024-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:45:00 | 727.48 | 724.57 | 0.00 | ORB-long ORB[718.40,724.90] vol=6.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 10:05:00 | 730.90 | 725.49 | 0.00 | T1 1.5R @ 730.90 |
| Target hit | 2024-01-31 15:05:00 | 731.45 | 731.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 734.28 | 736.28 | 0.00 | ORB-short ORB[734.80,740.43] vol=2.0x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-02-02 11:15:00 | 735.75 | 736.18 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-02-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 11:10:00 | 717.83 | 719.40 | 0.00 | ORB-short ORB[718.48,723.50] vol=1.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-02-06 11:30:00 | 718.91 | 719.22 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 705.75 | 715.07 | 0.00 | ORB-short ORB[714.30,718.05] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-02-08 11:50:00 | 707.54 | 712.50 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-09 10:55:00 | 703.80 | 700.55 | 0.00 | ORB-long ORB[693.50,700.78] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-02-09 11:05:00 | 701.72 | 700.74 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 11:05:00 | 698.68 | 696.12 | 0.00 | ORB-long ORB[692.18,697.43] vol=1.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-02-13 11:30:00 | 697.09 | 696.29 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:35:00 | 683.80 | 686.19 | 0.00 | ORB-short ORB[684.70,691.50] vol=1.6x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-02-14 10:00:00 | 685.93 | 685.70 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:05:00 | 716.98 | 712.25 | 0.00 | ORB-long ORB[705.90,712.10] vol=1.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:20:00 | 719.52 | 714.35 | 0.00 | T1 1.5R @ 719.52 |
| Target hit | 2024-02-20 15:20:00 | 726.53 | 721.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 706.03 | 710.25 | 0.00 | ORB-short ORB[708.75,712.08] vol=2.5x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:00:00 | 704.24 | 709.40 | 0.00 | T1 1.5R @ 704.24 |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 706.03 | 709.01 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-03-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 09:45:00 | 722.50 | 723.38 | 0.00 | ORB-short ORB[722.75,725.53] vol=2.1x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-03-07 15:20:00 | 723.05 | 722.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 720.15 | 717.24 | 0.00 | ORB-long ORB[714.63,717.45] vol=1.9x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:35:00 | 722.47 | 719.12 | 0.00 | T1 1.5R @ 722.47 |
| Target hit | 2024-03-12 10:40:00 | 725.03 | 725.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:15:00 | 720.35 | 722.51 | 0.00 | ORB-short ORB[723.05,726.00] vol=3.7x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-03-18 11:20:00 | 721.97 | 722.37 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:45:00 | 712.83 | 720.76 | 0.00 | ORB-short ORB[723.03,725.83] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-03-20 10:50:00 | 714.50 | 720.14 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:20:00 | 724.33 | 722.47 | 0.00 | ORB-long ORB[719.15,723.73] vol=1.6x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-03-21 11:40:00 | 722.61 | 723.00 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:50:00 | 716.80 | 715.69 | 0.00 | ORB-long ORB[710.63,714.95] vol=1.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-03-27 11:10:00 | 715.76 | 715.76 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:30:00 | 742.53 | 736.98 | 0.00 | ORB-long ORB[731.63,738.73] vol=2.9x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 09:35:00 | 745.13 | 738.55 | 0.00 | T1 1.5R @ 745.13 |
| Stop hit — per-position SL triggered | 2024-04-02 10:25:00 | 742.53 | 741.47 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 11:15:00 | 746.25 | 741.13 | 0.00 | ORB-long ORB[735.70,740.35] vol=2.1x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-04-03 12:20:00 | 745.05 | 742.51 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-04-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 11:05:00 | 773.48 | 770.21 | 0.00 | ORB-long ORB[765.08,770.83] vol=1.6x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 11:25:00 | 775.29 | 770.82 | 0.00 | T1 1.5R @ 775.29 |
| Stop hit — per-position SL triggered | 2024-04-05 11:30:00 | 773.48 | 770.89 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-04-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:45:00 | 762.80 | 760.87 | 0.00 | ORB-long ORB[756.18,760.50] vol=2.2x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-04-26 09:55:00 | 761.38 | 760.97 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 11:15:00 | 759.73 | 756.52 | 0.00 | ORB-long ORB[755.00,758.13] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-04-29 11:40:00 | 758.51 | 756.96 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:50:00 | 768.98 | 765.40 | 0.00 | ORB-long ORB[761.50,765.40] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-04-30 10:55:00 | 767.27 | 765.57 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 10:35:00 | 817.13 | 2023-05-19 11:15:00 | 818.72 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-05-31 10:20:00 | 808.13 | 2023-05-31 11:35:00 | 805.78 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-05-31 10:20:00 | 808.13 | 2023-05-31 15:20:00 | 805.65 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2023-06-02 11:15:00 | 803.73 | 2023-06-02 11:35:00 | 804.81 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-06-06 10:25:00 | 800.93 | 2023-06-06 10:55:00 | 799.52 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2023-06-06 10:25:00 | 800.93 | 2023-06-06 15:05:00 | 799.80 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-06-08 09:40:00 | 806.98 | 2023-06-08 09:50:00 | 808.57 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-08 09:40:00 | 806.98 | 2023-06-08 11:50:00 | 809.75 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-12 10:40:00 | 802.50 | 2023-06-12 11:10:00 | 803.49 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-06-21 09:50:00 | 811.85 | 2023-06-21 10:15:00 | 810.25 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-22 10:30:00 | 823.65 | 2023-06-22 11:00:00 | 822.14 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-03 09:50:00 | 877.30 | 2023-07-03 09:55:00 | 874.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-06 10:20:00 | 831.45 | 2023-07-06 10:25:00 | 832.77 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-07-11 11:00:00 | 833.63 | 2023-07-11 11:05:00 | 832.34 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-07-12 11:15:00 | 828.50 | 2023-07-12 11:20:00 | 826.98 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-14 11:00:00 | 820.08 | 2023-07-14 12:05:00 | 821.27 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-07-19 11:15:00 | 839.68 | 2023-07-19 12:10:00 | 837.65 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-07-19 11:15:00 | 839.68 | 2023-07-19 12:20:00 | 839.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-24 10:50:00 | 841.20 | 2023-07-24 11:10:00 | 840.08 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-07-25 09:30:00 | 846.13 | 2023-07-25 09:35:00 | 844.67 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-27 10:45:00 | 846.13 | 2023-07-27 10:50:00 | 847.46 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-08-04 11:00:00 | 819.93 | 2023-08-04 11:25:00 | 822.04 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-08-04 11:00:00 | 819.93 | 2023-08-04 13:35:00 | 819.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-07 10:45:00 | 825.18 | 2023-08-07 11:45:00 | 826.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-09 10:30:00 | 820.38 | 2023-08-09 11:00:00 | 818.56 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-09 10:30:00 | 820.38 | 2023-08-09 13:00:00 | 819.55 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-08-11 09:55:00 | 812.60 | 2023-08-11 10:40:00 | 814.03 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-08-17 11:05:00 | 804.33 | 2023-08-17 11:15:00 | 803.19 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-22 11:00:00 | 793.73 | 2023-08-22 12:05:00 | 794.76 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-08-25 10:55:00 | 780.65 | 2023-08-25 11:05:00 | 782.09 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-08-28 09:50:00 | 783.80 | 2023-08-28 10:25:00 | 785.62 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-28 09:50:00 | 783.80 | 2023-08-28 15:20:00 | 789.15 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2023-09-01 11:10:00 | 784.65 | 2023-09-01 11:35:00 | 782.39 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-09-01 11:10:00 | 784.65 | 2023-09-01 12:45:00 | 784.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-04 10:55:00 | 788.13 | 2023-09-04 11:00:00 | 789.53 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-09-05 10:15:00 | 789.43 | 2023-09-05 11:00:00 | 787.95 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2023-09-05 10:15:00 | 789.43 | 2023-09-05 11:50:00 | 789.43 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-06 09:30:00 | 792.08 | 2023-09-06 11:50:00 | 794.00 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-09-06 09:30:00 | 792.08 | 2023-09-06 12:35:00 | 792.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 11:10:00 | 801.38 | 2023-09-07 11:30:00 | 800.07 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-13 10:55:00 | 821.35 | 2023-09-13 12:55:00 | 823.34 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-09-13 10:55:00 | 821.35 | 2023-09-13 15:05:00 | 821.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 11:00:00 | 831.55 | 2023-09-15 11:20:00 | 830.36 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-09-22 10:05:00 | 774.30 | 2023-09-22 10:40:00 | 776.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-09-27 10:00:00 | 758.38 | 2023-09-27 10:35:00 | 759.64 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-29 10:50:00 | 765.45 | 2023-09-29 12:50:00 | 764.14 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-04 09:35:00 | 757.38 | 2023-10-04 10:30:00 | 760.40 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-10-04 09:35:00 | 757.38 | 2023-10-04 15:20:00 | 767.33 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2023-10-10 09:50:00 | 758.00 | 2023-10-10 10:05:00 | 758.98 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-10-17 09:50:00 | 769.25 | 2023-10-17 10:55:00 | 770.83 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-18 10:40:00 | 764.10 | 2023-10-18 11:00:00 | 762.37 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-10-18 10:40:00 | 764.10 | 2023-10-18 15:20:00 | 759.68 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2023-10-20 11:00:00 | 759.28 | 2023-10-20 11:55:00 | 758.26 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-10-23 10:50:00 | 760.03 | 2023-10-23 11:00:00 | 761.05 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-11-06 10:50:00 | 743.35 | 2023-11-06 11:00:00 | 741.92 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2023-11-06 10:50:00 | 743.35 | 2023-11-06 11:30:00 | 743.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-07 10:00:00 | 741.48 | 2023-11-07 10:35:00 | 742.41 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-11-16 11:05:00 | 755.03 | 2023-11-16 11:30:00 | 756.08 | PARTIAL | 0.50 | 0.14% |
| BUY | retest1 | 2023-11-16 11:05:00 | 755.03 | 2023-11-16 15:00:00 | 755.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 11:10:00 | 751.25 | 2023-11-20 12:15:00 | 752.26 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-11-29 09:35:00 | 770.33 | 2023-11-29 10:45:00 | 772.02 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-11-29 09:35:00 | 770.33 | 2023-11-29 15:20:00 | 780.50 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2023-12-06 09:30:00 | 811.80 | 2023-12-06 09:35:00 | 809.33 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-06 09:30:00 | 811.80 | 2023-12-06 09:50:00 | 811.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-08 11:00:00 | 825.15 | 2023-12-08 11:20:00 | 827.02 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-12-08 11:00:00 | 825.15 | 2023-12-08 11:55:00 | 825.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-21 10:55:00 | 835.00 | 2023-12-21 11:00:00 | 837.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-21 10:55:00 | 835.00 | 2023-12-21 15:20:00 | 843.75 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2023-12-27 10:05:00 | 844.30 | 2023-12-27 10:55:00 | 846.01 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-12-27 10:05:00 | 844.30 | 2023-12-27 15:20:00 | 852.50 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-01-02 09:55:00 | 845.78 | 2024-01-02 10:25:00 | 847.46 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-01-04 11:15:00 | 845.15 | 2024-01-04 11:20:00 | 843.86 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-01-05 09:40:00 | 840.00 | 2024-01-05 10:05:00 | 842.02 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-08 11:05:00 | 837.43 | 2024-01-08 11:15:00 | 838.77 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-01-10 11:05:00 | 826.00 | 2024-01-10 13:10:00 | 828.15 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-01-10 11:05:00 | 826.00 | 2024-01-10 15:20:00 | 828.03 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-01-11 10:05:00 | 826.48 | 2024-01-11 10:15:00 | 827.77 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-01-23 09:30:00 | 730.03 | 2024-01-23 09:55:00 | 727.14 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-01-23 09:30:00 | 730.03 | 2024-01-23 15:20:00 | 714.50 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2024-01-31 09:45:00 | 727.48 | 2024-01-31 10:05:00 | 730.90 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-01-31 09:45:00 | 727.48 | 2024-01-31 15:05:00 | 731.45 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-02-02 11:10:00 | 734.28 | 2024-02-02 11:15:00 | 735.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-02-06 11:10:00 | 717.83 | 2024-02-06 11:30:00 | 718.91 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-02-08 11:00:00 | 705.75 | 2024-02-08 11:50:00 | 707.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-09 10:55:00 | 703.80 | 2024-02-09 11:05:00 | 701.72 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-13 11:05:00 | 698.68 | 2024-02-13 11:30:00 | 697.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-14 09:35:00 | 683.80 | 2024-02-14 10:00:00 | 685.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-02-20 10:05:00 | 716.98 | 2024-02-20 10:20:00 | 719.52 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-02-20 10:05:00 | 716.98 | 2024-02-20 15:20:00 | 726.53 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2024-02-28 10:55:00 | 706.03 | 2024-02-28 11:00:00 | 704.24 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-28 10:55:00 | 706.03 | 2024-02-28 11:15:00 | 706.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-07 09:45:00 | 722.50 | 2024-03-07 15:20:00 | 723.05 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2024-03-12 09:30:00 | 720.15 | 2024-03-12 09:35:00 | 722.47 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-12 09:30:00 | 720.15 | 2024-03-12 10:40:00 | 725.03 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-03-18 11:15:00 | 720.35 | 2024-03-18 11:20:00 | 721.97 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-03-20 10:45:00 | 712.83 | 2024-03-20 10:50:00 | 714.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-21 10:20:00 | 724.33 | 2024-03-21 11:40:00 | 722.61 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-03-27 10:50:00 | 716.80 | 2024-03-27 11:10:00 | 715.76 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-04-02 09:30:00 | 742.53 | 2024-04-02 09:35:00 | 745.13 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-04-02 09:30:00 | 742.53 | 2024-04-02 10:25:00 | 742.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 11:15:00 | 746.25 | 2024-04-03 12:20:00 | 745.05 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-04-05 11:05:00 | 773.48 | 2024-04-05 11:25:00 | 775.29 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2024-04-05 11:05:00 | 773.48 | 2024-04-05 11:30:00 | 773.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 09:45:00 | 762.80 | 2024-04-26 09:55:00 | 761.38 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-04-29 11:15:00 | 759.73 | 2024-04-29 11:40:00 | 758.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-04-30 10:50:00 | 768.98 | 2024-04-30 10:55:00 | 767.27 | STOP_HIT | 1.00 | -0.22% |

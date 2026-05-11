# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 533.75
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 12 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 84
- **Target hits / Stop hits / Partials:** 12 / 84 / 34
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 5.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 18 | 31.6% | 5 | 39 | 13 | 0.00% | 0.1% |
| BUY @ 2nd Alert (retest1) | 57 | 18 | 31.6% | 5 | 39 | 13 | 0.00% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 28 | 38.4% | 7 | 45 | 21 | 0.07% | 5.0% |
| SELL @ 2nd Alert (retest1) | 73 | 28 | 38.4% | 7 | 45 | 21 | 0.07% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 46 | 35.4% | 12 | 84 | 34 | 0.04% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:40:00 | 829.60 | 824.20 | 0.00 | ORB-long ORB[817.88,825.56] vol=1.6x ATR=3.00 |
| Stop hit — per-position SL triggered | 2024-05-15 09:55:00 | 826.60 | 825.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:45:00 | 831.20 | 827.69 | 0.00 | ORB-long ORB[823.28,829.32] vol=1.9x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-05-16 09:55:00 | 828.14 | 828.09 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 860.44 | 868.65 | 0.00 | ORB-short ORB[867.08,880.00] vol=2.2x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 864.74 | 866.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 875.20 | 869.58 | 0.00 | ORB-long ORB[861.44,874.08] vol=1.7x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-05-23 09:45:00 | 871.70 | 870.29 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:55:00 | 894.80 | 884.64 | 0.00 | ORB-long ORB[880.08,887.96] vol=4.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-05-24 11:00:00 | 891.76 | 886.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:50:00 | 881.40 | 884.06 | 0.00 | ORB-short ORB[883.20,889.04] vol=1.7x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:10:00 | 875.76 | 882.76 | 0.00 | T1 1.5R @ 875.76 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 881.40 | 882.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:00:00 | 858.56 | 862.44 | 0.00 | ORB-short ORB[860.44,869.36] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-05-30 11:30:00 | 861.12 | 861.44 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 852.68 | 856.75 | 0.00 | ORB-short ORB[854.28,861.08] vol=1.5x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:55:00 | 848.18 | 855.52 | 0.00 | T1 1.5R @ 848.18 |
| Target hit | 2024-05-31 13:20:00 | 850.32 | 846.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2024-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:45:00 | 885.76 | 879.95 | 0.00 | ORB-long ORB[871.72,882.04] vol=1.6x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:55:00 | 889.65 | 882.96 | 0.00 | T1 1.5R @ 889.65 |
| Target hit | 2024-06-12 15:20:00 | 911.84 | 908.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-06-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:35:00 | 916.36 | 910.97 | 0.00 | ORB-long ORB[905.72,913.60] vol=2.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-06-14 10:50:00 | 913.58 | 911.70 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 824.00 | 828.50 | 0.00 | ORB-short ORB[828.04,833.24] vol=1.8x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-07-02 10:00:00 | 826.28 | 828.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 830.08 | 825.05 | 0.00 | ORB-long ORB[819.24,826.04] vol=2.6x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 09:40:00 | 834.07 | 827.81 | 0.00 | T1 1.5R @ 834.07 |
| Stop hit — per-position SL triggered | 2024-07-04 09:45:00 | 830.08 | 829.96 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:45:00 | 832.08 | 826.93 | 0.00 | ORB-long ORB[821.72,828.00] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 829.36 | 827.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:00:00 | 842.00 | 846.92 | 0.00 | ORB-short ORB[844.00,851.76] vol=2.0x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:25:00 | 836.62 | 845.21 | 0.00 | T1 1.5R @ 836.62 |
| Target hit | 2024-07-08 15:20:00 | 833.80 | 837.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 818.40 | 828.10 | 0.00 | ORB-short ORB[832.04,838.76] vol=2.8x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:40:00 | 813.52 | 823.39 | 0.00 | T1 1.5R @ 813.52 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 818.40 | 823.19 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:15:00 | 840.84 | 844.54 | 0.00 | ORB-short ORB[842.16,853.44] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:25:00 | 837.06 | 844.13 | 0.00 | T1 1.5R @ 837.06 |
| Stop hit — per-position SL triggered | 2024-07-12 12:45:00 | 840.84 | 840.40 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 843.24 | 849.74 | 0.00 | ORB-short ORB[849.80,860.00] vol=2.3x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-07-16 13:40:00 | 845.19 | 847.39 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:45:00 | 836.44 | 841.18 | 0.00 | ORB-short ORB[839.44,850.40] vol=2.4x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-07-18 09:55:00 | 839.48 | 840.76 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:30:00 | 822.52 | 816.58 | 0.00 | ORB-long ORB[806.68,813.32] vol=8.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-07-25 09:35:00 | 818.91 | 817.57 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 819.36 | 818.75 | 0.00 | ORB-long ORB[812.80,818.68] vol=4.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-07-26 11:30:00 | 817.60 | 818.70 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:05:00 | 843.12 | 841.11 | 0.00 | ORB-long ORB[833.00,839.96] vol=1.5x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-07-29 12:00:00 | 840.85 | 841.57 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:20:00 | 841.04 | 834.90 | 0.00 | ORB-long ORB[830.44,837.12] vol=1.8x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-07-30 12:00:00 | 838.16 | 838.90 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:40:00 | 835.76 | 840.33 | 0.00 | ORB-short ORB[837.60,847.36] vol=1.9x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-07-31 10:45:00 | 837.64 | 839.90 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 829.96 | 831.09 | 0.00 | ORB-short ORB[830.92,835.96] vol=2.3x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:40:00 | 827.25 | 830.89 | 0.00 | T1 1.5R @ 827.25 |
| Stop hit — per-position SL triggered | 2024-08-01 13:00:00 | 829.96 | 830.02 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:50:00 | 768.24 | 776.28 | 0.00 | ORB-short ORB[774.28,783.20] vol=1.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-08-12 10:05:00 | 771.17 | 773.83 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:10:00 | 778.36 | 780.56 | 0.00 | ORB-short ORB[782.44,788.76] vol=1.8x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:20:00 | 775.50 | 779.52 | 0.00 | T1 1.5R @ 775.50 |
| Target hit | 2024-08-13 15:20:00 | 765.20 | 775.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 755.60 | 762.06 | 0.00 | ORB-short ORB[761.84,770.40] vol=2.4x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-08-14 11:10:00 | 758.13 | 760.70 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:05:00 | 768.72 | 767.24 | 0.00 | ORB-long ORB[763.36,768.08] vol=1.5x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-08-16 10:30:00 | 766.66 | 767.39 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:15:00 | 789.64 | 787.19 | 0.00 | ORB-long ORB[778.60,788.40] vol=2.4x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-08-21 11:25:00 | 787.94 | 787.36 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:30:00 | 789.32 | 790.71 | 0.00 | ORB-short ORB[789.60,795.28] vol=6.1x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-08-22 10:35:00 | 791.03 | 790.70 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:15:00 | 798.68 | 793.63 | 0.00 | ORB-long ORB[790.40,796.80] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-08-23 10:30:00 | 796.46 | 794.56 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 796.36 | 795.09 | 0.00 | ORB-long ORB[792.12,795.92] vol=2.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:55:00 | 799.03 | 795.85 | 0.00 | T1 1.5R @ 799.03 |
| Stop hit — per-position SL triggered | 2024-08-26 10:10:00 | 796.36 | 796.03 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:45:00 | 791.24 | 792.99 | 0.00 | ORB-short ORB[793.16,796.48] vol=3.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-08-27 11:05:00 | 792.66 | 792.89 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 767.72 | 772.74 | 0.00 | ORB-short ORB[773.92,779.20] vol=2.2x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 769.55 | 772.19 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:50:00 | 776.28 | 774.13 | 0.00 | ORB-long ORB[772.04,775.80] vol=1.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-08-30 10:05:00 | 773.91 | 774.28 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:45:00 | 769.68 | 772.37 | 0.00 | ORB-short ORB[773.64,776.92] vol=4.2x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 13:15:00 | 767.53 | 769.65 | 0.00 | T1 1.5R @ 767.53 |
| Stop hit — per-position SL triggered | 2024-09-05 13:35:00 | 769.68 | 769.46 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 761.48 | 763.50 | 0.00 | ORB-short ORB[763.28,769.80] vol=1.7x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:15:00 | 758.68 | 763.20 | 0.00 | T1 1.5R @ 758.68 |
| Stop hit — per-position SL triggered | 2024-09-06 10:20:00 | 761.48 | 763.11 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:45:00 | 756.80 | 752.90 | 0.00 | ORB-long ORB[751.16,756.36] vol=1.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 755.09 | 753.07 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:40:00 | 766.72 | 767.48 | 0.00 | ORB-short ORB[767.04,772.72] vol=2.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:25:00 | 763.84 | 766.54 | 0.00 | T1 1.5R @ 763.84 |
| Target hit | 2024-09-16 15:20:00 | 761.24 | 764.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 756.44 | 758.82 | 0.00 | ORB-short ORB[756.92,763.84] vol=1.9x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:15:00 | 753.67 | 757.13 | 0.00 | T1 1.5R @ 753.67 |
| Stop hit — per-position SL triggered | 2024-09-17 10:40:00 | 756.44 | 756.82 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 752.68 | 755.01 | 0.00 | ORB-short ORB[754.88,757.56] vol=1.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:15:00 | 750.32 | 753.57 | 0.00 | T1 1.5R @ 750.32 |
| Stop hit — per-position SL triggered | 2024-09-18 11:25:00 | 752.68 | 753.26 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 734.96 | 743.28 | 0.00 | ORB-short ORB[742.44,748.44] vol=1.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:45:00 | 731.13 | 741.14 | 0.00 | T1 1.5R @ 731.13 |
| Stop hit — per-position SL triggered | 2024-09-19 10:50:00 | 734.96 | 740.99 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-09-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:10:00 | 708.88 | 711.16 | 0.00 | ORB-short ORB[710.40,714.40] vol=1.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 710.81 | 711.13 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:40:00 | 713.28 | 710.73 | 0.00 | ORB-long ORB[708.44,711.76] vol=1.6x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:15:00 | 715.06 | 711.51 | 0.00 | T1 1.5R @ 715.06 |
| Stop hit — per-position SL triggered | 2024-09-26 11:35:00 | 713.28 | 711.86 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-09-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:10:00 | 727.16 | 721.98 | 0.00 | ORB-long ORB[713.60,723.84] vol=4.9x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-09-27 11:20:00 | 725.06 | 723.81 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:35:00 | 735.64 | 737.66 | 0.00 | ORB-short ORB[736.00,741.56] vol=1.6x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 737.64 | 737.39 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:05:00 | 723.20 | 727.35 | 0.00 | ORB-short ORB[723.80,732.48] vol=6.7x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-10-03 11:10:00 | 724.90 | 727.16 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 688.72 | 698.34 | 0.00 | ORB-short ORB[704.24,711.80] vol=4.1x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-10-07 10:50:00 | 691.89 | 697.15 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 09:35:00 | 711.88 | 713.74 | 0.00 | ORB-short ORB[712.04,717.56] vol=1.8x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-10-09 09:40:00 | 713.83 | 713.71 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:55:00 | 725.08 | 721.09 | 0.00 | ORB-long ORB[716.00,721.96] vol=1.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-10-10 10:15:00 | 722.44 | 721.67 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-10-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:50:00 | 718.68 | 713.92 | 0.00 | ORB-long ORB[706.12,713.16] vol=4.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-10-11 11:00:00 | 716.66 | 714.11 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:35:00 | 703.00 | 703.27 | 0.00 | ORB-short ORB[704.20,707.92] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-10-15 14:10:00 | 704.57 | 703.21 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 703.20 | 706.40 | 0.00 | ORB-short ORB[705.64,711.20] vol=2.1x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:05:00 | 700.78 | 704.39 | 0.00 | T1 1.5R @ 700.78 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 703.20 | 704.20 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 691.88 | 696.05 | 0.00 | ORB-short ORB[693.32,700.76] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 694.09 | 694.18 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-10-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:50:00 | 640.00 | 644.92 | 0.00 | ORB-short ORB[648.12,655.84] vol=2.7x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:00:00 | 636.58 | 642.98 | 0.00 | T1 1.5R @ 636.58 |
| Target hit | 2024-10-25 11:15:00 | 637.08 | 636.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — SELL (started 2024-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:40:00 | 629.68 | 635.44 | 0.00 | ORB-short ORB[633.04,640.92] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-10-28 09:45:00 | 632.97 | 634.81 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:55:00 | 665.80 | 670.01 | 0.00 | ORB-short ORB[666.96,676.20] vol=2.1x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-11-04 11:15:00 | 668.23 | 669.25 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-11-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:45:00 | 692.68 | 689.34 | 0.00 | ORB-long ORB[684.20,691.12] vol=1.8x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-11-07 09:55:00 | 690.29 | 689.68 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:40:00 | 637.60 | 644.01 | 0.00 | ORB-short ORB[643.20,651.20] vol=1.7x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 640.42 | 642.13 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:35:00 | 627.16 | 629.83 | 0.00 | ORB-short ORB[628.76,632.36] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-11-18 10:00:00 | 629.04 | 628.81 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:35:00 | 640.92 | 639.06 | 0.00 | ORB-long ORB[633.16,640.80] vol=1.7x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:55:00 | 644.22 | 640.79 | 0.00 | T1 1.5R @ 644.22 |
| Target hit | 2024-11-19 10:20:00 | 642.84 | 642.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 645.56 | 642.84 | 0.00 | ORB-long ORB[639.44,644.16] vol=1.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:00:00 | 648.12 | 645.00 | 0.00 | T1 1.5R @ 648.12 |
| Target hit | 2024-11-27 11:35:00 | 647.84 | 648.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2024-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:55:00 | 652.88 | 650.77 | 0.00 | ORB-long ORB[648.04,651.92] vol=1.7x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:00:00 | 654.90 | 651.24 | 0.00 | T1 1.5R @ 654.90 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 652.88 | 651.39 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:40:00 | 662.80 | 660.32 | 0.00 | ORB-long ORB[656.96,661.52] vol=1.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-11-29 10:05:00 | 660.73 | 661.10 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 675.96 | 677.06 | 0.00 | ORB-short ORB[676.08,679.92] vol=3.1x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:30:00 | 673.39 | 676.46 | 0.00 | T1 1.5R @ 673.39 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 675.96 | 676.16 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:40:00 | 684.24 | 685.82 | 0.00 | ORB-short ORB[685.64,691.16] vol=3.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-12-10 10:45:00 | 685.64 | 685.84 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:45:00 | 686.20 | 684.78 | 0.00 | ORB-long ORB[680.28,685.16] vol=2.7x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-12-11 10:55:00 | 684.68 | 684.79 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 652.84 | 654.36 | 0.00 | ORB-short ORB[654.40,659.48] vol=1.9x ATR=2.23 |
| Stop hit — per-position SL triggered | 2024-12-13 10:25:00 | 655.07 | 654.20 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 654.68 | 656.72 | 0.00 | ORB-short ORB[657.00,663.12] vol=1.5x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:10:00 | 652.71 | 655.53 | 0.00 | T1 1.5R @ 652.71 |
| Stop hit — per-position SL triggered | 2024-12-16 13:25:00 | 654.68 | 655.19 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:55:00 | 625.40 | 624.06 | 0.00 | ORB-long ORB[619.68,623.60] vol=1.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 624.02 | 624.37 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 630.76 | 625.92 | 0.00 | ORB-long ORB[622.08,628.16] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-12-30 10:55:00 | 628.62 | 626.34 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:35:00 | 616.08 | 619.93 | 0.00 | ORB-short ORB[620.96,629.56] vol=1.6x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:45:00 | 613.24 | 619.16 | 0.00 | T1 1.5R @ 613.24 |
| Target hit | 2025-01-06 14:50:00 | 611.84 | 610.66 | 0.00 | Trail-exit close>VWAP |

### Cycle 73 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:15:00 | 619.20 | 617.00 | 0.00 | ORB-long ORB[612.00,618.08] vol=5.4x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 13:25:00 | 622.02 | 618.26 | 0.00 | T1 1.5R @ 622.02 |
| Target hit | 2025-01-16 15:20:00 | 620.80 | 619.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2025-01-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:55:00 | 623.36 | 621.72 | 0.00 | ORB-long ORB[618.00,622.56] vol=4.9x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-01-17 10:00:00 | 621.89 | 622.93 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:55:00 | 613.96 | 616.87 | 0.00 | ORB-short ORB[617.36,621.88] vol=2.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 615.54 | 615.50 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-01-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:20:00 | 612.12 | 607.16 | 0.00 | ORB-long ORB[600.00,604.60] vol=2.4x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:55:00 | 614.82 | 609.05 | 0.00 | T1 1.5R @ 614.82 |
| Stop hit — per-position SL triggered | 2025-01-23 11:45:00 | 612.12 | 610.17 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 612.48 | 617.00 | 0.00 | ORB-short ORB[616.56,621.08] vol=1.5x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:00:00 | 609.91 | 615.09 | 0.00 | T1 1.5R @ 609.91 |
| Stop hit — per-position SL triggered | 2025-01-24 10:10:00 | 612.48 | 614.63 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-01-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:35:00 | 592.00 | 594.56 | 0.00 | ORB-short ORB[594.08,601.68] vol=1.7x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-01-27 10:40:00 | 594.16 | 594.52 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 598.92 | 596.85 | 0.00 | ORB-long ORB[592.80,597.28] vol=6.4x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:40:00 | 601.10 | 597.48 | 0.00 | T1 1.5R @ 601.10 |
| Target hit | 2025-01-30 14:05:00 | 600.96 | 603.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2025-02-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:25:00 | 594.36 | 592.33 | 0.00 | ORB-long ORB[587.12,591.92] vol=2.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-02-05 12:00:00 | 592.52 | 593.33 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-02-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:00:00 | 578.84 | 581.16 | 0.00 | ORB-short ORB[580.40,587.68] vol=2.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-02-07 10:05:00 | 580.44 | 581.11 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:55:00 | 573.84 | 575.49 | 0.00 | ORB-short ORB[577.12,581.04] vol=1.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:25:00 | 571.42 | 574.86 | 0.00 | T1 1.5R @ 571.42 |
| Target hit | 2025-02-10 15:20:00 | 569.56 | 571.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2025-02-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:45:00 | 546.64 | 553.04 | 0.00 | ORB-short ORB[551.92,559.80] vol=1.8x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:30:00 | 543.65 | 550.44 | 0.00 | T1 1.5R @ 543.65 |
| Stop hit — per-position SL triggered | 2025-02-14 11:50:00 | 546.64 | 549.85 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:45:00 | 557.52 | 555.47 | 0.00 | ORB-long ORB[550.52,556.80] vol=1.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-02-20 12:30:00 | 555.38 | 557.67 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:50:00 | 537.00 | 539.86 | 0.00 | ORB-short ORB[537.72,542.64] vol=2.5x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 538.58 | 539.30 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:35:00 | 537.68 | 536.54 | 0.00 | ORB-long ORB[529.24,537.28] vol=2.8x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:05:00 | 540.87 | 537.31 | 0.00 | T1 1.5R @ 540.87 |
| Stop hit — per-position SL triggered | 2025-03-07 10:20:00 | 537.68 | 537.45 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:50:00 | 521.28 | 518.93 | 0.00 | ORB-long ORB[515.20,520.28] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-03-12 10:00:00 | 519.38 | 519.16 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:15:00 | 549.20 | 543.31 | 0.00 | ORB-long ORB[540.68,545.24] vol=1.6x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:30:00 | 551.52 | 545.85 | 0.00 | T1 1.5R @ 551.52 |
| Stop hit — per-position SL triggered | 2025-03-21 13:35:00 | 549.20 | 548.73 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-03-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:05:00 | 563.76 | 560.26 | 0.00 | ORB-long ORB[555.96,561.16] vol=2.1x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-03-24 10:25:00 | 561.59 | 560.94 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 560.00 | 564.83 | 0.00 | ORB-short ORB[563.40,571.16] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 562.25 | 564.10 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 553.84 | 551.08 | 0.00 | ORB-long ORB[546.24,553.52] vol=2.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-03-27 12:30:00 | 552.09 | 552.39 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2025-04-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 10:00:00 | 555.36 | 561.75 | 0.00 | ORB-short ORB[562.12,569.84] vol=1.5x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-04-04 10:05:00 | 557.56 | 561.33 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:00:00 | 562.48 | 559.60 | 0.00 | ORB-long ORB[555.92,561.48] vol=1.7x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:10:00 | 565.06 | 560.72 | 0.00 | T1 1.5R @ 565.06 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 562.48 | 561.31 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 566.16 | 564.06 | 0.00 | ORB-long ORB[559.36,565.44] vol=4.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 564.52 | 564.47 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2025-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:20:00 | 563.84 | 568.38 | 0.00 | ORB-short ORB[568.08,573.20] vol=2.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-04-23 10:25:00 | 565.52 | 568.17 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:10:00 | 553.40 | 551.28 | 0.00 | ORB-long ORB[546.24,550.40] vol=2.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-05-05 11:35:00 | 552.04 | 551.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:40:00 | 829.60 | 2024-05-15 09:55:00 | 826.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-16 09:45:00 | 831.20 | 2024-05-16 09:55:00 | 828.14 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-22 09:40:00 | 860.44 | 2024-05-22 09:50:00 | 864.74 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-05-23 09:30:00 | 875.20 | 2024-05-23 09:45:00 | 871.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-24 10:55:00 | 894.80 | 2024-05-24 11:00:00 | 891.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-28 09:50:00 | 881.40 | 2024-05-28 10:10:00 | 875.76 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-28 09:50:00 | 881.40 | 2024-05-28 10:15:00 | 881.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 11:00:00 | 858.56 | 2024-05-30 11:30:00 | 861.12 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-31 09:45:00 | 852.68 | 2024-05-31 09:55:00 | 848.18 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-05-31 09:45:00 | 852.68 | 2024-05-31 13:20:00 | 850.32 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-06-12 09:45:00 | 885.76 | 2024-06-12 09:55:00 | 889.65 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-12 09:45:00 | 885.76 | 2024-06-12 15:20:00 | 911.84 | TARGET_HIT | 0.50 | 2.94% |
| BUY | retest1 | 2024-06-14 10:35:00 | 916.36 | 2024-06-14 10:50:00 | 913.58 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-02 09:50:00 | 824.00 | 2024-07-02 10:00:00 | 826.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-04 09:30:00 | 830.08 | 2024-07-04 09:40:00 | 834.07 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-04 09:30:00 | 830.08 | 2024-07-04 09:45:00 | 830.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 09:45:00 | 832.08 | 2024-07-05 09:50:00 | 829.36 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-08 11:00:00 | 842.00 | 2024-07-08 11:25:00 | 836.62 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-07-08 11:00:00 | 842.00 | 2024-07-08 15:20:00 | 833.80 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-07-10 10:20:00 | 818.40 | 2024-07-10 10:40:00 | 813.52 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-07-10 10:20:00 | 818.40 | 2024-07-10 10:45:00 | 818.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 11:15:00 | 840.84 | 2024-07-12 11:25:00 | 837.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-12 11:15:00 | 840.84 | 2024-07-12 12:45:00 | 840.84 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-16 11:05:00 | 843.24 | 2024-07-16 13:40:00 | 845.19 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-18 09:45:00 | 836.44 | 2024-07-18 09:55:00 | 839.48 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-25 09:30:00 | 822.52 | 2024-07-25 09:35:00 | 818.91 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-26 11:05:00 | 819.36 | 2024-07-26 11:30:00 | 817.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-29 11:05:00 | 843.12 | 2024-07-29 12:00:00 | 840.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-30 10:20:00 | 841.04 | 2024-07-30 12:00:00 | 838.16 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-31 10:40:00 | 835.76 | 2024-07-31 10:45:00 | 837.64 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-01 11:10:00 | 829.96 | 2024-08-01 11:40:00 | 827.25 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-01 11:10:00 | 829.96 | 2024-08-01 13:00:00 | 829.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-12 09:50:00 | 768.24 | 2024-08-12 10:05:00 | 771.17 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-13 11:10:00 | 778.36 | 2024-08-13 12:20:00 | 775.50 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-13 11:10:00 | 778.36 | 2024-08-13 15:20:00 | 765.20 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2024-08-14 10:55:00 | 755.60 | 2024-08-14 11:10:00 | 758.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-16 10:05:00 | 768.72 | 2024-08-16 10:30:00 | 766.66 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-21 11:15:00 | 789.64 | 2024-08-21 11:25:00 | 787.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-22 10:30:00 | 789.32 | 2024-08-22 10:35:00 | 791.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-23 10:15:00 | 798.68 | 2024-08-23 10:30:00 | 796.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-26 09:45:00 | 796.36 | 2024-08-26 09:55:00 | 799.03 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-26 09:45:00 | 796.36 | 2024-08-26 10:10:00 | 796.36 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 10:45:00 | 791.24 | 2024-08-27 11:05:00 | 792.66 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-29 10:55:00 | 767.72 | 2024-08-29 11:10:00 | 769.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-30 09:50:00 | 776.28 | 2024-08-30 10:05:00 | 773.91 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-05 10:45:00 | 769.68 | 2024-09-05 13:15:00 | 767.53 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-05 10:45:00 | 769.68 | 2024-09-05 13:35:00 | 769.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:05:00 | 761.48 | 2024-09-06 10:15:00 | 758.68 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-06 10:05:00 | 761.48 | 2024-09-06 10:20:00 | 761.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 09:45:00 | 756.80 | 2024-09-11 09:55:00 | 755.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-16 10:40:00 | 766.72 | 2024-09-16 11:25:00 | 763.84 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-16 10:40:00 | 766.72 | 2024-09-16 15:20:00 | 761.24 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-09-17 09:40:00 | 756.44 | 2024-09-17 10:15:00 | 753.67 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-17 09:40:00 | 756.44 | 2024-09-17 10:40:00 | 756.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:30:00 | 752.68 | 2024-09-18 11:15:00 | 750.32 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-09-18 10:30:00 | 752.68 | 2024-09-18 11:25:00 | 752.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:15:00 | 734.96 | 2024-09-19 10:45:00 | 731.13 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-19 10:15:00 | 734.96 | 2024-09-19 10:50:00 | 734.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 10:10:00 | 708.88 | 2024-09-25 10:15:00 | 710.81 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-26 10:40:00 | 713.28 | 2024-09-26 11:15:00 | 715.06 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-09-26 10:40:00 | 713.28 | 2024-09-26 11:35:00 | 713.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 11:10:00 | 727.16 | 2024-09-27 11:20:00 | 725.06 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-01 10:35:00 | 735.64 | 2024-10-01 10:55:00 | 737.64 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-03 11:05:00 | 723.20 | 2024-10-03 11:10:00 | 724.90 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-07 10:45:00 | 688.72 | 2024-10-07 10:50:00 | 691.89 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-09 09:35:00 | 711.88 | 2024-10-09 09:40:00 | 713.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-10 09:55:00 | 725.08 | 2024-10-10 10:15:00 | 722.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-11 10:50:00 | 718.68 | 2024-10-11 11:00:00 | 716.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-15 10:35:00 | 703.00 | 2024-10-15 14:10:00 | 704.57 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-17 09:35:00 | 703.20 | 2024-10-17 10:05:00 | 700.78 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-17 09:35:00 | 703.20 | 2024-10-17 10:15:00 | 703.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:30:00 | 691.88 | 2024-10-21 09:55:00 | 694.09 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-25 09:50:00 | 640.00 | 2024-10-25 10:00:00 | 636.58 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-25 09:50:00 | 640.00 | 2024-10-25 11:15:00 | 637.08 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-28 09:40:00 | 629.68 | 2024-10-28 09:45:00 | 632.97 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-11-04 10:55:00 | 665.80 | 2024-11-04 11:15:00 | 668.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-07 09:45:00 | 692.68 | 2024-11-07 09:55:00 | 690.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-13 09:40:00 | 637.60 | 2024-11-13 09:50:00 | 640.42 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-18 09:35:00 | 627.16 | 2024-11-18 10:00:00 | 629.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-19 09:35:00 | 640.92 | 2024-11-19 09:55:00 | 644.22 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-19 09:35:00 | 640.92 | 2024-11-19 10:20:00 | 642.84 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-11-27 09:35:00 | 645.56 | 2024-11-27 10:00:00 | 648.12 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-27 09:35:00 | 645.56 | 2024-11-27 11:35:00 | 647.84 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-11-28 09:55:00 | 652.88 | 2024-11-28 10:00:00 | 654.90 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-11-28 09:55:00 | 652.88 | 2024-11-28 10:05:00 | 652.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 09:40:00 | 662.80 | 2024-11-29 10:05:00 | 660.73 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-05 10:55:00 | 675.96 | 2024-12-05 11:30:00 | 673.39 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-05 10:55:00 | 675.96 | 2024-12-05 12:05:00 | 675.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 10:40:00 | 684.24 | 2024-12-10 10:45:00 | 685.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-11 10:45:00 | 686.20 | 2024-12-11 10:55:00 | 684.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-13 10:15:00 | 652.84 | 2024-12-13 10:25:00 | 655.07 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-16 11:00:00 | 654.68 | 2024-12-16 12:10:00 | 652.71 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-16 11:00:00 | 654.68 | 2024-12-16 13:25:00 | 654.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:55:00 | 625.40 | 2024-12-24 12:15:00 | 624.02 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-30 10:45:00 | 630.76 | 2024-12-30 10:55:00 | 628.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-06 10:35:00 | 616.08 | 2025-01-06 10:45:00 | 613.24 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-06 10:35:00 | 616.08 | 2025-01-06 14:50:00 | 611.84 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-16 10:15:00 | 619.20 | 2025-01-16 13:25:00 | 622.02 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-16 10:15:00 | 619.20 | 2025-01-16 15:20:00 | 620.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-01-17 09:55:00 | 623.36 | 2025-01-17 10:00:00 | 621.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-21 10:55:00 | 613.96 | 2025-01-21 11:15:00 | 615.54 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-23 10:20:00 | 612.12 | 2025-01-23 10:55:00 | 614.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-23 10:20:00 | 612.12 | 2025-01-23 11:45:00 | 612.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:45:00 | 612.48 | 2025-01-24 10:00:00 | 609.91 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-24 09:45:00 | 612.48 | 2025-01-24 10:10:00 | 612.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:35:00 | 592.00 | 2025-01-27 10:40:00 | 594.16 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-30 09:30:00 | 598.92 | 2025-01-30 09:40:00 | 601.10 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-30 09:30:00 | 598.92 | 2025-01-30 14:05:00 | 600.96 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-02-05 10:25:00 | 594.36 | 2025-02-05 12:00:00 | 592.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-07 10:00:00 | 578.84 | 2025-02-07 10:05:00 | 580.44 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-10 10:55:00 | 573.84 | 2025-02-10 11:25:00 | 571.42 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-10 10:55:00 | 573.84 | 2025-02-10 15:20:00 | 569.56 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2025-02-14 10:45:00 | 546.64 | 2025-02-14 11:30:00 | 543.65 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-02-14 10:45:00 | 546.64 | 2025-02-14 11:50:00 | 546.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 09:45:00 | 557.52 | 2025-02-20 12:30:00 | 555.38 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-27 09:50:00 | 537.00 | 2025-02-27 10:15:00 | 538.58 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-07 09:35:00 | 537.68 | 2025-03-07 10:05:00 | 540.87 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-07 09:35:00 | 537.68 | 2025-03-07 10:20:00 | 537.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 09:50:00 | 521.28 | 2025-03-12 10:00:00 | 519.38 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-21 10:15:00 | 549.20 | 2025-03-21 11:30:00 | 551.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-21 10:15:00 | 549.20 | 2025-03-21 13:35:00 | 549.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 10:05:00 | 563.76 | 2025-03-24 10:25:00 | 561.59 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-26 09:40:00 | 560.00 | 2025-03-26 09:55:00 | 562.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-27 10:45:00 | 553.84 | 2025-03-27 12:30:00 | 552.09 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-04 10:00:00 | 555.36 | 2025-04-04 10:05:00 | 557.56 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-16 10:00:00 | 562.48 | 2025-04-16 10:10:00 | 565.06 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-16 10:00:00 | 562.48 | 2025-04-16 10:15:00 | 562.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:30:00 | 566.16 | 2025-04-21 09:45:00 | 564.52 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 10:20:00 | 563.84 | 2025-04-23 10:25:00 | 565.52 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-05 11:10:00 | 553.40 | 2025-05-05 11:35:00 | 552.04 | STOP_HIT | 1.00 | -0.25% |
